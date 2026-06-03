from __future__ import annotations

"""3090 侧 SpecEdge V1 smoke runner。

默认跑 target_only、linear、tree 三个方法，并把输出分到
experiments/<run_id>/{target_only,linear,tree}/，用于验证 tree speculative
decoding 是否和 target-only greedy 输出一致。
"""

import argparse
import csv
import json
import subprocess
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from specplatform.config import as_list, get_in, load_config
from specplatform.core import DraftBudget, PhaseEvent, RuntimeContext
from specplatform.draft import (
    DraftSpeedProfile,
    DraftWorker,
    DraftWorkerRegistry,
    GreedyDraftRunner,
    TopKTreeDraftRunner,
    draft_worker_configs_from_settings,
)
from specplatform.methods import (
    DiPSDPlanningPolicy,
    GreedyPrefixAcceptancePolicy,
    LinearCandidateStrategy,
    OfficialSpecEdgeDraftState,
    SLEDAsyncDraftPolicy,
    SLEDAsyncReconcilePolicy,
    SLEDDynamicCandidateStrategy,
    SLEDPlanningPolicy,
    SpecEdgeOfficialAcceptancePolicy,
    SpecEdgeOfficialCandidateStrategy,
    SpecEdgeOfficialProactiveDraftPolicy,
    SpecEdgeOfficialReconcilePolicy,
    SpecEdgePipelinePlanningPolicy,
    SpecEdgeProactiveDraftPolicy,
    SpecEdgeReconcilePolicy,
    SpecEdgeTreeAcceptancePolicy,
    SpecEdgeTreeCandidateStrategy,
)
from specplatform.methods.dip_sd.calibration import calibration_from_events, recommended_method_config_from_profile
from specplatform.methods.dip_sd.model import DiPSDModelConfig
from specplatform.metrics import (
    EventLogger,
    write_phase_events_csv,
    write_phase_summary_csv,
    write_request_results_json,
    write_timing_audit,
    write_timing_charts,
    write_tree_snapshots_jsonl,
)
from specplatform.model import TransformersCausalLMRunner, load_causal_lm_runner
from specplatform.runtime import (
    AsyncPipelineRuntimeEngine,
    DistributedBatchPipelineRuntimeEngine,
    GenerationSession,
    RuntimeEngine,
    RuntimeRequestResult,
    RuntimeRunResult,
)
from specplatform.schedulers import RoundRobinRequestScheduler
from specplatform.timing import TimingRecorder, event_from_span
from specplatform.verification import (
    HttpGreedyGeneratorClient,
    HttpLinearVerifierClient,
    HttpLinearVerifierPoolClient,
    HttpTreeVerifierClient,
    TransportProfile,
)


@dataclass(frozen=True)
class PromptSpec:
    """一次实验输入 prompt。"""

    request_id: str
    prompt: str
    prompt_ids: list[int]


@dataclass(frozen=True)
class DraftWarmupResult:
    """One completed draft warmup operation."""

    draft_type: str
    worker_id: str
    runner_id: str | None
    start_ns: int
    end_ns: int
    produced_count: int
    reset_after_warmup: bool


def build_parser() -> argparse.ArgumentParser:
    """解析 smoke 参数。"""
    parser = argparse.ArgumentParser(description="Run SpecEdge target_only/linear/tree smoke on 3090 -> A100.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--draft-model-path", default=None)
    parser.add_argument("--target-url", default=None)
    parser.add_argument("--target-urls", default=None)
    parser.add_argument("--target-required-replicas", type=int, default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--use-sample-prompts", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--draft-tokens", type=int, default=None)
    parser.add_argument("--tree-max-depth", type=int, default=None)
    parser.add_argument("--tree-branch-width", type=int, default=None)
    parser.add_argument("--tree-max-budget", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--torch-dtype", default=None, choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--draft-backend", default=None)
    parser.add_argument("--draft-worker-count", type=int, default=None)
    parser.add_argument("--no-backend-fallback", action="store_true")
    parser.add_argument("--no-zero-draft-fallback", action="store_true")
    parser.add_argument("--sled-confidence-threshold", type=float, default=None)
    parser.add_argument("--dip-sd-calibration-profile", default=None)
    parser.add_argument("--methods", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--plot-formats", default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--sync-dest", default=None)
    parser.add_argument("--transport-uplink-mbps", type=float, default=None)
    parser.add_argument("--transport-downlink-mbps", type=float, default=None)
    parser.add_argument("--transport-rtt-ms", type=float, default=None)
    return parser


def main() -> None:
    """运行三基线 smoke，并写 artifacts。"""
    args = build_parser().parse_args()
    config = load_config(args.config) if args.config else {}
    settings = _settings(args, config)
    methods = [str(method) for method in as_list(settings["methods"], item_type=str)]
    method_set = set(methods)
    if settings.get("sled_strict") and (method_set & {"sled", "sled_async"}):
        if "sled" in methods:
            raise ValueError("Strict SLED paper reproduction uses sled_async only; remove sled from run.methods.")
        if "sled_async" not in methods:
            raise ValueError("Strict SLED paper reproduction requires sled_async in run.methods.")
    if settings.get("target_require_graph"):
        _assert_target_graph_backend(settings)
    output_root = Path(settings["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    setup_recorder = TimingRecorder()
    with setup_recorder.span(
        phase="setup.load_draft_model",
        method="specedge_smoke",
        plan_id=f"{settings['run_id']}:setup",
        run_id=settings["run_id"],
        shared=True,
        metadata={
            "draft_model_path": settings["draft_model_path"],
            "device": settings["device"],
            "torch_dtype": settings["torch_dtype"],
            "draft_backend": settings["draft_backend"],
            "draft_worker_count": settings["draft_worker_count"],
            "allow_backend_fallback": settings["allow_backend_fallback"],
        },
    ) as setup_span:
        draft_registry = _load_draft_registry(settings)
        setup_span.metadata["draft_registry"] = draft_registry.to_metadata()
        registry_audit = _audit_draft_registry(settings, draft_registry)
        setup_span.metadata["draft_registry_audit"] = registry_audit
        if registry_audit["violations"] and registry_audit["enforce"]:
            raise ValueError(
                "Draft registry audit failed: "
                + "; ".join(str(item) for item in registry_audit["violations"])
            )
        draft_model = draft_registry.first_model()
        setup_span.metadata["backend_capabilities"] = draft_model.backend_capabilities().to_dict()
    setup_event = event_from_span(
        setup_span,
        event_id_factory=setup_recorder.next_event_id,
        span_kind="setup",
        attribution="system",
        metadata=dict(setup_span.metadata),
    )

    prompts = _prompt_specs(settings, draft_model)
    setup_events = [setup_event]
    warmup_events = _warm_draft_registry(
        settings=settings,
        draft_registry=draft_registry,
        prompts=prompts,
        methods=methods,
        recorder=setup_recorder,
    )
    setup_events.extend(warmup_events)
    observed_profiles = _apply_observed_warmup_profiles(draft_registry, warmup_events)
    if observed_profiles:
        setup_event.metadata["observed_draft_speed_profiles"] = observed_profiles
        setup_event.metadata["draft_registry_after_warmup"] = draft_registry.to_metadata()
    eos_token_ids = _tokenizer_eos_ids(draft_model.tokenizer)
    method_outputs: dict[str, dict[str, list[int]]] = {}
    method_results: dict[str, RuntimeRunResult] = {}

    if "target_only" in methods:
        target_result = _run_target_only(settings, prompts, eos_token_ids, setup_events)
        method_outputs["target_only"] = _outputs_by_request(target_result)
        method_results["target_only"] = target_result
        _write_method_artifacts(
            result=target_result,
            output_dir=output_root / "target_only",
            render_plots=not settings["no_plots"],
            plot_formats=settings["plot_formats"],
            metadata=_metadata(settings, "target_only", prompts, target_result),
        )

    if "linear" in methods:
        transport_profile = _transport_profile(settings)
        linear_result = _run_runtime_method(
            settings=settings,
            method="linear",
            prompts=prompts,
            eos_token_ids=eos_token_ids,
            setup_events=setup_events,
            draft_runners=draft_registry.runners_for("greedy"),
            candidate_strategy=LinearCandidateStrategy(),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            verifier=_linear_verifier(settings, transport_profile),
            budget=DraftBudget(max_tokens=settings["draft_tokens"]),
            method_config={},
        )
        method_outputs["linear"] = _outputs_by_request(linear_result)
        method_results["linear"] = linear_result
        _write_method_artifacts(
            result=linear_result,
            output_dir=output_root / "linear",
            render_plots=not settings["no_plots"],
            plot_formats=settings["plot_formats"],
            metadata=_metadata(settings, "linear", prompts, linear_result),
        )

    if "tree" in methods or "tree_stop_wait" in methods:
        method_name = "tree_stop_wait" if "tree_stop_wait" in methods else "tree"
        transport_profile = _transport_profile(settings)
        tree_result = _run_runtime_method(
            settings=settings,
            method=method_name,
            prompts=prompts,
            eos_token_ids=eos_token_ids,
            setup_events=setup_events,
            draft_runners=draft_registry.runners_for("tree"),
            candidate_strategy=SpecEdgeTreeCandidateStrategy(
                default_max_budget=settings["tree_max_budget"],
                default_max_branch_width=settings["tree_branch_width"],
            ),
            acceptance_policy=SpecEdgeTreeAcceptancePolicy(),
            verifier=HttpTreeVerifierClient(
                base_url=settings["target_url"],
                transport_profile=transport_profile,
                max_batch_items=settings["tree_max_verify_batch_items"],
            ),
            budget=DraftBudget(max_tokens=settings["tree_max_depth"], max_branches=settings["tree_branch_width"]),
            method_config={
                "max_budget": settings["tree_max_budget"],
                "max_branch_width": settings["tree_branch_width"],
            },
        )
        method_outputs[method_name] = _outputs_by_request(tree_result)
        method_results[method_name] = tree_result
        _write_method_artifacts(
            result=tree_result,
            output_dir=output_root / method_name,
            render_plots=not settings["no_plots"],
            plot_formats=settings["plot_formats"],
            metadata=_metadata(settings, method_name, prompts, tree_result),
        )

    if "specedge_pipeline" in methods:
        transport_profile = _transport_profile(settings)
        pipeline_result = _run_pipeline_method(
            settings=settings,
            method="specedge_pipeline",
            prompts=prompts,
            eos_token_ids=eos_token_ids,
            setup_events=setup_events,
            draft_runners=draft_registry.runners_for("tree"),
            verifier=HttpTreeVerifierClient(
                base_url=settings["target_url"],
                transport_profile=transport_profile,
                max_batch_items=settings["tree_max_verify_batch_items"],
            ),
            budget=DraftBudget(max_tokens=settings["tree_max_depth"], max_branches=settings["tree_branch_width"]),
            method_config={
                "min_depth": settings["pipeline_min_depth"],
                "max_depth": settings["pipeline_max_depth"],
                "max_branch_width": settings["tree_branch_width"],
                "max_budget": settings["tree_max_budget"],
                "proactive_max_depth": settings["pipeline_proactive_depth"],
                "proactive_branch_width": settings["tree_branch_width"],
                "proactive_max_budget": settings["tree_max_budget"],
                "disable_bonus": False,
                "force_root_guard": False,
                "proactive_force_root_guard": True,
            },
        )
        method_outputs["specedge_pipeline"] = _outputs_by_request(pipeline_result)
        method_results["specedge_pipeline"] = pipeline_result
        _write_method_artifacts(
            result=pipeline_result,
            output_dir=output_root / "specedge_pipeline",
            render_plots=not settings["no_plots"],
            plot_formats=settings["plot_formats"],
            metadata=_metadata(settings, "specedge_pipeline", prompts, pipeline_result),
        )

    if "dip_sd" in methods:
        transport_profile = _transport_profile(settings)
        dip_sd_result = _run_distributed_pipeline_method(
            settings=settings,
            method="dip_sd",
            prompts=prompts,
            eos_token_ids=eos_token_ids,
            setup_events=setup_events,
            draft_runners=draft_registry.runners_for("greedy"),
            candidate_strategy=LinearCandidateStrategy(proposal_prefix="dip-sd-linear"),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            verifier=_linear_verifier(settings, transport_profile),
            budget=DraftBudget(max_tokens=settings["dip_sd_max_draft_length"], max_branches=1),
            method_config={
                "max_budget": settings["tree_max_budget"],
                "max_branch_width": 1,
                "dip_sd_solver": settings["dip_sd_solver"],
                "dip_sd_min_batch_count": settings["dip_sd_min_batch_count"],
                "dip_sd_max_batch_count": settings["dip_sd_max_batch_count"],
                "dip_sd_single_batch_small_request_threshold": settings[
                    "dip_sd_single_batch_small_request_threshold"
                ],
                "dip_sd_initial_draft_length": settings["dip_sd_initial_draft_length"],
                "dip_sd_max_draft_length": settings["dip_sd_max_draft_length"],
                "dip_sd_alpha": settings["dip_sd_alpha"],
                "dip_sd_acceptance": settings["dip_sd_acceptance"],
                "dip_sd_comm_latency_ms": settings["dip_sd_comm_latency_ms"],
                "dip_sd_draft_c": settings["dip_sd_draft_c"],
                "dip_sd_draft_beta": settings["dip_sd_draft_beta"],
                "dip_sd_verify_c": settings["dip_sd_verify_c"],
                "dip_sd_verify_beta": settings["dip_sd_verify_beta"],
                "dip_sd_memory_cap_bytes": settings["dip_sd_memory_cap_bytes"],
                "dip_sd_plan_cache_enabled": settings["dip_sd_plan_cache_enabled"],
                "dip_sd_steady_state_enabled": settings["dip_sd_steady_state_enabled"],
                "dip_sd_worker_assignment_strategy": settings["dip_sd_worker_assignment_strategy"],
                "dip_sd_prefetch_sticky_worker_enabled": settings[
                    "dip_sd_prefetch_sticky_worker_enabled"
                ],
                "dip_sd_use_measured_worker_latency": settings["dip_sd_use_measured_worker_latency"],
                "dip_sd_adaptive_draft_length_enabled": settings["dip_sd_adaptive_draft_length_enabled"],
                "dip_sd_min_draft_length": settings["dip_sd_min_draft_length"],
                "dip_sd_adaptive_length_target_acceptance": settings[
                    "dip_sd_adaptive_length_target_acceptance"
                ],
                "dip_sd_adaptive_length_min_factor": settings[
                    "dip_sd_adaptive_length_min_factor"
                ],
                "dip_sd_ready_aware_rebatch_enabled": settings["dip_sd_ready_aware_rebatch_enabled"],
                "dip_sd_ready_aware_rebatch_min_spread_ms": settings[
                    "dip_sd_ready_aware_rebatch_min_spread_ms"
                ],
                "dip_sd_slow_worker_length_threshold": settings["dip_sd_slow_worker_length_threshold"],
                "dip_sd_slow_worker_length_multiplier": settings["dip_sd_slow_worker_length_multiplier"],
                "dip_sd_prefetch_adaptive_length_enabled": settings[
                    "dip_sd_prefetch_adaptive_length_enabled"
                ],
                "dip_sd_prefetch_acceptance_lookahead_tokens": settings[
                    "dip_sd_prefetch_acceptance_lookahead_tokens"
                ],
                "dip_sd_prefetch_min_tokens": settings["dip_sd_prefetch_min_tokens"],
                "dip_sd_prefetch_max_tokens": settings["dip_sd_prefetch_max_tokens"],
                "dip_sd_prefetch_use_source_budget_floor": settings[
                    "dip_sd_prefetch_use_source_budget_floor"
                ],
                "dip_sd_no_online_solver": settings["dip_sd_no_online_solver"],
                "dip_sd_no_online_solver_fallback": settings["dip_sd_no_online_solver_fallback"],
                "dip_sd_offline_plan_table": settings["dip_sd_offline_plan_table"],
                "dip_sd_offline_plan_prefix_bucket": settings["dip_sd_offline_plan_prefix_bucket"],
                "dip_sd_acceptance_feedback_enabled": settings["dip_sd_acceptance_feedback_enabled"],
                "dip_sd_acceptance_feedback_min_draft_tokens": settings[
                    "dip_sd_acceptance_feedback_min_draft_tokens"
                ],
                "dip_sd_acceptance_feedback_prior_weight": settings[
                    "dip_sd_acceptance_feedback_prior_weight"
                ],
                "dip_sd_acceptance_cache_bucket": settings["dip_sd_acceptance_cache_bucket"],
                "dip_sd_calibration_profile": settings["dip_sd_calibration_profile"],
                "dip_sd_calibration_enabled": settings["dip_sd_calibration_enabled"],
                "dip_sd_calibration_applied": settings["dip_sd_calibration_applied"],
                "dip_sd_calibration_overrides": settings["dip_sd_calibration_overrides"],
            },
            planning_policy=DiPSDPlanningPolicy(
                max_draft_length=settings["dip_sd_max_draft_length"],
                initial_draft_length=settings["dip_sd_initial_draft_length"],
                min_batch_count=settings["dip_sd_min_batch_count"],
                max_batch_count=settings["dip_sd_max_batch_count"],
                single_batch_small_request_threshold=settings["dip_sd_single_batch_small_request_threshold"],
                default_acceptance=settings["dip_sd_alpha"],
                default_comm_latency_ms=settings["dip_sd_comm_latency_ms"],
                default_draft_c=settings["dip_sd_draft_c"],
                default_draft_beta=settings["dip_sd_draft_beta"],
                solver_mode=settings["dip_sd_solver"],
                plan_cache_enabled=settings["dip_sd_plan_cache_enabled"],
            ),
        )
        method_outputs["dip_sd"] = _outputs_by_request(dip_sd_result)
        method_results["dip_sd"] = dip_sd_result
        _write_method_artifacts(
            result=dip_sd_result,
            output_dir=output_root / "dip_sd",
            render_plots=not settings["no_plots"],
            plot_formats=settings["plot_formats"],
            metadata=_metadata(settings, "dip_sd", prompts, dip_sd_result),
        )

    if "sled" in methods:
        transport_profile = _transport_profile(settings)
        sled_result = _run_runtime_method(
            settings=settings,
            method="sled",
            prompts=prompts,
            eos_token_ids=eos_token_ids,
            setup_events=setup_events,
            draft_runners=draft_registry.runners_for("greedy"),
            candidate_strategy=SLEDDynamicCandidateStrategy(
                proposal_prefix="sled-linear",
                confidence_threshold=settings["sled_confidence_threshold"],
            ),
            acceptance_policy=GreedyPrefixAcceptancePolicy(),
            verifier=_linear_verifier(settings, transport_profile),
            budget=DraftBudget(max_tokens=settings["sled_max_speculation_tokens"], max_branches=1),
            method_config={
                "max_budget": settings["tree_max_budget"],
                "max_branch_width": 1,
                "sled_batch_size": settings["sled_batch_size"],
                "sled_confidence_threshold": settings["sled_confidence_threshold"],
            },
            planning_policy=SLEDPlanningPolicy(
                min_depth=settings["pipeline_min_depth"],
                max_depth=settings["pipeline_max_depth"],
                max_speculation_tokens=settings["sled_max_speculation_tokens"],
                target_batch_size=settings["sled_batch_size"],
                confidence_threshold=settings["sled_confidence_threshold"],
            ),
        )
        method_outputs["sled"] = _outputs_by_request(sled_result)
        method_results["sled"] = sled_result
        _write_method_artifacts(
            result=sled_result,
            output_dir=output_root / "sled",
            render_plots=not settings["no_plots"],
            plot_formats=settings["plot_formats"],
            metadata=_metadata(settings, "sled", prompts, sled_result),
        )

    if "sled_async" in methods:
        transport_profile = _transport_profile(settings)
        sled_async_result = _run_sled_async_method(
            settings=settings,
            method="sled_async",
            prompts=prompts,
            eos_token_ids=eos_token_ids,
            setup_events=setup_events,
            draft_runners=draft_registry.runners_for("greedy"),
            verifier=_linear_verifier(settings, transport_profile),
        )
        method_outputs["sled_async"] = _outputs_by_request(sled_async_result)
        method_results["sled_async"] = sled_async_result
        _write_method_artifacts(
            result=sled_async_result,
            output_dir=output_root / "sled_async",
            render_plots=not settings["no_plots"],
            plot_formats=settings["plot_formats"],
            metadata=_metadata(settings, "sled_async", prompts, sled_async_result),
        )

    summary = _build_combined_summary(method_outputs, method_results=method_results)
    (output_root / "combined_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if settings["sync_dest"]:
        _sync_output_dir(output_root, settings["sync_dest"])

    print("run_id:", settings["run_id"])
    print("output_dir:", output_root)
    print("methods:", ",".join(methods))
    print("combined_summary:", summary)
    if settings["sync_dest"]:
        print("synced_to:", settings["sync_dest"])


def _settings(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    """合并 YAML 和 CLI 参数；CLI 非空值覆盖 YAML。"""
    run_id = _value(args.run_id, config, ("run", "id"), "3090_a100_specedge_smoke")
    target_urls = _target_urls(args, config)
    raw_workers = get_in(config, ("draft", "workers"), []) or []
    worker_count = len(raw_workers) if raw_workers else max(1, int(_value(args.draft_worker_count, config, ("draft", "worker_count"), 1)))
    sled_strict = bool(get_in(config, ("sled", "strict"), False))
    dip_sd_calibration_profile = _value(
        args.dip_sd_calibration_profile,
        config,
        ("dip_sd", "calibration_profile"),
        None,
    )
    dip_sd_calibration_config_path = None if args.dip_sd_calibration_profile is not None else args.config
    dip_sd_calibration_enabled = bool(
        args.dip_sd_calibration_profile is not None
        or get_in(config, ("dip_sd", "calibration_enabled"), dip_sd_calibration_profile is not None)
    )
    dip_sd_calibration_overrides = _dip_sd_calibration_overrides(
        dip_sd_calibration_profile,
        enabled=dip_sd_calibration_enabled,
        config_path=dip_sd_calibration_config_path,
    )
    dip_sd_offline_plan_table = _dip_sd_offline_plan_table_config(config, config_path=args.config)
    base_dip_sd_draft_c = float(get_in(config, ("dip_sd", "draft_c"), 4.0305e-11))
    base_dip_sd_draft_beta = float(get_in(config, ("dip_sd", "draft_beta"), 33.8151))
    base_dip_sd_verify_c = float(get_in(config, ("dip_sd", "verify_c"), 1.2077e-11))
    base_dip_sd_verify_beta = float(get_in(config, ("dip_sd", "verify_beta"), 95.1074))
    return {
        "run_id": run_id,
        "draft_model_path": _value(
            args.draft_model_path,
            config,
            ("models", "draft"),
            "/home/chajiahao/data/hf_models/Qwen3-1.7B",
        ),
        "target_url": target_urls[0],
        "target_urls": target_urls,
        "target_require_graph": bool(
            get_in(config, ("target", "require_graph"), get_in(config, ("sled", "require_graph_target"), False))
        ),
        "target_required_backend": str(get_in(config, ("target", "required_backend"), "qwen3_graph")),
        "target_required_replicas": int(
            _value(args.target_required_replicas, config, ("target", "required_replicas"), len(target_urls)) or 0
        ),
        "prompt": _value(args.prompt, config, ("data", "prompt"), "介绍一下 speculative decoding"),
        "prompts_file": _value(args.prompts_file, config, ("data", "sample_prompts"), None),
        "sample_count": int(_value(args.sample_count, config, ("data", "sample_count"), 8)),
        "use_sample_prompts": bool(args.use_sample_prompts or get_in(config, ("data", "use_sample_prompts"), False)),
        "max_new_tokens": int(_value(args.max_new_tokens, config, ("generation", "max_new_tokens"), 16)),
        "draft_tokens": int(_value(args.draft_tokens, config, ("linear", "draft_tokens"), 4)),
        "tree_max_depth": int(_value(args.tree_max_depth, config, ("tree", "max_depth"), 2)),
        "tree_branch_width": int(_value(args.tree_branch_width, config, ("tree", "branch_width"), 2)),
        "tree_max_budget": int(_value(args.tree_max_budget, config, ("tree", "max_budget"), 8)),
        "tree_max_verify_batch_items": int(get_in(config, ("tree", "max_verify_batch_items"), 8) or 0) or None,
        "pipeline_min_depth": int(get_in(config, ("pipeline", "min_depth"), 1)),
        "pipeline_max_depth": int(get_in(config, ("pipeline", "max_depth"), 8)),
        "pipeline_proactive_depth": int(get_in(config, ("pipeline", "proactive_depth"), 2)),
        "specedge_official": bool(get_in(config, ("specedge", "official"), True)),
        "dip_sd_solver": str(get_in(config, ("dip_sd", "solver"), "enumerate")),
        "dip_sd_min_batch_count": int(get_in(config, ("dip_sd", "min_batch_count"), 2) or 2),
        "dip_sd_max_batch_count": int(get_in(config, ("dip_sd", "max_batch_count"), 0) or 0) or None,
        "dip_sd_single_batch_small_request_threshold": int(
            get_in(config, ("dip_sd", "single_batch_small_request_threshold"), 2) or 0
        ),
        "dip_sd_initial_draft_length": int(get_in(config, ("dip_sd", "initial_draft_length"), 7) or 7),
        "dip_sd_max_draft_length": int(get_in(config, ("dip_sd", "max_draft_length"), get_in(config, ("tree", "max_depth"), 8)) or 8),
        "sled_batch_size": int(get_in(config, ("sled", "batch_size"), get_in(config, ("pipeline", "batch_size"), 0)) or 0),
        "sled_max_speculation_tokens": int(get_in(config, ("sled", "max_speculation_tokens"), get_in(config, ("tree", "max_depth"), 2))),
        "sled_confidence_threshold": float(
            _value(args.sled_confidence_threshold, config, ("sled", "confidence_threshold"), 0.5)
        ),
        "sled_async_proactive_tokens": int(get_in(config, ("sled", "async", "proactive_tokens"), get_in(config, ("sled", "max_speculation_tokens"), 2))),
        "sled_verify_timeout_ms": get_in(config, ("sled", "timeout_ms"), get_in(config, ("sled", "verify_timeout_ms"), None)),
        "sled_retry_count": int(get_in(config, ("sled", "retry_count"), 0) or 0),
        "sled_fallback_failure_threshold": int(get_in(config, ("sled", "fallback_failure_threshold"), 0) or 0),
        "sled_enable_fallback_release": bool(get_in(config, ("sled", "enable_fallback_release"), False)),
        "sled_strict": sled_strict,
        "sled_static_queue_enabled": bool(
            get_in(config, ("sled", "static_queue", "enabled"), True if sled_strict else False)
        ),
        "sled_queue_max_wait_ms": get_in(config, ("sled", "static_queue", "max_wait_ms"), None),
        "sled_queue_pad_to_max_length": bool(get_in(config, ("sled", "static_queue", "pad_to_max_length"), True)),
        "estimated_server_verify_ms": float(
            get_in(config, ("planning", "estimated_server_verify_ms"), get_in(config, ("dip_sd", "estimated_server_verify_ms"), 120.0))
        ),
        "estimated_network_residual_ms": float(
            get_in(
                config,
                ("planning", "estimated_network_residual_ms"),
                get_in(config, ("transport", "rtt_ms"), 0.0) or 0.0,
            )
        ),
        "estimated_server_batch_per_request_ms": float(
            get_in(config, ("planning", "estimated_server_batch_per_request_ms"), get_in(config, ("dip_sd", "estimated_server_batch_per_request_ms"), 0.0))
        ),
        "estimated_server_batch_per_token_ms": float(
            get_in(config, ("planning", "estimated_server_batch_per_token_ms"), get_in(config, ("dip_sd", "estimated_server_batch_per_token_ms"), 0.0))
        ),
        "dip_sd_alpha": float(get_in(config, ("dip_sd", "alpha"), 0.78)),
        "dip_sd_acceptance": get_in(config, ("dip_sd", "acceptance"), None),
        "dip_sd_comm_latency_ms": float(get_in(config, ("dip_sd", "comm_latency_ms"), 3.0)),
        "dip_sd_draft_c": float(dip_sd_calibration_overrides.get("dip_sd_draft_c", base_dip_sd_draft_c)),
        "dip_sd_draft_beta": float(dip_sd_calibration_overrides.get("dip_sd_draft_beta", base_dip_sd_draft_beta)),
        "dip_sd_verify_c": float(dip_sd_calibration_overrides.get("dip_sd_verify_c", base_dip_sd_verify_c)),
        "dip_sd_verify_beta": float(dip_sd_calibration_overrides.get("dip_sd_verify_beta", base_dip_sd_verify_beta)),
        "dip_sd_memory_cap_bytes": float(get_in(config, ("dip_sd", "memory_cap_bytes"), 8.0e10)),
        "dip_sd_plan_cache_enabled": bool(get_in(config, ("dip_sd", "plan_cache_enabled"), True)),
        "dip_sd_steady_state_enabled": bool(get_in(config, ("dip_sd", "steady_state_enabled"), True)),
        "dip_sd_worker_assignment_strategy": str(get_in(config, ("dip_sd", "worker_assignment_strategy"), "latency_first")),
        "dip_sd_prefetch_sticky_worker_enabled": bool(
            get_in(config, ("dip_sd", "prefetch_sticky_worker_enabled"), True)
        ),
        "dip_sd_use_measured_worker_latency": bool(get_in(config, ("dip_sd", "use_measured_worker_latency"), True)),
        "dip_sd_adaptive_draft_length_enabled": bool(
            get_in(config, ("dip_sd", "adaptive_draft_length_enabled"), True)
        ),
        "dip_sd_min_draft_length": int(get_in(config, ("dip_sd", "min_draft_length"), 1) or 1),
        "dip_sd_adaptive_length_target_acceptance": float(
            get_in(config, ("dip_sd", "adaptive_length_target_acceptance"), 0.78) or 0.78
        ),
        "dip_sd_adaptive_length_min_factor": float(
            get_in(config, ("dip_sd", "adaptive_length_min_factor"), 0.35) or 0.35
        ),
        "dip_sd_ready_aware_rebatch_enabled": bool(get_in(config, ("dip_sd", "ready_aware_rebatch_enabled"), True)),
        "dip_sd_ready_aware_rebatch_min_spread_ms": float(
            get_in(config, ("dip_sd", "ready_aware_rebatch_min_spread_ms"), 5.0) or 0.0
        ),
        "dip_sd_slow_worker_length_threshold": float(
            get_in(config, ("dip_sd", "slow_worker_length_threshold"), 1.75) or 1.75
        ),
        "dip_sd_slow_worker_length_multiplier": float(
            get_in(config, ("dip_sd", "slow_worker_length_multiplier"), 1.25) or 1.25
        ),
        "dip_sd_prefetch_adaptive_length_enabled": bool(
            get_in(config, ("dip_sd", "prefetch_adaptive_length_enabled"), True)
        ),
        "dip_sd_prefetch_acceptance_lookahead_tokens": int(
            get_in(config, ("dip_sd", "prefetch_acceptance_lookahead_tokens"), 1) or 0
        ),
        "dip_sd_prefetch_min_tokens": int(get_in(config, ("dip_sd", "prefetch_min_tokens"), 1) or 1),
        "dip_sd_prefetch_max_tokens": int(get_in(config, ("dip_sd", "prefetch_max_tokens"), 0) or 0),
        "dip_sd_prefetch_use_source_budget_floor": bool(
            get_in(config, ("dip_sd", "prefetch_use_source_budget_floor"), False)
        ),
        "dip_sd_no_online_solver": bool(get_in(config, ("dip_sd", "no_online_solver"), False)),
        "dip_sd_no_online_solver_fallback": str(get_in(config, ("dip_sd", "no_online_solver_fallback"), "error")),
        "dip_sd_offline_plan_table": dip_sd_offline_plan_table,
        "dip_sd_offline_plan_prefix_bucket": int(get_in(config, ("dip_sd", "offline_plan_prefix_bucket"), 32) or 0),
        "dip_sd_acceptance_feedback_enabled": bool(
            get_in(config, ("dip_sd", "acceptance_feedback_enabled"), True)
        ),
        "dip_sd_acceptance_feedback_min_draft_tokens": int(
            get_in(config, ("dip_sd", "acceptance_feedback_min_draft_tokens"), 1) or 1
        ),
        "dip_sd_acceptance_feedback_prior_weight": float(
            get_in(config, ("dip_sd", "acceptance_feedback_prior_weight"), 1.0) or 0.0
        ),
        "dip_sd_acceptance_cache_bucket": float(
            get_in(config, ("dip_sd", "acceptance_cache_bucket"), 0.25) or 0.0
        ),
        "dip_sd_calibration_profile": None if dip_sd_calibration_profile is None else str(
            _resolve_profile_path(dip_sd_calibration_profile, dip_sd_calibration_config_path)
        ),
        "dip_sd_calibration_enabled": dip_sd_calibration_enabled,
        "dip_sd_calibration_applied": bool(dip_sd_calibration_overrides),
        "dip_sd_calibration_overrides": dict(dip_sd_calibration_overrides),
        "device": _value(args.device, config, ("draft", "device"), "cuda:0"),
        "torch_dtype": _value(args.torch_dtype, config, ("draft", "torch_dtype"), "fp16"),
        "device_map": _value(args.device_map, config, ("draft", "device_map"), None),
        "draft_backend": _value(args.draft_backend, config, ("draft", "backend"), "hf_eager"),
        "draft_max_graph_len": get_in(config, ("draft", "max_graph_len"), None),
        "draft_max_graph_tokens": get_in(config, ("draft", "max_graph_tokens"), None),
        "draft_max_graph_batch_size": get_in(config, ("draft", "max_graph_batch_size"), None),
        "draft_worker_count": worker_count,
        "draft_workers": raw_workers,
        "draft_registry_audit": dict(get_in(config, ("draft", "audit"), {}) or {}),
        "allow_backend_fallback": not bool(args.no_backend_fallback or get_in(config, ("draft", "disable_backend_fallback"), False)),
        "methods": _value(args.methods, config, ("run", "methods"), ["target_only", "linear", "tree"]),
        "output_dir": _value(args.output_dir, config, ("run", "output_dir"), f"experiments/{run_id}"),
        "plot_formats": tuple(
            part.strip()
            for part in str(_value(args.plot_formats, config, ("plots", "formats"), "png,svg")).split(",")
            if part.strip()
        ),
        "no_plots": bool(args.no_plots or get_in(config, ("plots", "disabled"), False)),
        "sync_dest": _value(args.sync_dest, config, ("run", "sync_dest"), None),
        "transport": {
            "mode": str(get_in(config, ("transport", "mode"), "observe")),
            "uplink_mbps": _value(args.transport_uplink_mbps, config, ("transport", "uplink_mbps"), None),
            "downlink_mbps": _value(args.transport_downlink_mbps, config, ("transport", "downlink_mbps"), None),
            "rtt_ms": _value(args.transport_rtt_ms, config, ("transport", "rtt_ms"), None),
        },
        "draft_warmup_enabled": bool(get_in(config, ("draft", "warmup", "enabled"), True)),
        "draft_warmup_tokens": int(get_in(config, ("draft", "warmup", "tokens"), 1)),
        "draft_warmup_tree_depth": int(
            get_in(config, ("draft", "warmup", "tree_depth"), get_in(config, ("tree", "max_depth"), 1))
        ),
        "draft_warmup_branch_width": int(
            get_in(config, ("draft", "warmup", "branch_width"), get_in(config, ("tree", "branch_width"), 2))
        ),
        "draft_warmup_max_budget": int(
            get_in(config, ("draft", "warmup", "max_budget"), get_in(config, ("tree", "max_budget"), 2))
        ),
        "draft_warmup_parallelism": int(get_in(config, ("draft", "warmup", "parallelism"), 0) or 0),
    }


def _load_draft_registry(settings: dict[str, Any]) -> DraftWorkerRegistry:
    """Load configured draft workers.

    Legacy configs with only ``draft.worker_count`` keep sharing one model to
    avoid surprising VRAM use.  Explicit ``draft.workers`` entries are loaded as
    independent configurable workers.
    """
    if settings.get("draft_workers"):
        return DraftWorkerRegistry.from_configs(draft_worker_configs_from_settings(settings))
    draft_model = load_causal_lm_runner(
        settings["draft_model_path"],
        runner_id="3090-qwen3-draft",
        backend=settings["draft_backend"],
        device=settings["device"],
        torch_dtype=settings["torch_dtype"],
        device_map=settings["device_map"],
        allow_fallback=settings["allow_backend_fallback"],
        max_graph_len=settings.get("draft_max_graph_len"),
        max_graph_tokens=settings.get("draft_max_graph_tokens"),
        max_graph_batch_size=settings.get("draft_max_graph_batch_size"),
    )
    return DraftWorkerRegistry.from_shared_model(
        draft_model,
        worker_count=settings["draft_worker_count"],
        model_path=settings["draft_model_path"],
        draft_type="both",
        device=settings["device"],
        backend=settings["draft_backend"],
        torch_dtype=settings["torch_dtype"],
        max_graph_len=settings.get("draft_max_graph_len"),
        max_graph_tokens=settings.get("draft_max_graph_tokens"),
        max_graph_batch_size=settings.get("draft_max_graph_batch_size"),
    )


def _audit_draft_registry(settings: dict[str, Any], draft_registry: DraftWorkerRegistry) -> dict[str, Any]:
    """Validate that a run is using the requested real draft worker pool."""
    raw_audit = dict(settings.get("draft_registry_audit") or {})
    enabled = bool(raw_audit.get("enabled", False))
    enforce = bool(raw_audit.get("enforce", enabled))
    registry_metadata = draft_registry.to_metadata()
    workers = [
        dict(worker)
        for worker in registry_metadata.get("draft_workers", [])
        if isinstance(worker, dict)
    ]
    devices = sorted({
        str(worker.get("device"))
        for worker in workers
        if worker.get("device") is not None
    })
    shared_workers = [
        str(worker.get("worker_id"))
        for worker in workers
        if dict(worker.get("metadata") or {}).get("shared_model")
    ]
    fallback_workers = []
    fallback_backend_names = set()
    for worker in workers:
        capabilities = dict(worker.get("backend_capabilities") or {})
        if capabilities.get("backend_fallback"):
            fallback_workers.append(str(worker.get("worker_id")))
            if capabilities.get("backend_name"):
                fallback_backend_names.add(str(capabilities["backend_name"]))
    violations: list[str] = []
    if enabled and raw_audit.get("require_explicit_workers") and not settings.get("draft_workers"):
        violations.append("draft.workers is required for explicit registry audit")
    if enabled and raw_audit.get("forbid_shared_model") and shared_workers:
        violations.append(f"shared draft workers present: {','.join(shared_workers)}")
    if enabled and raw_audit.get("forbid_backend_fallback") and fallback_workers:
        violations.append(f"backend fallback workers present: {','.join(fallback_workers)}")
    min_device_count = int(raw_audit.get("min_device_count") or 0)
    if enabled and min_device_count and len(devices) < min_device_count:
        violations.append(f"draft device count {len(devices)} < required {min_device_count}")
    required_devices = [
        str(device)
        for device in raw_audit.get("required_devices", [])
        if str(device).strip()
    ]
    missing_devices = [device for device in required_devices if device not in devices]
    if enabled and missing_devices:
        violations.append(f"missing required draft devices: {','.join(missing_devices)}")
    return {
        "enabled": enabled,
        "enforce": enforce,
        "ok": not violations,
        "violations": violations,
        "worker_count": len(workers),
        "device_count": len(devices),
        "device_set": devices,
        "shared_model_worker_count": len(shared_workers),
        "shared_model_workers": shared_workers,
        "backend_fallback_worker_count": len(fallback_workers),
        "backend_fallback_workers": fallback_workers,
        "backend_fallback_set": sorted(fallback_backend_names),
        "required_devices": required_devices,
        "min_device_count": min_device_count,
    }


def _value(cli_value: Any, config: dict[str, Any], path: tuple[str, ...], default: Any) -> Any:
    """CLI 非空值优先，否则读配置，最后使用默认。"""
    return cli_value if cli_value is not None else get_in(config, path, default)


def _target_urls(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    """Resolve target verifier URLs, letting CLI override YAML."""
    raw_urls = args.target_urls if getattr(args, "target_urls", None) is not None else get_in(config, ("target", "urls"), None)
    if raw_urls is None:
        raw_urls = _value(args.target_url, config, ("target", "url"), "http://172.16.11.62:8010")
    elif args.target_url is not None:
        raw_urls = args.target_url
    urls = [str(url).rstrip("/") for url in as_list(raw_urls, item_type=str) if str(url).strip()]
    if not urls:
        raise ValueError("At least one target URL is required.")
    return urls


def _resolve_profile_path(path: str | Path, config_path: str | None) -> Path:
    """Resolve calibration/profile paths relative to the YAML that names them."""
    raw = Path(path).expanduser()
    if raw.is_absolute():
        return raw
    if config_path:
        return (Path(config_path).expanduser().resolve().parent / raw).resolve()
    return raw.resolve()


def _dip_sd_offline_plan_table_config(config: dict[str, Any], *, config_path: str | None) -> dict[str, Any]:
    """Load an inline or file-backed DiP-SD offline plan table."""
    inline = get_in(config, ("dip_sd", "offline_plan_table"), None)
    table_file = get_in(config, ("dip_sd", "offline_plan_table_file"), None)
    if table_file is None:
        table_file = get_in(config, ("dip_sd", "offline_plan_table_path"), None)

    loaded: dict[str, Any] = {}
    if table_file:
        resolved = _resolve_profile_path(table_file, config_path)
        loaded = load_config(resolved)

    inline_table: dict[str, Any] = {}
    if inline:
        if not isinstance(inline, dict):
            raise ValueError("dip_sd.offline_plan_table must be a mapping.")
        inline_table = dict(inline)

    if loaded and inline_table:
        return _merge_dip_sd_offline_plan_tables(loaded, inline_table)
    if loaded:
        return dict(loaded)
    return inline_table


def _merge_dip_sd_offline_plan_tables(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    base_entries = base.get("entries") if isinstance(base.get("entries"), dict) else None
    override_entries = override.get("entries") if isinstance(override.get("entries"), dict) else None
    if base_entries is not None or override_entries is not None:
        entries: dict[str, Any] = {}
        if base_entries:
            entries.update(dict(base_entries))
        elif _looks_like_dip_sd_offline_entry(base):
            entries["default"] = {
                key: value
                for key, value in base.items()
                if key not in {"version", "entries"}
            }
        if override_entries:
            entries.update(dict(override_entries))
        elif _looks_like_dip_sd_offline_entry(override):
            entries["default"] = {
                key: value
                for key, value in override.items()
                if key not in {"version", "entries"}
            }
        merged.update({key: value for key, value in override.items() if key != "entries"})
        merged["entries"] = entries
        return merged
    merged.update(override)
    return merged


def _looks_like_dip_sd_offline_entry(value: dict[str, Any]) -> bool:
    return any(
        key in value
        for key in {"draft_lengths", "draft_length", "preferred_batches", "single_batch", "batch_size"}
    )


def _dip_sd_calibration_overrides(
    profile_path: str | Path | None,
    *,
    enabled: bool,
    config_path: str | None,
) -> dict[str, float]:
    """Load calibrated DiP-SD latency constants from a previous run artifact."""
    if not enabled or profile_path is None:
        return {}
    resolved = _resolve_profile_path(profile_path, config_path)
    return recommended_method_config_from_profile(resolved)


def _warm_draft_registry(
    *,
    settings: dict[str, Any],
    draft_registry: DraftWorkerRegistry,
    prompts: list[PromptSpec],
    methods: list[str],
    recorder: TimingRecorder,
) -> list[PhaseEvent]:
    """Warm draft workers before method timing starts, recording setup events."""
    draft_types = _warmup_draft_types(methods)
    if not settings.get("draft_warmup_enabled", True) or not draft_types:
        return []
    prefix_ids = _warmup_prefix_ids(prompts)
    if not prefix_ids:
        return []

    warmup_events: list[PhaseEvent] = []
    plan_id = f"{settings['run_id']}:setup"
    with recorder.span(
        phase="setup.warm_draft_workers",
        method="specedge_smoke",
        plan_id=plan_id,
        run_id=settings["run_id"],
        shared=True,
        metadata={
            "draft_types": list(draft_types),
            "prefix_len": len(prefix_ids),
            "warmup_tokens": int(settings["draft_warmup_tokens"]),
            "warmup_tree_depth": int(settings["draft_warmup_tree_depth"]),
            "warmup_branch_width": int(settings["draft_warmup_branch_width"]),
            "warmup_max_budget": int(settings["draft_warmup_max_budget"]),
        },
    ) as aggregate_span:
        tasks = [
            (draft_type, worker_id, runner)
            for draft_type in draft_types
            for worker_id, runner in draft_registry.runners_for(draft_type).items()
        ]
        warmup_results = _run_draft_warmup_tasks(
            settings=settings,
            tasks=tasks,
            prefix_ids=prefix_ids,
        )
        reset_count = sum(1 for result in warmup_results if result.reset_after_warmup)
        for result in warmup_results:
            warmup_span = recorder.record_completed(
                phase="setup.warm_draft_worker",
                method="specedge_smoke",
                plan_id=plan_id,
                run_id=settings["run_id"],
                worker_id=result.worker_id,
                start_ns=result.start_ns,
                end_ns=result.end_ns,
                metadata={
                    "warmup_type": result.draft_type,
                    "prefix_len": len(prefix_ids),
                    "runner_id": result.runner_id,
                    "produced_count": result.produced_count,
                    "reset_after_warmup": result.reset_after_warmup,
                    "warmup_parallelism": _warmup_parallelism(settings, len(tasks)),
                },
            )
            warmup_events.append(
                event_from_span(
                    warmup_span,
                    event_id_factory=recorder.next_event_id,
                    span_kind="setup",
                    attribution="system",
                    metadata=dict(warmup_span.metadata),
                )
            )
        aggregate_span.metadata.update(
            {
                "warmup_worker_event_count": len(warmup_results),
                "warmup_parallelism": _warmup_parallelism(settings, len(tasks)),
                "reset_count": reset_count,
            }
        )
    aggregate_event = event_from_span(
        aggregate_span,
        event_id_factory=recorder.next_event_id,
        span_kind="setup",
        attribution="system",
        metadata=dict(aggregate_span.metadata),
    )
    return [aggregate_event, *warmup_events]


def _warmup_draft_types(methods: list[str]) -> list[str]:
    method_set = {str(method) for method in methods}
    draft_types: list[str] = []
    if method_set & {"linear", "dip_sd", "sled", "sled_async"}:
        draft_types.append("greedy")
    if method_set & {"tree", "tree_stop_wait", "specedge_pipeline"}:
        draft_types.append("tree")
    return draft_types


def _warmup_prefix_ids(prompts: list[PromptSpec]) -> list[int]:
    for prompt in prompts:
        if prompt.prompt_ids:
            return list(prompt.prompt_ids)
    return []


def _run_draft_warmup(settings: dict[str, Any], runner: Any, draft_type: str, prefix_ids: list[int]) -> int:
    if draft_type == "greedy":
        generation = runner.generate_tokens(
            prefix_ids=list(prefix_ids),
            max_tokens=max(1, int(settings["draft_warmup_tokens"])),
            request_id="setup-warmup",
            metadata={"setup_warmup": True},
        )
        return len(getattr(generation, "tokens", []) or [])
    if draft_type == "tree":
        generation = runner.generate_tree(
            prefix_ids=list(prefix_ids),
            max_depth=max(1, int(settings["draft_warmup_tree_depth"])),
            max_branches=max(1, int(settings["draft_warmup_branch_width"])),
            max_nodes=max(1, int(settings["draft_warmup_max_budget"])),
            request_id="setup-warmup",
            metadata={"setup_warmup": True},
        )
        tree = getattr(generation, "tree", None)
        return len(getattr(tree, "nodes", []) or [])
    raise ValueError(f"Unsupported draft warmup type: {draft_type}")


def _run_draft_warmup_tasks(
    *,
    settings: dict[str, Any],
    tasks: list[tuple[str, str, Any]],
    prefix_ids: list[int],
) -> list[DraftWarmupResult]:
    if not tasks:
        return []

    def run_one(task: tuple[str, str, Any]) -> DraftWarmupResult:
        draft_type, worker_id, runner = task
        start_ns = perf_counter_ns()
        reset_after_warmup = False
        try:
            produced_count = _run_draft_warmup(settings, runner, draft_type, prefix_ids)
        finally:
            reset_after_warmup = _reset_warm_runner(runner)
        end_ns = perf_counter_ns()
        return DraftWarmupResult(
            draft_type=draft_type,
            worker_id=worker_id,
            runner_id=getattr(runner, "runner_id", worker_id),
            start_ns=start_ns,
            end_ns=end_ns,
            produced_count=produced_count,
            reset_after_warmup=reset_after_warmup,
        )

    def resource_key(task: tuple[str, str, Any]) -> int:
        runner = task[2]
        return id(getattr(runner, "model", runner))

    def run_group(group: list[tuple[str, str, Any]]) -> list[DraftWarmupResult]:
        return [run_one(group_task) for group_task in group]

    groups_by_resource: dict[int, list[tuple[str, str, Any]]] = {}
    for task in tasks:
        groups_by_resource.setdefault(resource_key(task), []).append(task)
    grouped_tasks = list(groups_by_resource.values())
    parallelism = _warmup_parallelism(settings, len(tasks))
    if parallelism <= 1 or len(grouped_tasks) <= 1:
        return [result for group in grouped_tasks for result in run_group(group)]
    with ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(run_group, group) for group in grouped_tasks]
        return [result for future in futures for result in future.result()]


def _apply_observed_warmup_profiles(
    draft_registry: DraftWorkerRegistry,
    warmup_events: list[PhaseEvent],
) -> dict[str, dict[str, Any]]:
    """Use measured greedy warmup speed as scheduler cost metadata."""
    observed_ms_per_token: dict[str, float] = {}
    for event in warmup_events:
        if event.phase != "setup.warm_draft_worker":
            continue
        if dict(event.metadata or {}).get("warmup_type") != "greedy":
            continue
        worker_id = str(event.worker_id or "")
        produced_count = int(dict(event.metadata or {}).get("produced_count") or 0)
        if not worker_id or produced_count <= 0:
            continue
        duration_ms = float(event.measured_duration_ms or event.duration_ms or 0.0)
        if duration_ms <= 0:
            continue
        observed_ms_per_token[worker_id] = duration_ms / produced_count
    if not observed_ms_per_token:
        return {}

    fastest_ms = max(1e-6, min(observed_ms_per_token.values()))
    applied: dict[str, dict[str, Any]] = {}
    for worker_id, ms_per_token in observed_ms_per_token.items():
        worker = draft_registry.workers.get(worker_id)
        if worker is None:
            continue
        old_profile = worker.config.speed_profile
        old_metadata = old_profile.to_dict()
        new_profile = DraftSpeedProfile(
            name=f"observed_warmup:{old_profile.name}",
            tokens_per_second=1000.0 / max(ms_per_token, 1e-6),
            latency_ms=0.0,
            relative_speed=fastest_ms / max(ms_per_token, 1e-6),
            quality=old_profile.quality,
            expected_acceptance=old_profile.expected_acceptance,
            metadata={
                **dict(old_profile.metadata),
                "configured_speed_profile": old_metadata,
                "observed_warmup_ms_per_token": ms_per_token,
            },
        )
        new_config = replace(
            worker.config,
            speed_profile=new_profile,
            metadata={
                **dict(worker.config.metadata),
                "observed_warmup_ms_per_token": ms_per_token,
            },
        )
        draft_registry.workers[worker_id] = DraftWorker(config=new_config, model=worker.model)
        applied[worker_id] = new_profile.to_dict()
    return applied


def _warmup_parallelism(settings: dict[str, Any], task_count: int) -> int:
    configured = int(settings.get("draft_warmup_parallelism") or 0)
    if configured > 0:
        return max(1, min(task_count, configured))
    return max(1, task_count)


def _reset_warm_runner(runner: Any) -> bool:
    model = getattr(runner, "model", None)
    reset = getattr(model, "reset", None)
    if reset is None:
        return False
    reset("setup-warmup")
    return True


def _run_target_only(
    settings: dict[str, Any],
    prompts: list[PromptSpec],
    eos_token_ids: list[int],
    setup_events: list[PhaseEvent],
) -> RuntimeRunResult:
    """通过 A100 /generate_greedy 跑 target-only baseline。"""
    recorder = TimingRecorder()
    logger = EventLogger(events=list(setup_events))
    client = HttpGreedyGeneratorClient(base_url=settings["target_url"])
    request_results: list[RuntimeRequestResult] = []
    for prompt in prompts:
        with recorder.span(
            phase="target.generate_total",
            method="target_only",
            plan_id=f"{settings['run_id']}:target_only:{prompt.request_id}",
            run_id=settings["run_id"],
            request_id=prompt.request_id,
            session_id=prompt.request_id,
            shared=False,
        ) as span:
            response, timing = client.generate(
                request_id=prompt.request_id,
                prefix_ids=prompt.prompt_ids,
                max_new_tokens=settings["max_new_tokens"],
                eos_token_ids=eos_token_ids,
                metadata={"batch_id": f"{settings['run_id']}:target_only"},
            )
        logger.record(
            event_from_span(
                span,
                event_id_factory=recorder.next_event_id,
                span_kind="leaf",
                attribution="request",
                tokens_out=len(response.generated_tokens),
                metadata={"response_timing": response.timing, **timing},
            )
        )
        for event_spec in timing.get("client_events", []):
            _record_detail_event(logger, recorder, event_spec, settings["run_id"], "target_only", prompt.request_id)
        request_results.append(
            RuntimeRequestResult(
                request_id=response.request_id,
                output_token_ids=list(response.generated_tokens),
                proposals=["target_only"],
                stop_reason=response.stop_reason,
            )
        )
    return RuntimeRunResult(
        request_results=request_results,
        events=logger,
    )


def _run_runtime_method(
    *,
    settings: dict[str, Any],
    method: str,
    prompts: list[PromptSpec],
    eos_token_ids: list[int],
    setup_events: list[PhaseEvent],
    draft_runners: dict[str, Any],
    candidate_strategy: Any,
    acceptance_policy: Any,
    verifier: Any,
    budget: DraftBudget,
    method_config: dict[str, Any],
    planning_policy: Any | None = None,
) -> RuntimeRunResult:
    """运行一个 RuntimeEngine 方法，并把 setup event 放到事件开头。"""
    recorder = TimingRecorder()
    first_runner = next(iter(draft_runners.values()))
    sessions = [
        GenerationSession(
            request_id=prompt.request_id,
            prompt_ids=prompt.prompt_ids,
            max_new_tokens=settings["max_new_tokens"],
            max_len=getattr(first_runner.model, "max_len", 32768),
            eos_token_ids=eos_token_ids,
        )
        for prompt in prompts
    ]
    engine = RuntimeEngine(
        candidate_strategy=candidate_strategy,
        acceptance_policy=acceptance_policy,
        scheduler=RoundRobinRequestScheduler(default_budget=budget),
        verifier=verifier,
        planning_policy=planning_policy,
        timing_recorder=recorder,
    )
    result = engine.run(
        run_id=f"{settings['run_id']}:{method}",
        sessions=sessions,
        draft_runners=draft_runners,
        context=RuntimeContext(
            run_config={
                "method": method,
                "eos_token_ids": eos_token_ids,
                "draft_worker_count": len(draft_runners),
            },
            method_config=method_config,
            backend_info={
                "target_placement": "a100",
                "target_host": settings["target_url"],
            },
        ),
    )
    result.events.events[0:0] = list(setup_events)
    return result


def _run_distributed_pipeline_method(
    *,
    settings: dict[str, Any],
    method: str,
    prompts: list[PromptSpec],
    eos_token_ids: list[int],
    setup_events: list[PhaseEvent],
    draft_runners: dict[str, Any],
    candidate_strategy: Any,
    acceptance_policy: Any,
    verifier: Any,
    budget: DraftBudget,
    method_config: dict[str, Any],
    planning_policy: Any | None = None,
) -> RuntimeRunResult:
    """运行通用 distributed draft / central batch verify pipeline runtime。"""
    recorder = TimingRecorder()
    first_runner = next(iter(draft_runners.values()))
    sessions = [
        GenerationSession(
            request_id=prompt.request_id,
            prompt_ids=prompt.prompt_ids,
            max_new_tokens=settings["max_new_tokens"],
            max_len=getattr(first_runner.model, "max_len", 32768),
            eos_token_ids=eos_token_ids,
        )
        for prompt in prompts
    ]
    engine = DistributedBatchPipelineRuntimeEngine(
        candidate_strategy=candidate_strategy,
        acceptance_policy=acceptance_policy,
        scheduler=RoundRobinRequestScheduler(default_budget=budget),
        verifier=verifier,
        planning_policy=planning_policy,
        timing_recorder=recorder,
    )
    result = engine.run(
        run_id=f"{settings['run_id']}:{method}",
        sessions=sessions,
        draft_runners=draft_runners,
        context=RuntimeContext(
            run_config={
                "method": method,
                "eos_token_ids": eos_token_ids,
                "draft_worker_count": len(draft_runners),
            },
            method_config=method_config,
            backend_info={
                "target_placement": "a100",
                "target_host": settings["target_url"],
            },
        ),
    )
    result.events.events[0:0] = list(setup_events)
    return result


def _run_pipeline_method(
    *,
    settings: dict[str, Any],
    method: str,
    prompts: list[PromptSpec],
    eos_token_ids: list[int],
    setup_events: list[PhaseEvent],
    draft_runners: dict[str, Any],
    verifier: Any,
    budget: DraftBudget,
    method_config: dict[str, Any],
) -> RuntimeRunResult:
    """运行异步 SpecEdge pipeline 方法，并把 setup event 放到事件开头。"""
    recorder = TimingRecorder()
    first_runner = next(iter(draft_runners.values()))
    sessions = [
        GenerationSession(
            request_id=prompt.request_id,
            prompt_ids=prompt.prompt_ids,
            max_new_tokens=settings["max_new_tokens"],
            max_len=getattr(first_runner.model, "max_len", 32768),
            eos_token_ids=eos_token_ids,
        )
        for prompt in prompts
    ]
    official_state = (
        OfficialSpecEdgeDraftState(
            max_batch_size=max(1, len(sessions)),
            draft_worker_id="specedge-official-draft",
        )
        if settings.get("specedge_official", True)
        else None
    )
    planning_policy = SpecEdgePipelinePlanningPolicy(
        min_depth=int(method_config.get("min_depth", 1)),
        max_depth=int(method_config.get("max_depth", 8)),
        initial_depth=int(budget.max_tokens),
        official_state=official_state,
    )
    candidate_strategy = (
        SpecEdgeOfficialCandidateStrategy(
            state=official_state,
            default_max_budget=settings["tree_max_budget"],
            default_max_branch_width=settings["tree_branch_width"],
        )
        if official_state is not None
        else SpecEdgeTreeCandidateStrategy(
            default_max_budget=settings["tree_max_budget"],
            default_max_branch_width=settings["tree_branch_width"],
        )
    )
    acceptance_policy = (
        SpecEdgeOfficialAcceptancePolicy(state=official_state)
        if official_state is not None
        else SpecEdgeTreeAcceptancePolicy()
    )
    proactive_policy = (
        SpecEdgeOfficialProactiveDraftPolicy(
            state=official_state,
            default_max_depth=settings["pipeline_proactive_depth"],
            default_max_branch_width=settings["tree_branch_width"],
            default_max_budget=settings["tree_max_budget"],
        )
        if official_state is not None
        else SpecEdgeProactiveDraftPolicy(
            default_max_depth=settings["pipeline_proactive_depth"],
            default_max_branch_width=settings["tree_branch_width"],
            default_max_budget=settings["tree_max_budget"],
        )
    )
    method_config = {
        **dict(method_config),
        "official_specedge_state": official_state is not None,
    }
    engine = AsyncPipelineRuntimeEngine(
        candidate_strategy=candidate_strategy,
        acceptance_policy=acceptance_policy,
        scheduler=RoundRobinRequestScheduler(default_budget=budget),
        verifier=verifier,
        proactive_policy=proactive_policy,
        reconcile_policy=SpecEdgeOfficialReconcilePolicy(state=official_state)
        if official_state is not None
        else SpecEdgeReconcilePolicy(),
        planning_policy=planning_policy,
        timing_recorder=recorder,
    )
    result = engine.run(
        run_id=f"{settings['run_id']}:{method}",
        sessions=sessions,
        draft_runners=draft_runners,
        context=RuntimeContext(
            run_config={
                "method": method,
                "eos_token_ids": eos_token_ids,
                "draft_worker_count": len(draft_runners),
            },
            method_config=method_config,
            backend_info={
                "target_placement": "a100",
                "target_host": settings["target_url"],
            },
        ),
    )
    result.events.events[0:0] = list(setup_events)
    return result


def _run_sled_async_method(
    *,
    settings: dict[str, Any],
    method: str,
    prompts: list[PromptSpec],
    eos_token_ids: list[int],
    setup_events: list[PhaseEvent],
    draft_runners: dict[str, Any],
    verifier: Any,
) -> RuntimeRunResult:
    """Run paper-style SLED async drafting with one edge stream per request."""
    if settings.get("sled_strict") and settings.get("sled_enable_fallback_release"):
        raise ValueError("Strict sled_async cannot enable local fallback release.")
    recorder = TimingRecorder()
    first_runner = next(iter(draft_runners.values()))
    sessions = [
        GenerationSession(
            request_id=prompt.request_id,
            prompt_ids=prompt.prompt_ids,
            max_new_tokens=settings["max_new_tokens"],
            max_len=getattr(first_runner.model, "max_len", 32768),
            eos_token_ids=eos_token_ids,
        )
        for prompt in prompts
    ]
    method_config = {
        "max_budget": settings["tree_max_budget"],
        "max_branch_width": 1,
        "sled_batch_size": settings["sled_batch_size"],
        "sled_confidence_threshold": settings["sled_confidence_threshold"],
        "sled_async_proactive_tokens": settings["sled_async_proactive_tokens"],
        "sled_allow_bonus": False,
        "sled_verify_timeout_ms": settings["sled_verify_timeout_ms"],
        "sled_retry_count": settings["sled_retry_count"],
        "sled_fallback_failure_threshold": settings["sled_fallback_failure_threshold"],
        "sled_enable_fallback_release": settings["sled_enable_fallback_release"],
        "sled_static_queue_enabled": settings["sled_static_queue_enabled"],
        "sled_queue_max_wait_ms": settings["sled_queue_max_wait_ms"],
        "sled_queue_pad_to_max_length": settings["sled_queue_pad_to_max_length"],
        "sled_strict": settings["sled_strict"],
    }
    budget = DraftBudget(max_tokens=settings["sled_max_speculation_tokens"], max_branches=1)
    planning_policy = SLEDPlanningPolicy(
        min_depth=settings["pipeline_min_depth"],
        max_depth=settings["pipeline_max_depth"],
        max_speculation_tokens=settings["sled_max_speculation_tokens"],
        target_batch_size=settings["sled_batch_size"],
        confidence_threshold=settings["sled_confidence_threshold"],
    )
    engine = AsyncPipelineRuntimeEngine(
        candidate_strategy=SLEDDynamicCandidateStrategy(
            proposal_prefix="sled-linear",
            confidence_threshold=settings["sled_confidence_threshold"],
        ),
        acceptance_policy=GreedyPrefixAcceptancePolicy(),
        scheduler=RoundRobinRequestScheduler(default_budget=budget),
        verifier=verifier,
        proactive_policy=SLEDAsyncDraftPolicy(
            default_max_tokens=settings["sled_async_proactive_tokens"],
            confidence_threshold=settings["sled_confidence_threshold"],
        ),
        reconcile_policy=SLEDAsyncReconcilePolicy(),
        planning_policy=planning_policy,
        timing_recorder=recorder,
        max_verify_workers=max(1, min(4, len(settings.get("target_urls") or []) + int(settings["sled_retry_count"] or 0))),
    )
    result = engine.run(
        run_id=f"{settings['run_id']}:{method}",
        sessions=sessions,
        draft_runners=draft_runners,
        context=RuntimeContext(
            run_config={
                "method": method,
                "eos_token_ids": eos_token_ids,
                "draft_worker_count": len(draft_runners),
            },
            method_config=method_config,
            backend_info={
                "target_placement": "a100",
                "target_host": settings["target_url"],
                "target_urls": list(settings.get("target_urls") or [settings["target_url"]]),
            },
        ),
    )
    result.events.events[0:0] = list(setup_events)
    return result


def _transport_profile(settings: dict[str, Any]) -> TransportProfile:
    raw = dict(settings.get("transport") or {})
    return TransportProfile(
        mode=str(raw.get("mode") or "observe"),
        uplink_mbps=None if raw.get("uplink_mbps") is None else float(raw["uplink_mbps"]),
        downlink_mbps=None if raw.get("downlink_mbps") is None else float(raw["downlink_mbps"]),
        rtt_ms=None if raw.get("rtt_ms") is None else float(raw["rtt_ms"]),
    )


def _linear_verifier(settings: dict[str, Any], transport_profile: TransportProfile) -> Any:
    urls = [str(url) for url in settings.get("target_urls") or [settings["target_url"]]]
    if len(urls) <= 1:
        return HttpLinearVerifierClient(base_url=urls[0], transport_profile=transport_profile)
    return HttpLinearVerifierPoolClient(base_urls=urls, transport_profile=transport_profile)


def _assert_target_graph_backend(settings: dict[str, Any]) -> None:
    """Fail fast for strict SLED runs unless A100 is serving true qwen3_graph."""
    target_urls = [str(url).rstrip("/") for url in settings.get("target_urls") or [settings["target_url"]]]
    required_backend = str(settings.get("target_required_backend") or "qwen3_graph")
    required_replicas = int(settings.get("target_required_replicas") or 0)
    if required_replicas > 0 and len(target_urls) < required_replicas:
        raise RuntimeError(
            f"Strict SLED requires at least {required_replicas} target replicas, "
            f"but only {len(target_urls)} URL(s) are configured: {target_urls}."
        )
    for target_url in target_urls:
        health_url = target_url.rstrip("/") + "/health"
        try:
            with urllib.request.urlopen(health_url, timeout=5) as response:
                health = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Strict SLED target health check failed at {health_url}: {exc}") from None
        capabilities = dict(health.get("model_backend_capabilities") or {})
        backend = str(health.get("model_backend") or capabilities.get("backend_name") or "")
        if backend != required_backend:
            raise RuntimeError(
                f"Strict SLED requires target backend {required_backend!r}, "
                f"but {health_url} reports {backend!r}."
            )
        if capabilities.get("backend_fallback"):
            raise RuntimeError(f"Strict SLED target backend is fallback at {health_url}: {capabilities}")
        if not capabilities.get("supports_cuda_graph"):
            raise RuntimeError(f"Strict SLED target backend does not support CUDA graph at {health_url}: {capabilities}")
        if not capabilities.get("supports_kv_cache"):
            raise RuntimeError(f"Strict SLED target backend does not support explicit/KV cache at {health_url}: {capabilities}")


def _greedy_draft_runners(model: TransformersCausalLMRunner, worker_count: int) -> dict[str, GreedyDraftRunner]:
    """构造逻辑 draft worker map；后续可替换成多设备/多模型 worker。"""
    return {
        f"draft-worker-{index}": GreedyDraftRunner(model=model, runner_id=f"draft-worker-{index}")
        for index in range(max(1, int(worker_count)))
    }


def _topk_tree_draft_runners(model: TransformersCausalLMRunner, worker_count: int) -> dict[str, TopKTreeDraftRunner]:
    """构造 tree draft worker map，供 SpecEdge/DiP-SD/SLED 复用。"""
    return {
        f"draft-worker-{index}": TopKTreeDraftRunner(model=model, runner_id=f"draft-worker-{index}")
        for index in range(max(1, int(worker_count)))
    }


def _record_detail_event(
    logger: EventLogger,
    recorder: TimingRecorder,
    event_spec: dict[str, Any],
    run_id: str,
    method: str,
    request_id: str,
) -> None:
    """把 target-only HTTP client 细节事件写成 PhaseEvent。"""
    start_ns = event_spec.get("start_ns")
    end_ns = event_spec.get("end_ns")
    if start_ns is None or end_ns is None:
        return
    span = recorder.record_completed(
        phase=str(event_spec.get("phase")),
        method=method,
        plan_id=f"{run_id}:{method}",
        run_id=run_id,
        request_id=request_id,
        session_id=request_id,
        start_ns=int(start_ns),
        end_ns=int(end_ns),
        metadata={key: value for key, value in event_spec.items() if key not in {"start_ns", "end_ns"}},
    )
    logger.record(
        event_from_span(
            span,
            event_id_factory=recorder.next_event_id,
            span_kind="detail",
            attribution="request",
            metadata=dict(span.metadata),
        )
    )


def _write_method_artifacts(
    result: RuntimeRunResult,
    output_dir: Path,
    *,
    render_plots: bool,
    plot_formats: tuple[str, ...],
    metadata: dict[str, Any],
) -> None:
    """写单个 method 的 artifacts。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.events.write_jsonl(output_dir / "phase_events.jsonl")
    write_phase_events_csv(list(result.events.events), output_dir / "phase_events.csv")
    write_phase_summary_csv(list(result.events.events), output_dir / "phase_summary.csv")
    write_request_results_json(result.request_results, output_dir / "request_results.json")
    write_tree_snapshots_jsonl(list(result.events.events), output_dir / "tree_snapshots.jsonl")
    (output_dir / "smoke_output.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if metadata.get("method") == "dip_sd":
        _write_dip_sd_method_artifacts(result, output_dir)
    if render_plots:
        plot_start_ns = perf_counter_ns()
        write_timing_charts(list(result.events.events), output_dir / "plots", formats=plot_formats)
        plot_end_ns = perf_counter_ns()
        (output_dir / "plot_render_ms.txt").write_text(
            f"{(plot_end_ns - plot_start_ns) / 1_000_000:.6f}\n",
            encoding="utf-8",
        )
    else:
        write_timing_audit(list(result.events.events), output_dir / "plots")


def _write_dip_sd_method_artifacts(result: RuntimeRunResult, output_dir: Path) -> None:
    hints = _scheduler_hints_from_result(result)
    solution = dict(hints.get("dip_sd_solution") or {})
    (output_dir / "dip_sd_solver_trace.json").write_text(
        json.dumps(solution, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    stage_rows = [
        dict(row)
        for row in hints.get("preferred_batch_metadata", [])
        if isinstance(row, dict)
    ]
    _write_dict_rows(
        output_dir / "dip_sd_stage_plan.csv",
        stage_rows,
        fieldnames=[
            "stage_index",
            "planned_batch_count",
            "request_ids",
            "max_draft_len",
            "max_prefix_len",
            "estimated_draft_complete_ms",
            "estimated_verify_ms",
            "estimated_stage_duration_ms",
            "estimated_memory_bytes",
        ],
    )
    timeline_rows = []
    for event in result.events.events:
        if event.phase not in {
            "pipeline.stage",
            "pipeline.server_idle",
            "pipeline.draft_ready_wait",
            "pipeline.planner_wait",
            "pipeline.sync",
            "pipeline.steady_state_prefetch_submit",
            "pipeline.steady_state_prefetch_reuse",
            "draft.prefetch_discard",
            "verify.batch_total",
        }:
            continue
        metadata = dict(event.metadata or {})
        phase = "pipeline.verify_busy" if event.phase == "verify.batch_total" else event.phase
        timeline_rows.append(
            {
                "phase": phase,
                "round_id": event.round,
                "batch_id": event.batch_id,
                "stage_index": metadata.get("stage_index"),
                "request_ids": metadata.get("request_ids"),
                "proposal_ids": metadata.get("proposal_ids"),
                "start_ns": event.start_ns,
                "end_ns": event.end_ns,
                "duration_ms": event.measured_duration_ms,
                "runtime_engine": metadata.get("runtime_engine"),
            }
        )
    _write_dict_rows(
        output_dir / "pipeline_stage_timeline.csv",
        timeline_rows,
        fieldnames=[
            "phase",
            "round_id",
            "batch_id",
            "stage_index",
            "request_ids",
            "proposal_ids",
            "start_ns",
            "end_ns",
            "duration_ms",
            "runtime_engine",
        ],
    )
    solver_rows = []
    for event in result.events.events:
        if event.phase != "dip_sd.solver":
            continue
        metadata = dict(event.metadata or {})
        hints_metadata = dict(metadata.get("hints_metadata") or {})
        solver_rows.append(
            {
                "round_id": event.round,
                "duration_ms": event.measured_duration_ms,
                "requested_solver_mode": hints_metadata.get("requested_solver_mode"),
                "solver_mode": hints_metadata.get("solver_mode"),
                "solver_backend_name": hints_metadata.get("solver_backend_name"),
                "paper_solver_complete": hints_metadata.get("paper_solver_complete"),
                "solver_backend_fallback_used": hints_metadata.get("solver_backend_fallback_used"),
                "solver_backend_fallback_reason": hints_metadata.get("solver_backend_fallback_reason"),
                "solver_cache_hit": hints_metadata.get("solver_cache_hit"),
                "solver_cache_key": hints_metadata.get("solver_cache_key"),
                "online_solver_enabled": hints_metadata.get("online_solver_enabled"),
                "offline_plan_table_hit": hints_metadata.get("offline_plan_table_hit"),
                "offline_plan_key": hints_metadata.get("offline_plan_key"),
                "offline_plan_shape_key": hints_metadata.get("offline_plan_shape_key"),
                "offline_plan_source": hints_metadata.get("offline_plan_source"),
                "planned_batch_count": hints_metadata.get("planned_batch_count"),
                "solver_planned_batch_count": hints_metadata.get("solver_planned_batch_count"),
                "hybrid_single_batch_threshold": hints_metadata.get("hybrid_single_batch_threshold"),
                "hybrid_single_batch_applied": hints_metadata.get("hybrid_single_batch_applied"),
                "hybrid_single_batch_reason": hints_metadata.get("hybrid_single_batch_reason"),
                "estimated_pipeline_span_ms": hints_metadata.get("estimated_pipeline_span_ms"),
                "estimated_throughput_tokens_per_s": hints_metadata.get("estimated_throughput_tokens_per_s"),
                "latency_calibration_profile": hints_metadata.get("latency_calibration_profile"),
                "latency_calibration_enabled": hints_metadata.get("latency_calibration_enabled"),
                "latency_calibration_applied": hints_metadata.get("latency_calibration_applied"),
                "latency_calibration_overrides": hints_metadata.get("latency_calibration_overrides"),
                "acceptance_feedback_enabled": hints_metadata.get("acceptance_feedback_enabled"),
                "acceptance_feedback_applied_count": hints_metadata.get("acceptance_feedback_applied_count"),
                "acceptance_feedback_by_request": hints_metadata.get("acceptance_feedback_by_request"),
                "acceptance_cache_bucket": hints_metadata.get("acceptance_cache_bucket"),
                "solver_cache_shape_level": hints_metadata.get("solver_cache_shape_level"),
            }
        )
    _write_dict_rows(
        output_dir / "solver_time.csv",
        solver_rows,
        fieldnames=[
            "round_id",
            "duration_ms",
            "requested_solver_mode",
            "solver_mode",
            "solver_backend_name",
            "paper_solver_complete",
            "solver_backend_fallback_used",
            "solver_backend_fallback_reason",
            "solver_cache_hit",
            "solver_cache_key",
            "online_solver_enabled",
            "offline_plan_table_hit",
            "offline_plan_key",
            "offline_plan_shape_key",
            "offline_plan_source",
            "planned_batch_count",
            "solver_planned_batch_count",
            "hybrid_single_batch_threshold",
            "hybrid_single_batch_applied",
            "hybrid_single_batch_reason",
            "estimated_pipeline_span_ms",
            "estimated_throughput_tokens_per_s",
            "latency_calibration_profile",
            "latency_calibration_enabled",
            "latency_calibration_applied",
            "latency_calibration_overrides",
            "acceptance_feedback_enabled",
            "acceptance_feedback_applied_count",
            "acceptance_feedback_by_request",
            "acceptance_cache_bucket",
            "solver_cache_shape_level",
        ],
    )
    (output_dir / "dip_sd_offline_plan_table.json").write_text(
        json.dumps(
            _dip_sd_offline_plan_table_from_result(result),
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_dict_rows(
        output_dir / "estimated_vs_actual_pipeline_span.csv",
        _dip_sd_estimated_vs_actual_rows(result),
        fieldnames=[
            "round_id",
            "actual_round_total_ms",
            "actual_planner_wait_ms",
            "actual_draft_ready_wait_ms",
            "actual_verify_busy_ms",
            "actual_stage_sum_ms",
            "estimated_pipeline_span_ms",
            "estimated_verification_sum_ms",
            "estimated_max_draft_verify_ms",
            "planned_batch_count",
            "solver_planned_batch_count",
            "hybrid_single_batch_applied",
            "requested_solver_mode",
            "solver_mode",
            "solver_backend_name",
            "paper_solver_complete",
            "solver_backend_fallback_used",
            "solver_backend_fallback_reason",
            "solver_cache_hit",
            "latency_calibration_profile",
            "latency_calibration_applied",
        ],
    )
    calibration = calibration_from_events(
        list(result.events.events),
        model_config=_dip_sd_model_config_from_hints(hints),
    )
    _write_dict_rows(
        output_dir / "dip_sd_latency_observations.csv",
        list(calibration.get("observations") or []),
        fieldnames=[
            "kind",
            "source",
            "source_index",
            "round_id",
            "worker_id",
            "request_id",
            "batch_id",
            "prefix_len",
            "draft_len",
            "batch_size",
            "token_count",
            "x_value",
            "duration_ms",
            "model_path",
            "backend",
            "device",
        ],
    )
    _write_dict_rows(
        output_dir / "dip_sd_latency_calibration.csv",
        list(calibration.get("fits") or []),
        fieldnames=[
            "kind",
            "group",
            "observation_count",
            "c",
            "beta",
            "mae_ms",
            "rmse_ms",
            "r2",
            "fit_status",
            "x_min",
            "x_max",
            "duration_min_ms",
            "duration_max_ms",
        ],
    )
    calibration_summary = {
        key: value
        for key, value in calibration.items()
        if key != "observations"
    }
    (output_dir / "dip_sd_latency_calibration.json").write_text(
        json.dumps(calibration_summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _dip_sd_offline_plan_table_from_result(result: RuntimeRunResult) -> dict[str, Any]:
    """Build a no-online-solver replay table from DiP-SD scheduler plan events."""
    entries: dict[str, dict[str, Any]] = {}
    source_run_id = None
    plan_event_count = 0
    for event in result.events.events:
        if event.phase != "scheduler.plan":
            continue
        metadata = dict(event.metadata or {})
        hints_metadata = dict(metadata.get("hints_metadata") or metadata.get("planning_hints") or {})
        if hints_metadata.get("method_family") != "dip_sd":
            continue
        shape_key = str(hints_metadata.get("offline_plan_shape_key") or "")
        if not shape_key:
            continue
        entry = _dip_sd_offline_plan_entry_from_event(metadata, hints_metadata)
        if entry is None:
            continue
        plan_event_count += 1
        if source_run_id is None:
            source_run_id = event.run_id
        entries.setdefault(shape_key, entry)
        entries[shape_key].setdefault("observed_round_ids", []).append(event.round)
    return {
        "version": 1,
        "source": "dip_sd_scheduler_plan_events",
        "source_run_id": source_run_id,
        "plan_event_count": plan_event_count,
        "shape_key_count": len(entries),
        "entries": entries,
    }


def _dip_sd_offline_plan_entry_from_event(
    metadata: dict[str, Any],
    hints_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    draft_lengths = metadata.get("draft_lengths")
    if not isinstance(draft_lengths, dict) or not draft_lengths:
        return None
    request_order = [str(request_id) for request_id in draft_lengths]
    preferred_batches = metadata.get("preferred_batches")
    if not preferred_batches:
        preferred_batches = _preferred_batches_from_verify_metadata(metadata.get("verify_batches"))
    positional_batches = _dip_sd_positional_batches(preferred_batches, request_order)
    if not positional_batches:
        return None
    shape_key = str(hints_metadata.get("offline_plan_shape_key"))
    source = str(hints_metadata.get("offline_plan_source") or "online_solver_trace")
    entry: dict[str, Any] = {
        "key": shape_key,
        "source": source,
        "request_order": request_order,
        "draft_lengths": [
            int(draft_lengths[request_id])
            for request_id in request_order
        ],
        "preferred_batches": positional_batches,
        "solver_planned_batch_count": int(
            hints_metadata.get("solver_planned_batch_count") or len(positional_batches)
        ),
    }
    optional_fields = (
        "requested_solver_mode",
        "solver_mode",
        "solver_backend_name",
        "paper_solver_complete",
        "hybrid_single_batch_applied",
        "hybrid_single_batch_reason",
        "hybrid_single_batch_threshold",
        "estimated_pipeline_span_ms",
        "estimated_throughput_tokens_per_s",
        "acceptance_cache_bucket",
        "latency_calibration_profile",
        "latency_calibration_enabled",
        "latency_calibration_applied",
    )
    for key in optional_fields:
        if key in hints_metadata:
            entry[key] = hints_metadata.get(key)
    return entry


def _preferred_batches_from_verify_metadata(raw: Any) -> list[list[str]]:
    if not isinstance(raw, list):
        return []
    batches: list[list[str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        request_ids = item.get("request_ids")
        if isinstance(request_ids, list):
            batches.append([str(request_id) for request_id in request_ids])
    return batches


def _dip_sd_positional_batches(raw: Any, request_order: list[str]) -> list[list[int]]:
    if not isinstance(raw, list) or not request_order:
        return []
    index_by_request = {request_id: index for index, request_id in enumerate(request_order)}
    batches: list[list[int]] = []
    seen: set[int] = set()
    for batch in raw:
        if not isinstance(batch, list):
            return []
        converted: list[int] = []
        for item in batch:
            index = _dip_sd_request_position(item, index_by_request, len(request_order))
            if index is None or index in seen:
                return []
            seen.add(index)
            converted.append(index)
        if converted:
            batches.append(converted)
    if seen != set(range(len(request_order))):
        return []
    return batches


def _dip_sd_request_position(
    value: Any,
    index_by_request: dict[str, int],
    request_count: int,
) -> int | None:
    if isinstance(value, int):
        return value if 0 <= value < request_count else None
    text = str(value)
    if text in index_by_request:
        return index_by_request[text]
    if text.isdigit():
        index = int(text)
        if 0 <= index < request_count:
            return index
    return None


def _dip_sd_estimated_vs_actual_rows(result: RuntimeRunResult) -> list[dict[str, Any]]:
    by_round: dict[int, dict[str, Any]] = {}
    for event in result.events.events:
        if event.round is None:
            continue
        row = by_round.setdefault(
            int(event.round),
            {
                "round_id": int(event.round),
                "actual_round_total_ms": 0.0,
                "actual_planner_wait_ms": 0.0,
                "actual_draft_ready_wait_ms": 0.0,
                "actual_verify_busy_ms": 0.0,
                "actual_stage_sum_ms": 0.0,
                "estimated_pipeline_span_ms": None,
                "estimated_verification_sum_ms": None,
                "estimated_max_draft_verify_ms": None,
                "planned_batch_count": None,
                "solver_planned_batch_count": None,
                "hybrid_single_batch_applied": None,
                "requested_solver_mode": None,
                "solver_mode": None,
                "solver_backend_name": None,
                "paper_solver_complete": None,
                "solver_backend_fallback_used": None,
                "solver_backend_fallback_reason": None,
                "solver_cache_hit": None,
                "latency_calibration_profile": None,
                "latency_calibration_applied": None,
            },
        )
        duration = float(event.measured_duration_ms or event.duration_ms or 0.0)
        metadata = dict(event.metadata or {})
        if event.phase == "runtime.round_total" and event.span_kind == "aggregate":
            row["actual_round_total_ms"] += duration
        elif event.phase == "pipeline.planner_wait":
            row["actual_planner_wait_ms"] += duration
        elif event.phase in {"pipeline.server_idle", "pipeline.draft_ready_wait"}:
            row["actual_draft_ready_wait_ms"] += duration
        elif event.phase == "verify.batch_total":
            row["actual_verify_busy_ms"] += duration
        elif event.phase == "pipeline.stage":
            row["actual_stage_sum_ms"] += duration
        elif event.phase == "dip_sd.solver":
            hints_metadata = dict(metadata.get("hints_metadata") or {})
            row["estimated_pipeline_span_ms"] = hints_metadata.get("estimated_pipeline_span_ms")
            row["estimated_verification_sum_ms"] = hints_metadata.get("estimated_verification_sum_ms")
            row["estimated_max_draft_verify_ms"] = hints_metadata.get("estimated_max_draft_verify_ms")
            row["planned_batch_count"] = hints_metadata.get("planned_batch_count")
            row["solver_planned_batch_count"] = hints_metadata.get("solver_planned_batch_count")
            row["hybrid_single_batch_applied"] = hints_metadata.get("hybrid_single_batch_applied")
            row["requested_solver_mode"] = hints_metadata.get("requested_solver_mode")
            row["solver_mode"] = hints_metadata.get("solver_mode")
            row["solver_backend_name"] = hints_metadata.get("solver_backend_name")
            row["paper_solver_complete"] = hints_metadata.get("paper_solver_complete")
            row["solver_backend_fallback_used"] = hints_metadata.get("solver_backend_fallback_used")
            row["solver_backend_fallback_reason"] = hints_metadata.get("solver_backend_fallback_reason")
            row["solver_cache_hit"] = hints_metadata.get("solver_cache_hit")
            row["latency_calibration_profile"] = hints_metadata.get("latency_calibration_profile")
            row["latency_calibration_applied"] = hints_metadata.get("latency_calibration_applied")
    return [
        {
            **row,
            "actual_round_total_ms": round(float(row["actual_round_total_ms"]), 6),
            "actual_planner_wait_ms": round(float(row["actual_planner_wait_ms"]), 6),
            "actual_draft_ready_wait_ms": round(float(row["actual_draft_ready_wait_ms"]), 6),
            "actual_verify_busy_ms": round(float(row["actual_verify_busy_ms"]), 6),
            "actual_stage_sum_ms": round(float(row["actual_stage_sum_ms"]), 6),
        }
        for _round, row in sorted(by_round.items())
    ]


def _dip_sd_model_config_from_hints(hints: dict[str, Any]) -> DiPSDModelConfig:
    raw = hints.get("dip_sd_model_config")
    if not isinstance(raw, dict):
        return DiPSDModelConfig()
    defaults = DiPSDModelConfig()
    values: dict[str, Any] = {}
    for field_name in (
        "draft_blocks",
        "draft_hidden",
        "draft_ffn_hidden",
        "verify_blocks",
        "verify_hidden",
        "verify_ffn_hidden",
        "verify_c",
        "verify_beta",
        "memory_cap_bytes",
    ):
        values[field_name] = raw.get(field_name, getattr(defaults, field_name))
    return DiPSDModelConfig(
        draft_blocks=int(values["draft_blocks"]),
        draft_hidden=int(values["draft_hidden"]),
        draft_ffn_hidden=int(values["draft_ffn_hidden"]),
        verify_blocks=int(values["verify_blocks"]),
        verify_hidden=int(values["verify_hidden"]),
        verify_ffn_hidden=int(values["verify_ffn_hidden"]),
        verify_c=float(values["verify_c"]),
        verify_beta=float(values["verify_beta"]),
        memory_cap_bytes=float(values["memory_cap_bytes"]),
    )


def _scheduler_hints_from_result(result: RuntimeRunResult) -> dict[str, Any]:
    for event in result.events.events:
        if event.phase != "scheduler.plan":
            continue
        metadata = dict(event.metadata or {})
        hints = metadata.get("hints_metadata") or metadata.get("planning_hints") or {}
        if isinstance(hints, dict):
            return dict(hints)
    return {}


def _write_dict_rows(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {
                key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict, tuple)) else value
                for key, value in row.items()
            }
            writer.writerow(normalized)


def _metadata(settings: dict[str, Any], method: str, prompts: list[PromptSpec], result: RuntimeRunResult) -> dict[str, Any]:
    """构造 smoke_output.json。"""
    outputs = _outputs_by_request(result)
    setup_event = next((event for event in result.events.events if event.phase == "setup.load_draft_model"), None)
    draft_registry = dict((setup_event.metadata or {}).get("draft_registry") or {}) if setup_event is not None else {}
    draft_registry_audit = (
        dict((setup_event.metadata or {}).get("draft_registry_audit") or {})
        if setup_event is not None
        else {}
    )
    setup_summary = _setup_metrics(result)
    return {
        "run_id": settings["run_id"],
        "method": method,
        "request_count": len(prompts),
        "prompts": [
            {
                "request_id": prompt.request_id,
                "prompt": prompt.prompt,
                "prompt_len": len(prompt.prompt_ids),
            }
            for prompt in prompts
        ],
        "draft_model_path": settings["draft_model_path"],
        "draft_worker_count": settings["draft_worker_count"],
        "draft_registry": draft_registry,
        "draft_registry_audit": draft_registry_audit,
        "target_url": settings["target_url"],
        "target_urls": list(settings.get("target_urls") or [settings["target_url"]]),
        "max_new_tokens": settings["max_new_tokens"],
        "generated_ids_by_request": outputs,
        "generated_lens_by_request": {request_id: len(tokens) for request_id, tokens in outputs.items()},
        "setup": setup_summary,
    }


def _build_combined_summary(
    method_outputs: dict[str, dict[str, list[int]]],
    *,
    method_results: dict[str, RuntimeRunResult] | None = None,
) -> dict[str, Any]:
    """比较 target_only/linear/tree 输出是否一致。"""
    target = method_outputs.get("target_only")
    request_ids = sorted({request_id for outputs in method_outputs.values() for request_id in outputs})
    matches_by_request: dict[str, dict[str, bool | None]] = {}
    for request_id in request_ids:
        matches_by_request[request_id] = {
            method: (
                tokens.get(request_id) == target.get(request_id)
                if target is not None and request_id in target
                else None
            )
            for method, tokens in method_outputs.items()
            if method != "target_only"
        }
    matches_target_only = {
        method: (
            all(tokens.get(request_id) == target.get(request_id) for request_id in request_ids)
            if target is not None
            else None
        )
        for method, tokens in method_outputs.items()
        if method != "target_only"
    }
    summary = {
        "methods": sorted(method_outputs),
        "generated_ids_by_method": method_outputs,
        "matches_target_only": matches_target_only,
        "matches_by_request": matches_by_request,
        "request_count": len(request_ids),
    }
    if method_results:
        method_efficiency = {
            method: _method_efficiency_metrics(result)
            for method, result in method_results.items()
        }
        summary["method_efficiency"] = method_efficiency
        summary["theory_checks"] = _theory_checks(method_efficiency, matches_target_only)
        summary["method_reproduction"] = _method_reproduction_report(
            method_results,
            method_efficiency,
            matches_target_only,
        )
    return summary


def _method_efficiency_metrics(result: RuntimeRunResult) -> dict[str, Any]:
    """从 artifacts 事件中提取 speculative decoding 理论相关指标。"""
    output_token_count = sum(len(request.output_token_ids) for request in result.request_results)
    proposal_count = sum(len(request.proposals) for request in result.request_results)
    http_total_ms = 0.0
    target_forward_event_count = 0
    target_forward_call_count = 0
    target_forward_total_ms = 0.0
    main_target_forward_event_count = 0
    main_target_forward_call_count = 0
    main_target_forward_total_ms = 0.0
    tree_choice_prefix_count = 0
    tree_choice_batch_event_count = 0
    tree_choice_unbatched_event_count = 0
    max_tree_choice_batch_size = 0
    main_tree_choice_prefix_count = 0
    main_tree_choice_batch_event_count = 0
    main_tree_choice_unbatched_event_count = 0
    max_main_tree_choice_batch_size = 0
    tree_attention_event_count = 0
    tree_root_guard_event_count = 0
    tree_root_guard_total_ms = 0.0
    tree_corrective_fallback_event_count = 0
    tree_corrective_fallback_total_ms = 0.0
    tree_backend_fallback_event_count = 0
    tree_backend_fallback_total_ms = 0.0
    proactive_reused_token_count = 0
    proactive_discarded_token_count = 0
    proactive_alignment_count = 0
    proactive_reconcile_count = 0
    verify_batch_sizes: list[int] = []
    queue_wait_total_ms = 0.0
    modeled_upload_total_ms = 0.0
    modeled_downlink_total_ms = 0.0
    runtime_round_total_ms = 0.0
    round_count = 0
    planner_wait_total_ms = 0.0
    draft_ready_wait_total_ms = 0.0
    dip_sd_solver_total_ms = 0.0
    dip_sd_solver_cache_hit_count = 0
    dip_sd_solver_shape_cache_hit_count = 0
    dip_sd_offline_plan_table_hit_count = 0
    dip_sd_solver_event_count = 0
    dip_sd_requested_solver_mode = None
    dip_sd_solver_mode = None
    dip_sd_solver_backend_name = None
    dip_sd_online_solver_enabled = None
    dip_sd_paper_solver_complete = None
    dip_sd_solver_backend_fallback_used = None
    dip_sd_solver_backend_fallback_reason = None
    dip_sd_latency_calibration_profile = None
    dip_sd_latency_calibration_applied = None
    dip_sd_acceptance_feedback_enabled = None
    dip_sd_acceptance_feedback_applied_count = 0
    dip_sd_acceptance_feedback_request_count = 0
    negative_network_residual_count = 0
    tree_forward_batch_kinds: set[str] = set()
    linear_forward_batch_kinds: set[str] = set()
    linear_shared_forward_ids: set[str] = set()
    draft_worker_ids: set[str] = set()
    target_verify_urls: set[str] = set()
    verify_intervals: set[tuple[int, int, str | None]] = set()
    proactive_intervals: list[tuple[int, int]] = []
    candidate_accept_event_count = 0
    candidate_winner_count = 0
    candidate_loser_count = 0
    candidate_count_observations: list[int] = []
    dynamic_draft_event_count = 0
    dynamic_confidence_stop_count = 0
    draft_confidences: list[float] = []
    verify_timeout_count = 0
    verify_retry_enqueue_count = 0
    verify_retry_exhausted_count = 0
    fallback_release_count = 0
    fallback_released_token_count = 0
    sled_static_queue_batch_count = 0
    sled_static_queue_padding_token_count = 0
    sled_static_queue_token_slots = 0
    sled_static_queue_max_wait_ms = 0.0
    steady_state_prefetch_submit_count = 0
    steady_state_prefetch_reuse_event_count = 0
    steady_state_prefetch_reused_draft_count = 0
    steady_state_prefetch_truncated_reuse_count = 0
    steady_state_prefetch_original_draft_token_count = 0
    steady_state_prefetch_reused_draft_token_count = 0
    steady_state_prefetch_discard_count = 0
    for event in result.events.events:
        if event.phase.startswith("draft.") and event.worker_id:
            draft_worker_ids.add(str(event.worker_id))
        if event.phase == "draft.generate":
            metadata = dict(event.metadata or {})
            if metadata.get("steady_state_prefetch"):
                steady_state_prefetch_reused_draft_count += 1
                reused_tokens = int(event.tokens_out or metadata.get("prefetch_reused_budget_tokens") or 0)
                original_tokens = int(metadata.get("prefetch_original_draft_length") or reused_tokens)
                steady_state_prefetch_reused_draft_token_count += reused_tokens
                steady_state_prefetch_original_draft_token_count += original_tokens
                if metadata.get("prefetch_truncated"):
                    steady_state_prefetch_truncated_reuse_count += 1
            if metadata.get("dynamic_drafting"):
                dynamic_draft_event_count += 1
                if metadata.get("dynamic_stop_reason") == "confidence_below_threshold":
                    dynamic_confidence_stop_count += 1
                for value in metadata.get("draft_confidences") or []:
                    draft_confidences.append(float(value))
        if event.phase == "draft.token_forward":
            metadata = dict(event.metadata or {})
            if metadata.get("draft_confidence") is not None:
                draft_confidences.append(float(metadata["draft_confidence"]))
        if event.phase == "runtime.round_total" and event.span_kind == "aggregate":
            runtime_round_total_ms += float(event.measured_duration_ms or event.duration_ms or 0.0)
            round_count += 1
        if event.phase == "pipeline.planner_wait":
            planner_wait_total_ms += float(event.measured_duration_ms or event.duration_ms or 0.0)
        if event.phase in {"pipeline.server_idle", "pipeline.draft_ready_wait"}:
            draft_ready_wait_total_ms += float(event.measured_duration_ms or event.duration_ms or 0.0)
        if event.phase == "pipeline.steady_state_prefetch_submit":
            steady_state_prefetch_submit_count += 1
        if event.phase == "pipeline.steady_state_prefetch_reuse":
            steady_state_prefetch_reuse_event_count += 1
        if event.phase == "draft.prefetch_discard":
            steady_state_prefetch_discard_count += 1
        if event.phase == "dip_sd.solver":
            dip_sd_solver_total_ms += float(event.measured_duration_ms or event.duration_ms or 0.0)
            dip_sd_solver_event_count += 1
            hints_metadata = dict(dict(event.metadata or {}).get("hints_metadata") or {})
            if hints_metadata.get("solver_cache_hit"):
                dip_sd_solver_cache_hit_count += 1
            if hints_metadata.get("solver_cache_hit") and hints_metadata.get("solver_cache_shape_level"):
                dip_sd_solver_shape_cache_hit_count += 1
            if hints_metadata.get("offline_plan_table_hit"):
                dip_sd_offline_plan_table_hit_count += 1
            dip_sd_requested_solver_mode = hints_metadata.get("requested_solver_mode", dip_sd_requested_solver_mode)
            dip_sd_solver_mode = hints_metadata.get("solver_mode", dip_sd_solver_mode)
            dip_sd_solver_backend_name = hints_metadata.get("solver_backend_name", dip_sd_solver_backend_name)
            dip_sd_online_solver_enabled = hints_metadata.get(
                "online_solver_enabled",
                dip_sd_online_solver_enabled,
            )
            dip_sd_paper_solver_complete = hints_metadata.get("paper_solver_complete", dip_sd_paper_solver_complete)
            dip_sd_solver_backend_fallback_used = hints_metadata.get(
                "solver_backend_fallback_used",
                dip_sd_solver_backend_fallback_used,
            )
            dip_sd_solver_backend_fallback_reason = hints_metadata.get(
                "solver_backend_fallback_reason",
                dip_sd_solver_backend_fallback_reason,
            )
            dip_sd_latency_calibration_profile = hints_metadata.get(
                "latency_calibration_profile",
                dip_sd_latency_calibration_profile,
            )
            dip_sd_latency_calibration_applied = hints_metadata.get(
                "latency_calibration_applied",
                dip_sd_latency_calibration_applied,
            )
            dip_sd_acceptance_feedback_enabled = hints_metadata.get(
                "acceptance_feedback_enabled",
                dip_sd_acceptance_feedback_enabled,
            )
            applied_count = int(hints_metadata.get("acceptance_feedback_applied_count") or 0)
            dip_sd_acceptance_feedback_applied_count += applied_count
            feedback_by_request = hints_metadata.get("acceptance_feedback_by_request")
            if isinstance(feedback_by_request, dict):
                dip_sd_acceptance_feedback_request_count = max(
                    dip_sd_acceptance_feedback_request_count,
                    len(feedback_by_request),
                )
        if event.phase == "pipeline.reconcile":
            proactive_reconcile_count += 1
            metadata = dict(event.metadata or {})
            proactive_reused_token_count += int(metadata.get("reused_token_count") or 0)
            proactive_discarded_token_count += int(metadata.get("discarded_token_count") or 0)
            proactive_alignment_count += 1 if metadata.get("aligned") else 0
        if event.phase == "draft.proactive" and event.start_ns is not None and event.end_ns is not None:
            proactive_intervals.append((int(event.start_ns), int(event.end_ns)))
        if event.phase == "accept.apply":
            metadata = dict(event.metadata or {})
            if metadata.get("candidate_count") is not None:
                candidate_accept_event_count += 1
                candidate_count = int(metadata.get("candidate_count") or 0)
                candidate_count_observations.append(candidate_count)
                if metadata.get("candidate_winner"):
                    candidate_winner_count += 1
                elif candidate_count > 1:
                    candidate_loser_count += 1
        if event.phase == "verify.timeout":
            verify_timeout_count += 1
        if event.phase == "verify.retry_enqueue":
            verify_retry_enqueue_count += 1
        if event.phase == "verify.retry_exhausted":
            verify_retry_exhausted_count += 1
        if event.phase == "verify.fallback_release":
            fallback_release_count += 1
        if event.phase == "verify.batch_total":
            metadata = dict(event.metadata or {})
            fallback_released_token_count += int(metadata.get("fallback_released_token_count") or 0)
            if metadata.get("sled_static_queue"):
                sled_static_queue_batch_count += 1
                waits = [float(value) for value in dict(metadata.get("queue_wait_ms_by_request") or {}).values()]
                queue_wait_total_ms += sum(waits)
                sled_static_queue_max_wait_ms = max(sled_static_queue_max_wait_ms, max(waits, default=0.0))
                sled_static_queue_padding_token_count += int(metadata.get("padding_token_count") or 0)
                sled_static_queue_token_slots += int(metadata.get("token_slots") or 0)
        if event.phase == "verify.http_total" and event.start_ns is not None and event.end_ns is not None:
            verify_intervals.add((int(event.start_ns), int(event.end_ns), event.batch_id))
            metadata = dict(event.metadata or {})
            if metadata.get("target_pool_url") is not None:
                target_verify_urls.add(str(metadata["target_pool_url"]))
            elif metadata.get("url") is not None:
                target_verify_urls.add(str(metadata["url"]).rsplit("/", 1)[0])
            modeled_upload_total_ms += float(metadata.get("modeled_upload_ms") or 0.0)
            modeled_downlink_total_ms += float(metadata.get("modeled_downlink_ms") or 0.0)
            residual = metadata.get("network_or_queue_residual_ms")
            if residual is not None and float(residual) < 0:
                negative_network_residual_count += 1
        if event.phase not in {"verify.http_total", "target.generate_total"}:
            continue
        if event.phase == "target.generate_total":
            http_total_ms += float(event.duration_ms or 0.0)
        timing = dict((event.metadata or {}).get("response_timing") or {})
        tree_forward_batch_kinds.update(_tree_forward_batch_kinds_from_timing(timing))
        for kind in timing.get("linear_forward_batch_kinds", []):
            linear_forward_batch_kinds.add(str(kind))
        if timing.get("linear_forward_batch_kind"):
            linear_forward_batch_kinds.add(str(timing["linear_forward_batch_kind"]))
        if timing.get("batch_size") is not None:
            verify_batch_sizes.append(int(timing.get("batch_size") or 0))
        queue_wait_total_ms += float(timing.get("queue_wait_ms") or 0.0)
        if "target_forward_events" in timing:
            forward_events = list(timing.get("target_forward_events") or [])
            linear_call_count = _linear_forward_call_count(forward_events, linear_shared_forward_ids)
            target_forward_event_count += len(forward_events)
            target_forward_call_count += linear_call_count
            target_forward_total_ms += float(timing.get("target_forward_total_ms") or 0.0)
            main_target_forward_event_count += len(forward_events)
            main_target_forward_call_count += linear_call_count
            main_target_forward_total_ms += float(timing.get("target_forward_total_ms") or 0.0)
            continue
        if "target_tree_forward_events" in timing:
            forward_events = list(timing.get("target_tree_forward_events") or [])
            target_forward_event_count += len(forward_events)
            target_forward_total_ms += float(timing.get("target_tree_forward_total_ms") or 0.0)
            for forward_event in forward_events:
                kind = str(forward_event.get("kind", ""))
                event_duration_ms = float(forward_event.get("duration_ms") or 0.0)
                event_call_count = _estimated_target_call_count(forward_event)
                target_forward_call_count += event_call_count
                is_corrective_fallback = bool(
                    dict(forward_event.get("metadata") or {}).get("fallback_reason")
                )
                is_batch_tree_choice = _is_batch_tree_choice_event(forward_event)
                if is_batch_tree_choice:
                    tree_choice_batch_event_count += 1
                    batch_size = _tree_choice_width(forward_event)
                    tree_choice_prefix_count += batch_size
                    max_tree_choice_batch_size = max(max_tree_choice_batch_size, batch_size)
                elif kind == "tree_choice":
                    tree_choice_unbatched_event_count += 1
                    tree_choice_prefix_count += 1
                if kind.startswith("tree_attention"):
                    tree_attention_event_count += 1
                if "fallback" in kind and not is_corrective_fallback:
                    tree_backend_fallback_event_count += 1
                    tree_backend_fallback_total_ms += event_duration_ms
                if kind == "tree_root_guard":
                    tree_root_guard_event_count += 1
                    tree_root_guard_total_ms += event_duration_ms
                    continue
                if is_corrective_fallback:
                    tree_corrective_fallback_event_count += 1
                    tree_corrective_fallback_total_ms += event_duration_ms
                    continue
                main_target_forward_event_count += 1
                main_target_forward_call_count += event_call_count
                main_target_forward_total_ms += event_duration_ms
                if is_batch_tree_choice:
                    batch_size = _tree_choice_width(forward_event)
                    main_tree_choice_batch_event_count += event_call_count
                    main_tree_choice_prefix_count += batch_size
                    max_main_tree_choice_batch_size = max(max_main_tree_choice_batch_size, batch_size)
                elif kind == "tree_choice":
                    main_tree_choice_unbatched_event_count += 1
                    main_tree_choice_prefix_count += 1
                elif kind == "sequential_next_token_fallback":
                    batch_size = _tree_choice_width(forward_event)
                    main_tree_choice_unbatched_event_count += batch_size
                    main_tree_choice_prefix_count += batch_size
    target_calls_per_output_token = (
        target_forward_call_count / output_token_count
        if output_token_count
        else None
    )
    main_target_calls_per_output_token = (
        main_target_forward_call_count / output_token_count
        if output_token_count
        else None
    )
    tree_batch_compression_ratio = (
        tree_choice_prefix_count / (tree_choice_batch_event_count + tree_choice_unbatched_event_count)
        if tree_choice_prefix_count and (tree_choice_batch_event_count + tree_choice_unbatched_event_count)
        else None
    )
    main_tree_batch_compression_ratio = (
        main_tree_choice_prefix_count
        / (main_tree_choice_batch_event_count + main_tree_choice_unbatched_event_count)
        if main_tree_choice_prefix_count
        and (main_tree_choice_batch_event_count + main_tree_choice_unbatched_event_count)
        else None
    )
    overlap_ms = _interval_overlap_ms(list(verify_intervals), proactive_intervals)
    proactive_total_ms = sum((end_ns - start_ns) / 1_000_000 for start_ns, end_ns in proactive_intervals)
    overlap_ratio = overlap_ms / proactive_total_ms if proactive_total_ms > 0 else None
    proactive_alignment_rate = (
        proactive_alignment_count / proactive_reconcile_count
        if proactive_reconcile_count
        else None
    )
    avg_verify_batch_size = (
        sum(verify_batch_sizes) / len(verify_batch_sizes)
        if verify_batch_sizes
        else None
    )
    server_idle_gap_ms = _interval_idle_gap_ms(list(verify_intervals))
    http_total_ms += sum((end_ns - start_ns) / 1_000_000 for start_ns, end_ns, _batch_id in verify_intervals)
    effective_total_ms = runtime_round_total_ms if runtime_round_total_ms > 0 else http_total_ms
    wstgr_tokens_per_s = (
        output_token_count / (effective_total_ms / 1000.0)
        if effective_total_ms > 0
        else None
    )
    output_tokens_per_target_call = (
        output_token_count / target_forward_call_count
        if target_forward_call_count
        else None
    )
    setup_metrics = _setup_metrics(result)
    return {
        "output_token_count": output_token_count,
        "proposal_count": proposal_count,
        "http_total_ms": round(http_total_ms, 6),
        "target_forward_event_count": target_forward_event_count,
        "target_forward_call_count": target_forward_call_count,
        "target_forward_total_ms": round(target_forward_total_ms, 6),
        "target_calls_per_output_token": None
        if target_calls_per_output_token is None
        else round(target_calls_per_output_token, 6),
        "main_target_forward_event_count": main_target_forward_event_count,
        "main_target_forward_call_count": main_target_forward_call_count,
        "main_target_forward_total_ms": round(main_target_forward_total_ms, 6),
        "main_target_calls_per_output_token": None
        if main_target_calls_per_output_token is None
        else round(main_target_calls_per_output_token, 6),
        "tree_choice_prefix_count": tree_choice_prefix_count,
        "tree_choice_batch_event_count": tree_choice_batch_event_count,
        "tree_choice_unbatched_event_count": tree_choice_unbatched_event_count,
        "tree_batch_compression_ratio": None
        if tree_batch_compression_ratio is None
        else round(tree_batch_compression_ratio, 6),
        "max_tree_choice_batch_size": max_tree_choice_batch_size,
        "main_tree_choice_prefix_count": main_tree_choice_prefix_count,
        "main_tree_choice_batch_event_count": main_tree_choice_batch_event_count,
        "main_tree_choice_unbatched_event_count": main_tree_choice_unbatched_event_count,
        "main_tree_batch_compression_ratio": None
        if main_tree_batch_compression_ratio is None
        else round(main_tree_batch_compression_ratio, 6),
        "max_main_tree_choice_batch_size": max_main_tree_choice_batch_size,
        "tree_attention_event_count": tree_attention_event_count,
        "tree_root_guard_event_count": tree_root_guard_event_count,
        "tree_root_guard_total_ms": round(tree_root_guard_total_ms, 6),
        "tree_corrective_fallback_event_count": tree_corrective_fallback_event_count,
        "tree_corrective_fallback_total_ms": round(tree_corrective_fallback_total_ms, 6),
        "tree_backend_fallback_event_count": tree_backend_fallback_event_count,
        "tree_backend_fallback_total_ms": round(tree_backend_fallback_total_ms, 6),
        "proactive_alignment_rate": None
        if proactive_alignment_rate is None
        else round(proactive_alignment_rate, 6),
        "proactive_draft_event_count": len(proactive_intervals),
        "proactive_reconcile_count": proactive_reconcile_count,
        "proactive_reused_token_count": proactive_reused_token_count,
        "proactive_discarded_token_count": proactive_discarded_token_count,
        "overlap_ratio": None if overlap_ratio is None else round(overlap_ratio, 6),
        "server_idle_gap_ms": round(server_idle_gap_ms, 6),
        "avg_verify_batch_size": None if avg_verify_batch_size is None else round(avg_verify_batch_size, 6),
        "queue_wait_total_ms": round(queue_wait_total_ms, 6),
        "modeled_upload_ms": round(modeled_upload_total_ms, 6),
        "modeled_downlink_ms": round(modeled_downlink_total_ms, 6),
        "runtime_round_total_ms": round(runtime_round_total_ms, 6),
        "planner_wait_total_ms": round(planner_wait_total_ms, 6),
        "draft_ready_wait_total_ms": round(draft_ready_wait_total_ms, 6),
        "dip_sd_solver_total_ms": round(dip_sd_solver_total_ms, 6),
        "dip_sd_solver_event_count": dip_sd_solver_event_count,
        "dip_sd_solver_cache_hit_count": dip_sd_solver_cache_hit_count,
        "dip_sd_solver_shape_cache_hit_count": dip_sd_solver_shape_cache_hit_count,
        "dip_sd_offline_plan_table_hit_count": dip_sd_offline_plan_table_hit_count,
        "dip_sd_requested_solver_mode": dip_sd_requested_solver_mode,
        "dip_sd_solver_mode": dip_sd_solver_mode,
        "dip_sd_solver_backend_name": dip_sd_solver_backend_name,
        "dip_sd_online_solver_enabled": dip_sd_online_solver_enabled,
        "dip_sd_paper_solver_complete": dip_sd_paper_solver_complete,
        "dip_sd_solver_backend_fallback_used": dip_sd_solver_backend_fallback_used,
        "dip_sd_solver_backend_fallback_reason": dip_sd_solver_backend_fallback_reason,
        "dip_sd_latency_calibration_profile": dip_sd_latency_calibration_profile,
        "dip_sd_latency_calibration_applied": dip_sd_latency_calibration_applied,
        "dip_sd_acceptance_feedback_enabled": dip_sd_acceptance_feedback_enabled,
        "dip_sd_acceptance_feedback_applied_count": dip_sd_acceptance_feedback_applied_count,
        "dip_sd_acceptance_feedback_request_count": dip_sd_acceptance_feedback_request_count,
        "steady_state_prefetch_submit_count": steady_state_prefetch_submit_count,
        "steady_state_prefetch_reuse_event_count": steady_state_prefetch_reuse_event_count,
        "steady_state_prefetch_reused_draft_count": steady_state_prefetch_reused_draft_count,
        "steady_state_prefetch_truncated_reuse_count": steady_state_prefetch_truncated_reuse_count,
        "steady_state_prefetch_original_draft_token_count": steady_state_prefetch_original_draft_token_count,
        "steady_state_prefetch_reused_draft_token_count": steady_state_prefetch_reused_draft_token_count,
        "steady_state_prefetch_discard_count": steady_state_prefetch_discard_count,
        "effective_total_ms": round(effective_total_ms, 6),
        "wstgr_tokens_per_s": None if wstgr_tokens_per_s is None else round(wstgr_tokens_per_s, 6),
        "output_tokens_per_target_call": None
        if output_tokens_per_target_call is None
        else round(output_tokens_per_target_call, 6),
        "round_count": round_count,
        "tree_forward_batch_kinds": sorted(tree_forward_batch_kinds),
        "linear_forward_batch_kinds": sorted(linear_forward_batch_kinds),
        "negative_network_residual_count": negative_network_residual_count,
        "draft_worker_count": len(draft_worker_ids),
        "draft_worker_ids": sorted(draft_worker_ids),
        "target_verify_urls": sorted(target_verify_urls),
        "target_verify_url_count": len(target_verify_urls),
        "candidate_accept_event_count": candidate_accept_event_count,
        "candidate_winner_count": candidate_winner_count,
        "candidate_loser_count": candidate_loser_count,
        "avg_candidate_count": None
        if not candidate_count_observations
        else round(sum(candidate_count_observations) / len(candidate_count_observations), 6),
        "dynamic_draft_event_count": dynamic_draft_event_count,
        "dynamic_confidence_stop_count": dynamic_confidence_stop_count,
        "avg_draft_confidence": None
        if not draft_confidences
        else round(sum(draft_confidences) / len(draft_confidences), 6),
        "verify_timeout_count": verify_timeout_count,
        "verify_retry_enqueue_count": verify_retry_enqueue_count,
        "verify_retry_exhausted_count": verify_retry_exhausted_count,
        "fallback_release_count": fallback_release_count,
        "fallback_released_token_count": fallback_released_token_count,
        "sled_static_queue_batch_count": sled_static_queue_batch_count,
        "sled_static_queue_padding_token_count": sled_static_queue_padding_token_count,
        "sled_static_queue_token_slots": sled_static_queue_token_slots,
        "sled_static_queue_max_wait_ms": round(sled_static_queue_max_wait_ms, 6),
        "raw_target_forward_call_count": target_forward_call_count,
        "setup_load_draft_model_ms": setup_metrics["setup_load_draft_model_ms"],
        "setup_warm_draft_workers_ms": setup_metrics["setup_warm_draft_workers_ms"],
        "setup_warm_draft_worker_event_count": setup_metrics["setup_warm_draft_worker_event_count"],
        "setup_total_ms": setup_metrics["setup_total_ms"],
    }


def _setup_metrics(result: RuntimeRunResult) -> dict[str, Any]:
    """Summarize setup-layer timings without mixing them into runtime totals."""
    load_ms = 0.0
    warm_workers_ms = 0.0
    aggregate_child_count = 0
    direct_child_count = 0
    for event in result.events.events:
        if event.span_kind != "setup":
            continue
        duration = float(event.measured_duration_ms or event.duration_ms or 0.0)
        if event.phase == "setup.load_draft_model":
            load_ms += duration
        elif event.phase == "setup.warm_draft_workers":
            warm_workers_ms += duration
            metadata = dict(event.metadata or {})
            aggregate_child_count += int(metadata.get("warmup_worker_event_count") or 0)
        elif event.phase == "setup.warm_draft_worker":
            direct_child_count += 1
    return {
        "setup_load_draft_model_ms": round(load_ms, 6),
        "setup_warm_draft_workers_ms": round(warm_workers_ms, 6),
        "setup_warm_draft_worker_event_count": aggregate_child_count or direct_child_count,
        "setup_total_ms": round(load_ms + warm_workers_ms, 6),
    }


def _tree_forward_batch_kinds_from_timing(timing: dict[str, Any]) -> set[str]:
    """Collect backend tree-batch kinds from both timing and per-forward metadata."""
    kinds: set[str] = set()

    def add(value: Any) -> None:
        if value:
            kinds.add(str(value))

    def add_many(values: Any) -> None:
        if isinstance(values, (list, tuple, set)):
            for value in values:
                add(value)

    add(timing.get("tree_forward_batch_kind"))
    add_many(timing.get("tree_forward_batch_kinds"))
    for raw_forward_event in timing.get("target_tree_forward_events") or []:
        if not isinstance(raw_forward_event, dict):
            continue
        event_kinds_before = len(kinds)
        add(raw_forward_event.get("tree_forward_batch_kind"))
        metadata = raw_forward_event.get("metadata")
        if isinstance(metadata, dict):
            add(metadata.get("tree_forward_batch_kind"))
            add_many(metadata.get("tree_forward_batch_kinds"))
        kind = str(raw_forward_event.get("kind") or "")
        if len(kinds) == event_kinds_before and kind.startswith(("tree_attention_", "tree_choice_batch_")):
            add(kind)
    return kinds


def _interval_overlap_ms(
    verify_intervals: list[tuple[int, int, str | None]],
    proactive_intervals: list[tuple[int, int]],
) -> float:
    total_ns = 0
    for verify_start, verify_end, _batch_id in verify_intervals:
        for proactive_start, proactive_end in proactive_intervals:
            total_ns += max(0, min(verify_end, proactive_end) - max(verify_start, proactive_start))
    return total_ns / 1_000_000


def _interval_idle_gap_ms(verify_intervals: list[tuple[int, int, str | None]]) -> float:
    unique = sorted((start, end) for start, end, _batch_id in verify_intervals)
    if len(unique) <= 1:
        return 0.0
    merged: list[tuple[int, int]] = []
    for start, end in unique:
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return sum(max(0, start - previous_end) for (_previous_start, previous_end), (start, _end) in zip(merged, merged[1:])) / 1_000_000


def _tree_choice_width(forward_event: dict[str, Any]) -> int:
    """返回一次 tree choice forward 覆盖的 parent prefix 数。"""
    for key in ("choice_count", "batch_size", "parent_prefix_count"):
        value = forward_event.get(key)
        if value is not None:
            return int(value)
    token_ids = forward_event.get("token_ids")
    if isinstance(token_ids, list):
        return len(token_ids)
    return 1


def _is_batch_tree_choice_event(forward_event: dict[str, Any]) -> bool:
    """判断事件是否用一次 target forward 覆盖了多个 tree parent choice。"""
    kind = str(forward_event.get("kind", ""))
    return (
        kind in {"tree_choice_batch", "tree_attention", "tree_forward", "batched_next_token_fallback"}
        or kind.startswith("tree_attention_")
    )


def _estimated_target_call_count(forward_event: dict[str, Any]) -> int:
    """估计事件背后真实 target forward 调用次数。"""
    kind = str(forward_event.get("kind", ""))
    if kind == "sequential_next_token_fallback":
        return _tree_choice_width(forward_event)
    metadata = dict(forward_event.get("metadata") or {})
    if metadata.get("shared_batch_event_id") and int(forward_event.get("batch_index") or metadata.get("batch_index") or 0) > 0:
        return 0
    return 1


def _linear_forward_call_count(forward_events: list[dict[str, Any]], seen_shared: set[str]) -> int:
    """Count shared linear batch forwards once across item-attributed events."""
    calls = 0
    for event in forward_events:
        shared_id = event.get("shared_batch_event_id")
        if shared_id:
            shared_id = str(shared_id)
            if shared_id in seen_shared:
                continue
            seen_shared.add(shared_id)
        calls += max(0, int(event.get("target_forward_call_count") or 1))
    return calls


def _theory_checks(
    method_efficiency: dict[str, dict[str, Any]],
    matches_target_only: dict[str, bool | None],
) -> dict[str, Any]:
    """生成和推测解码理论相关的机器可读检查项。"""
    linear = method_efficiency.get("linear", {})
    stop_wait_tree = method_efficiency.get("tree_stop_wait") or method_efficiency.get("tree") or {}
    specedge = method_efficiency.get("specedge_pipeline") or {}
    tree = stop_wait_tree or specedge
    linear_calls = int(linear.get("main_target_forward_call_count") or linear.get("target_forward_call_count") or 0)
    tree_main_calls = int(tree.get("main_target_forward_call_count") or 0)
    tree_raw_calls = int(tree.get("target_forward_call_count") or tree.get("target_forward_event_count") or 0)
    tree_choice_prefixes = int(tree.get("main_tree_choice_prefix_count") or tree.get("tree_choice_prefix_count") or 0)
    tree_main_call_reduction_vs_linear = (
        1.0 - (tree_main_calls / linear_calls)
        if linear_calls
        else None
    )
    tree_raw_call_reduction_vs_linear = (
        1.0 - (tree_raw_calls / linear_calls)
        if linear_calls
        else None
    )
    tree_call_reduction_vs_unbatched_tree = (
        1.0 - (tree_main_calls / tree_choice_prefixes)
        if tree_choice_prefixes
        else None
    )
    linear_forward_ms = float(linear.get("main_target_forward_total_ms") or linear.get("target_forward_total_ms") or 0.0)
    tree_main_forward_ms = float(tree.get("main_target_forward_total_ms") or 0.0)
    tree_raw_forward_ms = float(tree.get("target_forward_total_ms") or 0.0)
    linear_http_ms = float(linear.get("http_total_ms") or 0.0)
    tree_http_ms = float(tree.get("http_total_ms") or 0.0)
    specedge_main_calls = int(specedge.get("main_target_forward_call_count") or 0)
    specedge_runtime_ms = float(specedge.get("runtime_round_total_ms") or 0.0)
    stop_wait_runtime_ms = float(stop_wait_tree.get("runtime_round_total_ms") or 0.0)
    specedge_server_idle_gap_ms = float(specedge.get("server_idle_gap_ms") or 0.0)
    stop_wait_server_idle_gap_ms = float(stop_wait_tree.get("server_idle_gap_ms") or 0.0)
    specedge_speedup_vs_stop_wait = (
        stop_wait_runtime_ms / specedge_runtime_ms
        if stop_wait_runtime_ms and specedge_runtime_ms
        else None
    )
    return {
        "linear_matches_target_only": matches_target_only.get("linear"),
        "tree_matches_target_only": (
            matches_target_only.get("tree_stop_wait", matches_target_only.get("tree"))
            if stop_wait_tree
            else matches_target_only.get("specedge_pipeline")
        ),
        "tree_uses_batched_choice_forward": bool(
            tree.get("main_tree_choice_batch_event_count") or tree.get("tree_choice_batch_event_count")
        ),
        "tree_uses_tree_attention": bool(tree.get("tree_attention_event_count")),
        "tree_target_call_count_lte_linear": (
            tree_main_calls <= linear_calls
            if linear_calls and tree_main_calls
            else None
        ),
        "tree_main_target_call_count_lte_linear": (
            tree_main_calls <= linear_calls
            if linear_calls and tree_main_calls
            else None
        ),
        "tree_raw_target_call_count_lte_linear": (
            tree_raw_calls <= linear_calls
            if linear_calls and tree_raw_calls
            else None
        ),
        "tree_main_forward_ms_lte_linear": (
            tree_main_forward_ms <= linear_forward_ms
            if linear_forward_ms and tree_main_forward_ms
            else None
        ),
        "tree_raw_forward_ms_lte_linear": (
            tree_raw_forward_ms <= linear_forward_ms
            if linear_forward_ms and tree_raw_forward_ms
            else None
        ),
        "tree_http_total_ms_lte_linear": (
            tree_http_ms <= linear_http_ms
            if linear_http_ms and tree_http_ms
            else None
        ),
        "tree_guard_overhead_present": bool(tree.get("tree_root_guard_event_count")),
        "tree_corrective_fallback_present": bool(tree.get("tree_corrective_fallback_event_count")),
        "tree_backend_fallback_present": bool(tree.get("tree_backend_fallback_event_count")),
        "tree_call_reduction_vs_linear": None
        if tree_main_call_reduction_vs_linear is None
        else round(tree_main_call_reduction_vs_linear, 6),
        "tree_main_call_reduction_vs_linear": None
        if tree_main_call_reduction_vs_linear is None
        else round(tree_main_call_reduction_vs_linear, 6),
        "tree_raw_call_reduction_vs_linear": None
        if tree_raw_call_reduction_vs_linear is None
        else round(tree_raw_call_reduction_vs_linear, 6),
        "tree_call_reduction_vs_unbatched_tree": None
        if tree_call_reduction_vs_unbatched_tree is None
        else round(tree_call_reduction_vs_unbatched_tree, 6),
        "specedge_matches_target_only": matches_target_only.get("specedge_pipeline"),
        "specedge_has_proactive_overlap": (
            float(specedge.get("overlap_ratio") or 0.0) > 0.0
            if specedge
            else None
        ),
        "specedge_avg_verify_batch_size_gt_one": (
            float(specedge.get("avg_verify_batch_size") or 0.0) > 1.0
            if specedge
            else None
        ),
        "specedge_main_target_call_count_lte_linear": (
            specedge_main_calls <= linear_calls
            if specedge and linear_calls and specedge_main_calls
            else None
        ),
        "specedge_server_idle_gap_lt_stop_wait": (
            specedge_server_idle_gap_ms < stop_wait_server_idle_gap_ms
            if specedge and stop_wait_tree
            else None
        ),
        "specedge_runtime_total_ms_lte_stop_wait": (
            specedge_runtime_ms <= stop_wait_runtime_ms
            if specedge and stop_wait_tree and specedge_runtime_ms and stop_wait_runtime_ms
            else None
        ),
        "specedge_runtime_speedup_vs_stop_wait": None
        if specedge_speedup_vs_stop_wait is None
        else round(specedge_speedup_vs_stop_wait, 6),
        "specedge_tree_forward_batch_kind_recorded": (
            bool(specedge.get("tree_forward_batch_kinds"))
            if specedge
            else None
        ),
        "specedge_timing_residual_nonnegative": (
            int(specedge.get("negative_network_residual_count") or 0) == 0
            if specedge
            else None
        ),
        "specedge_proactive_reuse_observed": (
            int(specedge.get("proactive_reused_token_count") or 0) > 0
            if specedge
            else None
        ),
    }


def _method_reproduction_report(
    method_results: dict[str, RuntimeRunResult],
    method_efficiency: dict[str, dict[str, Any]],
    matches_target_only: dict[str, bool | None],
) -> dict[str, dict[str, Any]]:
    """Report original-method fidelity separately from optimization ideas."""
    report: dict[str, dict[str, Any]] = {}
    for method, metrics in method_efficiency.items():
        signals = _method_execution_signals(method_results.get(method))
        if method == "target_only":
            report[method] = {
                "reference_scope": "target autoregressive baseline",
                "execution_mode": signals["execution_mode"],
                "matches_target_only": True,
                "implemented": ["server-side greedy target generation"],
                "partial_or_missing": [],
                "not_counted_as_original": [],
            }
        elif method == "linear":
            report[method] = _linear_reproduction_status(metrics, signals, matches_target_only.get(method))
        elif method in {"tree", "tree_stop_wait"}:
            report[method] = _tree_stop_wait_reproduction_status(metrics, signals, matches_target_only.get(method))
        elif method == "specedge_pipeline":
            report[method] = _specedge_reproduction_status(metrics, signals, matches_target_only.get(method))
        elif method == "dip_sd":
            report[method] = _dip_sd_reproduction_status(metrics, signals, matches_target_only.get(method))
        elif method in {"sled", "sled_async"}:
            report[method] = _sled_reproduction_status(metrics, signals, matches_target_only.get(method))
        else:
            report[method] = {
                "reference_scope": "custom method",
                "execution_mode": signals["execution_mode"],
                "matches_target_only": matches_target_only.get(method),
                "implemented": [],
                "partial_or_missing": ["no reproduction checklist is registered for this method"],
                "not_counted_as_original": [],
                "signals": signals,
            }
    return report


def _method_execution_signals(result: RuntimeRunResult | None) -> dict[str, Any]:
    phases: set[str] = set()
    scheduler_hints: dict[str, Any] = {}
    scheduler_plan_metadata: dict[str, Any] = {}
    if result is None:
        return {
            "execution_mode": "unknown",
            "scheduler_hints": {},
            "scheduler_plan_metadata": {},
            "phase_names": [],
        }
    for event in result.events.events:
        phases.add(str(event.phase))
        if event.phase == "scheduler.plan":
            metadata = dict(event.metadata or {})
            scheduler_plan_metadata = dict(metadata.get("plan_metadata") or {})
            scheduler_hints = dict(metadata.get("hints_metadata") or metadata.get("planning_hints") or {})
    if "draft.proactive" in phases or "pipeline.reconcile" in phases:
        execution_mode = "async_pipeline"
    elif "pipeline.stage" in phases:
        execution_mode = "distributed_batch_pipeline"
    elif "runtime.round_total" in phases:
        execution_mode = "stop_wait_round_runtime"
    elif "target.generate_total" in phases:
        execution_mode = "target_only_http"
    else:
        execution_mode = "artifact_only"
    return {
        "execution_mode": execution_mode,
        "scheduler_hints": scheduler_hints,
        "scheduler_plan_metadata": scheduler_plan_metadata,
        "phase_names": sorted(phases),
    }


def _linear_reproduction_status(
    metrics: dict[str, Any],
    signals: dict[str, Any],
    matches_target_only: bool | None,
) -> dict[str, Any]:
    implemented = ["draft-then-verify linear speculative decoding"]
    if _positive(metrics.get("main_target_forward_call_count")):
        implemented.append("target verifies draft tokens")
    return {
        "reference_scope": "basic linear speculative decoding baseline",
        "execution_mode": signals["execution_mode"],
        "matches_target_only": matches_target_only,
        "implemented": implemented,
        "partial_or_missing": [],
        "not_counted_as_original": [],
        "signals": _compact_reproduction_signals(metrics, signals),
    }


def _tree_stop_wait_reproduction_status(
    metrics: dict[str, Any],
    signals: dict[str, Any],
    matches_target_only: bool | None,
) -> dict[str, Any]:
    implemented = ["tree speculative decoding baseline"]
    if _positive(metrics.get("tree_attention_event_count")):
        implemented.append("tree attention verification")
    if _positive(metrics.get("avg_verify_batch_size"), threshold=1.0):
        implemented.append("server batch verification")
    partial_or_missing = ["stop-wait baseline, not SpecEdge proactive pipeline"]
    return {
        "reference_scope": "tree stop-wait speculative decoding baseline",
        "execution_mode": signals["execution_mode"],
        "matches_target_only": matches_target_only,
        "implemented": implemented,
        "partial_or_missing": partial_or_missing,
        "not_counted_as_original": [],
        "signals": _compact_reproduction_signals(metrics, signals),
    }


def _specedge_reproduction_status(
    metrics: dict[str, Any],
    signals: dict[str, Any],
    matches_target_only: bool | None,
) -> dict[str, Any]:
    implemented = ["edge tree drafting", "A100/server batch verification"]
    partial_or_missing: list[str] = []
    if _positive(metrics.get("proactive_draft_event_count")):
        implemented.append("proactive edge drafting")
    else:
        partial_or_missing.append("proactive edge drafting was not observed")
    if _positive(metrics.get("overlap_ratio")):
        implemented.append("draft/verify overlap")
    else:
        partial_or_missing.append("draft/verify overlap was not observed")
    if _positive(metrics.get("avg_verify_batch_size"), threshold=1.0):
        implemented.append("pipeline-aware multi-request batch verification")
    else:
        partial_or_missing.append("multi-request server batch was not observed")
    if metrics.get("tree_forward_batch_kinds"):
        implemented.append("heterogeneous tree batch verify path is recorded")
    else:
        partial_or_missing.append("tree batch verify kind was not recorded")
    return {
        "reference_scope": "SpecEdge original core: proactive drafting plus server-side pipeline-aware batch verification",
        "execution_mode": signals["execution_mode"],
        "matches_target_only": matches_target_only,
        "implemented": implemented,
        "partial_or_missing": partial_or_missing,
        "not_counted_as_original": [],
        "signals": _compact_reproduction_signals(metrics, signals),
    }


def _dip_sd_reproduction_status(
    metrics: dict[str, Any],
    signals: dict[str, Any],
    matches_target_only: bool | None,
) -> dict[str, Any]:
    hints = dict(signals.get("scheduler_hints") or {})
    implemented = ["central server batch verification"]
    partial_or_missing: list[str] = []
    if _positive(metrics.get("draft_worker_count"), threshold=1.0):
        implemented.append("distributed draft workers")
    else:
        partial_or_missing.append("multi-device local drafting was not observed")
    if _positive(metrics.get("avg_verify_batch_size"), threshold=1.0):
        implemented.append("multi-user batch verification")
    else:
        partial_or_missing.append("multi-user verify batch was not observed")
    if metrics.get("linear_forward_batch_kinds"):
        implemented.append("batched linear target verification")
    else:
        partial_or_missing.append("batched linear target verification was not recorded")
    if hints.get("latency_calibration_applied"):
        implemented.append("paper-style latency profile calibration")
    if hints.get("joint_batch_assignment"):
        implemented.append("joint batch assignment planning")
    else:
        partial_or_missing.append("joint batch assignment planning was not observed")
    if hints.get("joint_draft_length"):
        implemented.append("joint draft-length planning")
    else:
        partial_or_missing.append("joint draft-length planning was not observed")
    if hints.get("paper_solver_complete"):
        implemented.append("paper MILP/Dinkelbach solver backend")
    else:
        backend = hints.get("solver_backend_name") or hints.get("solver_mode") or "unknown"
        requested = hints.get("requested_solver_mode") or "unknown"
        reason = hints.get("solver_backend_fallback_reason")
        if "dinkelbach" in str(backend):
            implemented.append("dependency-free Dinkelbach draft-length optimization")
            detail = f"paper MILP/SCIP backend is not active (requested={requested}, backend={backend})"
        else:
            detail = f"paper MILP/Dinkelbach solver backend is not active (requested={requested}, backend={backend})"
        if reason:
            detail += f": {reason}"
        partial_or_missing.append(detail)
    if signals.get("execution_mode") == "distributed_batch_pipeline":
        implemented.append("phase-level draft/verify pipeline")
        if _positive(metrics.get("steady_state_prefetch_reused_draft_count"), threshold=0.0):
            implemented.append("steady-state cross-round draft prefetch")
        elif _positive(metrics.get("steady_state_prefetch_submit_count"), threshold=0.0):
            partial_or_missing.append("steady-state draft prefetch was submitted but not reused in this run")
        else:
            partial_or_missing.append("steady-state cross-round draft prefetch was not observed")
    else:
        partial_or_missing.append("phase-level draft/verify pipeline is not active in the current reproduction run")
    return {
        "reference_scope": "DiP-SD original core: distributed local drafting, central batch verification, joint batch/draft-length optimization, and pipelining",
        "execution_mode": signals["execution_mode"],
        "matches_target_only": matches_target_only,
        "implemented": implemented,
        "partial_or_missing": partial_or_missing,
        "not_counted_as_original": [
            "SpecEdge proactive single-head drafting is not part of the DiP-SD reproduction report"
        ],
        "signals": _compact_reproduction_signals(metrics, signals),
    }


def _sled_reproduction_status(
    metrics: dict[str, Any],
    signals: dict[str, Any],
    matches_target_only: bool | None,
) -> dict[str, Any]:
    hints = dict(signals.get("scheduler_hints") or {})
    implemented = ["shared server batch verification"]
    partial_or_missing: list[str] = []
    if _positive(metrics.get("draft_worker_count"), threshold=1.0):
        implemented.append("heterogeneous draft worker registry")
    else:
        partial_or_missing.append("multiple draft workers were not observed")
    if hints.get("single_edge_device_per_request"):
        implemented.append("single edge-device draft stream per request")
    else:
        partial_or_missing.append("single edge-device request ownership was not observed")
    if hints.get("dynamic_drafting") or _positive(metrics.get("dynamic_draft_event_count"), threshold=0.0):
        implemented.append("confidence-triggered dynamic drafting")
    else:
        partial_or_missing.append("confidence-triggered dynamic drafting was not observed")
    if _positive(metrics.get("avg_verify_batch_size"), threshold=1.0):
        implemented.append("shared server verifies candidate batch")
    else:
        partial_or_missing.append("server candidate batch was not observed")
    if _positive(metrics.get("sled_static_queue_batch_count"), threshold=0.0):
        implemented.append("central static queue batch dispatch")
    else:
        partial_or_missing.append("central static queue dispatch was not observed")
    if metrics.get("linear_forward_batch_kinds"):
        implemented.append("batched linear target verification")
    if hints.get("edge_device_worker_assignment") or hints.get("heterogeneous_worker_assignment"):
        implemented.append("edge-device worker assignment planning")
    else:
        partial_or_missing.append("edge-device worker assignment planning was not observed")
    if signals.get("execution_mode") == "async_pipeline":
        implemented.append("async edge drafting during server verification")
        if _positive(metrics.get("overlap_ratio"), threshold=0.0):
            implemented.append("draft/verify overlap")
    else:
        partial_or_missing.append("async edge drafting was not active in this run")
    if _positive(metrics.get("target_verify_url_count"), threshold=0.0):
        implemented.append("configured A100 target verifier dispatch")
    else:
        partial_or_missing.append("configured A100 target verifier dispatch was not observed")
    phase_names = set(signals.get("phase_names") or [])
    if phase_names & {"verify.timeout", "verify.retry_enqueue", "verify.fallback_release"}:
        implemented.append("timeout/retry/fallback control path")
    if _positive(metrics.get("avg_candidate_count"), threshold=1.0):
        partial_or_missing.append("legacy multi-candidate-per-request behavior was still observed")
    not_counted = ["same-request multi-worker candidate selection is not counted as SLED original reproduction"]
    if signals.get("execution_mode") != "async_pipeline":
        not_counted.append("async/proactive overlap is absent from this stop-wait SLED run")
    return {
        "reference_scope": "SLED original core: edge-device local dynamic drafting, central request queue, and shared target server batched verification",
        "execution_mode": signals["execution_mode"],
        "matches_target_only": matches_target_only,
        "implemented": implemented,
        "partial_or_missing": partial_or_missing,
        "not_counted_as_original": not_counted,
        "signals": _compact_reproduction_signals(metrics, signals),
    }


def _compact_reproduction_signals(metrics: dict[str, Any], signals: dict[str, Any]) -> dict[str, Any]:
    hints = dict(signals.get("scheduler_hints") or {})
    return {
        "execution_mode": signals.get("execution_mode"),
        "draft_worker_count": metrics.get("draft_worker_count"),
        "target_verify_url_count": metrics.get("target_verify_url_count"),
        "target_verify_urls": metrics.get("target_verify_urls"),
        "avg_verify_batch_size": metrics.get("avg_verify_batch_size"),
        "avg_candidate_count": metrics.get("avg_candidate_count"),
        "overlap_ratio": metrics.get("overlap_ratio"),
        "proactive_draft_event_count": metrics.get("proactive_draft_event_count"),
        "proactive_reconcile_count": metrics.get("proactive_reconcile_count"),
        "dynamic_draft_event_count": metrics.get("dynamic_draft_event_count"),
        "dynamic_confidence_stop_count": metrics.get("dynamic_confidence_stop_count"),
        "avg_draft_confidence": metrics.get("avg_draft_confidence"),
        "verify_timeout_count": metrics.get("verify_timeout_count"),
        "verify_retry_enqueue_count": metrics.get("verify_retry_enqueue_count"),
        "fallback_release_count": metrics.get("fallback_release_count"),
        "tree_forward_batch_kinds": metrics.get("tree_forward_batch_kinds"),
        "linear_forward_batch_kinds": metrics.get("linear_forward_batch_kinds"),
        "sled_static_queue_batch_count": metrics.get("sled_static_queue_batch_count"),
        "sled_static_queue_padding_token_count": metrics.get("sled_static_queue_padding_token_count"),
        "steady_state_prefetch_submit_count": metrics.get("steady_state_prefetch_submit_count"),
        "steady_state_prefetch_reused_draft_count": metrics.get("steady_state_prefetch_reused_draft_count"),
        "steady_state_prefetch_discard_count": metrics.get("steady_state_prefetch_discard_count"),
        "scheduler_method_family": hints.get("method_family"),
        "joint_batch_assignment": hints.get("joint_batch_assignment"),
        "joint_draft_length": hints.get("joint_draft_length"),
        "heterogeneous_worker_assignment": hints.get("heterogeneous_worker_assignment"),
        "edge_device_worker_assignment": hints.get("edge_device_worker_assignment"),
        "single_edge_device_per_request": hints.get("single_edge_device_per_request"),
        "dynamic_drafting": hints.get("dynamic_drafting"),
        "confidence_threshold": hints.get("confidence_threshold"),
        "requested_solver_mode": hints.get("requested_solver_mode"),
        "solver_backend_name": hints.get("solver_backend_name"),
        "paper_solver_complete": hints.get("paper_solver_complete"),
        "solver_backend_fallback_used": hints.get("solver_backend_fallback_used"),
        "solver_backend_fallback_reason": hints.get("solver_backend_fallback_reason"),
        "latency_calibration_profile": hints.get("latency_calibration_profile"),
        "latency_calibration_applied": hints.get("latency_calibration_applied"),
    }


def _positive(value: Any, *, threshold: float = 0.0) -> bool:
    try:
        return float(value) > threshold
    except (TypeError, ValueError):
        return False


def _outputs_by_request(result: RuntimeRunResult) -> dict[str, list[int]]:
    """把 RuntimeRunResult 转成 request_id -> generated token ids。"""
    return {
        request_result.request_id: list(request_result.output_token_ids)
        for request_result in result.request_results
    }


def _prompt_specs(settings: dict[str, Any], model: TransformersCausalLMRunner) -> list[PromptSpec]:
    """根据配置构造本轮 prompt 列表。"""
    if settings["use_sample_prompts"]:
        prompts_file = settings.get("prompts_file")
        if not prompts_file:
            raise ValueError("use_sample_prompts requires data.sample_prompts or --prompts-file.")
        raw_prompts = _expand_prompt_rows(
            _read_prompt_file(Path(str(prompts_file))),
            sample_count=int(settings["sample_count"]),
        )
        if not raw_prompts:
            raise ValueError(f"No prompts loaded from {prompts_file}.")
        return [
            PromptSpec(
                request_id=str(item.get("id") or f"sample-{index:03d}"),
                prompt=str(item["prompt"]),
                prompt_ids=model.encode(str(item["prompt"])),
            )
            for index, item in enumerate(raw_prompts)
        ]
    return [
        PromptSpec(
            request_id="smoke-1",
            prompt=str(settings["prompt"]),
            prompt_ids=model.encode(str(settings["prompt"])),
        )
    ]


def _expand_prompt_rows(rows: list[dict[str, Any]], *, sample_count: int) -> list[dict[str, Any]]:
    """Return exactly sample_count prompt rows, cycling with unique ids if needed."""
    if not rows:
        return []
    requested = max(0, int(sample_count))
    selected: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index in range(requested):
        source = rows[index % len(rows)]
        cycle = index // len(rows)
        base_id = str(source.get("id") or f"sample-{index % len(rows):03d}")
        request_id = base_id if cycle == 0 else f"{base_id}-repeat{cycle}"
        if request_id in seen_ids:
            request_id = f"{request_id}-{index:03d}"
        seen_ids.add(request_id)
        selected.append({"id": request_id, "prompt": str(source["prompt"])})
    return selected


def _read_prompt_file(path: Path) -> list[dict[str, Any]]:
    """读取 JSONL/JSON prompt 文件，返回含 id/prompt 的列表。"""
    if not path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {path}")
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if isinstance(payload, str):
                    payload = {"id": f"sample-{index:03d}", "prompt": payload}
                if not isinstance(payload, dict) or "prompt" not in payload:
                    raise ValueError(f"Prompt JSONL row must be a string or object with prompt: {path}:{index + 1}")
                rows.append(dict(payload))
        return rows
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            rows = []
            for index, item in enumerate(payload):
                if isinstance(item, str):
                    rows.append({"id": f"sample-{index:03d}", "prompt": item})
                elif isinstance(item, dict) and "prompt" in item:
                    rows.append(dict(item))
                else:
                    raise ValueError(f"Prompt JSON item must be a string or object with prompt: {path}:{index}")
            return rows
    raise ValueError(f"Unsupported prompt file format: {path}")


def _tokenizer_eos_ids(tokenizer: object) -> list[int]:
    """从 tokenizer 中提取一个或多个 EOS token id。"""
    eos_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        eos_ids.append(int(eos_token_id))
    additional = getattr(tokenizer, "additional_special_tokens_ids", None) or []
    for token_id in additional:
        if int(token_id) not in eos_ids:
            eos_ids.append(int(token_id))
    return eos_ids


def _sync_output_dir(output_dir: Path, sync_dest: str) -> None:
    """同步整个实验输出目录到用户指定位置。"""
    subprocess.run(["rsync", "-a", str(output_dir) + "/", sync_dest], check=True)


if __name__ == "__main__":
    main()
