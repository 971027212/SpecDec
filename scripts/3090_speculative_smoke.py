from __future__ import annotations

"""3090 侧最小跨机 speculative decoding smoke。

默认加载 /home/chajiahao/data/hf_models/Qwen3-1.7B 作为 draft model，
通过 HTTP 调 A100 的 /verify_linear target verifier。
"""

import argparse
import json
import subprocess
from pathlib import Path
from time import perf_counter_ns

from specplatform.core import DraftBudget, RuntimeContext
from specplatform.draft import GreedyDraftRunner
from specplatform.methods import GreedyPrefixAcceptancePolicy, LinearCandidateStrategy
from specplatform.metrics import (
    write_phase_events_csv,
    write_phase_summary_csv,
    write_request_results_json,
    write_timing_audit,
    write_timing_charts,
)
from specplatform.model import load_causal_lm_runner
from specplatform.runtime import GenerationSession, RuntimeEngine
from specplatform.schedulers import RoundRobinRequestScheduler
from specplatform.timing import TimingRecorder, event_from_span
from specplatform.verification import HttpLinearVerifierClient


def build_parser() -> argparse.ArgumentParser:
    """解析 smoke 参数。"""
    parser = argparse.ArgumentParser(description="Run one minimal 3090 -> A100 speculative decoding smoke.")
    parser.add_argument("--draft-model-path", default="/home/chajiahao/data/hf_models/Qwen3-1.7B")
    parser.add_argument("--target-url", default="http://172.16.11.62:8010")
    parser.add_argument("--prompt", default="介绍一下 speculative decoding")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--draft-tokens", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--draft-backend", default="hf_eager")
    parser.add_argument("--no-backend-fallback", action="store_true")
    parser.add_argument("--run-id", default="3090_a100_smoke")
    parser.add_argument("--output-dir", default="experiments/3090_a100_smoke/latest")
    parser.add_argument("--plot-formats", default="png,svg")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--sync-dest", default=None)
    return parser


def main() -> None:
    """启动单请求 speculative decoding，并打印 token 与文本结果。"""
    args = build_parser().parse_args()
    recorder = TimingRecorder()
    with recorder.span(
        phase="setup.load_draft_model",
        method="linear",
        plan_id=f"{args.run_id}:setup",
        run_id=args.run_id,
        shared=True,
        metadata={
            "draft_model_path": args.draft_model_path,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "draft_backend": args.draft_backend,
            "allow_backend_fallback": not args.no_backend_fallback,
        },
    ) as setup_span:
        draft_model = load_causal_lm_runner(
            args.draft_model_path,
            runner_id="3090-qwen3-1.7b",
            backend=args.draft_backend,
            device=args.device,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
            allow_fallback=not args.no_backend_fallback,
        )
        setup_span.metadata["backend_capabilities"] = draft_model.backend_capabilities().to_dict()
    setup_event = event_from_span(
        setup_span,
        event_id_factory=recorder.next_event_id,
        span_kind="setup",
        attribution="system",
        metadata=dict(setup_span.metadata),
    )
    prompt_ids = draft_model.encode(args.prompt)
    eos_token_ids = _tokenizer_eos_ids(draft_model.tokenizer)
    session = GenerationSession(
        request_id="smoke-1",
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        max_len=draft_model.max_len,
        eos_token_ids=eos_token_ids,
    )
    engine = RuntimeEngine(
        candidate_strategy=LinearCandidateStrategy(),
        acceptance_policy=GreedyPrefixAcceptancePolicy(),
        scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=args.draft_tokens)),
        verifier=HttpLinearVerifierClient(base_url=args.target_url),
        timing_recorder=recorder,
    )
    result = engine.run(
        run_id=args.run_id,
        sessions=[session],
        draft_runners={"draft-worker-0": GreedyDraftRunner(model=draft_model, runner_id="draft-worker-0")},
        context=RuntimeContext(
            run_config={
                "method": "linear",
                "eos_token_ids": eos_token_ids,
            },
            backend_info={
                "target_placement": "a100",
                "target_backend": "http_linear",
                "target_host": args.target_url,
            },
        ),
    )
    result.events.events.insert(0, setup_event)
    output_text = draft_model.decode(session.generated_ids)
    output_dir = Path(args.output_dir)
    _write_smoke_artifacts(
        result=result,
        output_dir=output_dir,
        recorder=recorder,
        render_plots=not args.no_plots,
        plot_formats=tuple(part.strip() for part in args.plot_formats.split(",") if part.strip()),
        metadata={
            "run_id": args.run_id,
            "request_id": session.request_id,
            "prompt": args.prompt,
            "draft_model_path": args.draft_model_path,
            "target_url": args.target_url,
            "max_new_tokens": args.max_new_tokens,
            "draft_tokens": args.draft_tokens,
            "device": args.device,
            "torch_dtype": args.torch_dtype,
            "generated_ids": list(session.generated_ids),
            "generated_len": len(session.generated_ids),
            "is_finished": session.is_finished,
            "text": output_text,
        },
    )
    if args.sync_dest:
        _sync_output_dir(output_dir, args.sync_dest)
    print("request_id:", result.request_results[0].request_id)
    print("generated_ids:", session.generated_ids)
    print("generated_len:", len(session.generated_ids))
    print("is_finished:", session.is_finished)
    print("text:", output_text)
    print("events:", len(result.events.events))
    print("output_dir:", output_dir)
    if args.sync_dest:
        print("synced_to:", args.sync_dest)


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


def _write_smoke_artifacts(
    result: object,
    output_dir: Path,
    *,
    recorder: TimingRecorder,
    render_plots: bool,
    plot_formats: tuple[str, ...],
    metadata: dict[str, object],
) -> None:
    """把 runtime timing/metrics 和 smoke 输出落盘。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_start_ns = perf_counter_ns()
    _write_artifact_files(result, output_dir, metadata)
    artifact_end_ns = perf_counter_ns()
    artifact_span = recorder.record_completed(
        phase="artifact.write",
        method="linear",
        plan_id=f"{metadata['run_id']}:artifact",
        run_id=str(metadata["run_id"]),
        start_ns=artifact_start_ns,
        end_ns=artifact_end_ns,
        shared=True,
        metadata={"output_dir": str(output_dir)},
    )
    result.events.record(
        event_from_span(
            artifact_span,
            event_id_factory=recorder.next_event_id,
            span_kind="detail",
            attribution="system",
            metadata=dict(artifact_span.metadata),
        )
    )
    if render_plots:
        plot_start_ns = perf_counter_ns()
        write_timing_charts(list(result.events.events), output_dir / "plots", formats=plot_formats)
        plot_end_ns = perf_counter_ns()
        plot_span = recorder.record_completed(
            phase="plot.render",
            method="linear",
            plan_id=f"{metadata['run_id']}:plots",
            run_id=str(metadata["run_id"]),
            start_ns=plot_start_ns,
            end_ns=plot_end_ns,
            shared=True,
            metadata={
                "output_dir": str(output_dir / "plots"),
                "formats": list(plot_formats),
            },
        )
        result.events.record(
            event_from_span(
                plot_span,
                event_id_factory=recorder.next_event_id,
                span_kind="detail",
                attribution="system",
                metadata=dict(plot_span.metadata),
            )
        )
        write_timing_audit(list(result.events.events), output_dir / "plots")
    _write_artifact_files(result, output_dir, metadata)


def _write_artifact_files(result: object, output_dir: Path, metadata: dict[str, object]) -> None:
    """写 runtime events 和 smoke 输出文件。"""
    events = list(result.events.events)
    result.events.write_jsonl(output_dir / "phase_events.jsonl")
    write_phase_events_csv(events, output_dir / "phase_events.csv")
    write_phase_summary_csv(events, output_dir / "phase_summary.csv")
    write_request_results_json(result.request_results, output_dir / "request_results.json")
    (output_dir / "smoke_output.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sync_output_dir(output_dir: Path, sync_dest: str) -> None:
    """同步整个实验输出目录到用户指定位置。"""
    subprocess.run(["rsync", "-a", str(output_dir) + "/", sync_dest], check=True)


if __name__ == "__main__":
    main()
