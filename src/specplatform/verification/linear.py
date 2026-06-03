from __future__ import annotations

"""本地 linear target verifier。

verification 层只验证 proposal，并返回 VerificationResult。它不决定最终接受哪些 token，
也不修改 GenerationSession；这些动作留给 methods.acceptance 和 runtime。
"""

from dataclasses import dataclass, field
from time import perf_counter_ns
from typing import Any

from specplatform.core import CandidateProposal, RuntimeContext, VerificationResult
from specplatform.model import CausalLMRunner, LinearForwardInput, LinearForwardOutput
from specplatform.verification.base import VerifierBackend
from specplatform.verification.schema import LinearVerifyRequest, LinearVerifyResponse


@dataclass
class LinearVerifier(VerifierBackend):
    """用 target CausalLMRunner 对 linear draft tokens 做逐 token greedy 验证。"""

    model: CausalLMRunner
    backend_name: str = "linear_local"
    metadata: dict[str, Any] = field(default_factory=dict)

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        """验证单个 proposal；只返回 verifier 看到的事实。"""
        if proposal.shape != "linear":
            raise ValueError("LinearVerifier only supports linear proposals.")
        request = LinearVerifyRequest(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            prefix_ids=_proposal_prefix_ids(proposal),
            draft_tokens=list(proposal.tokens),
            eos_token_ids=_eos_token_ids(proposal, context),
            allow_bonus=bool(proposal.metadata.get("allow_bonus", True)),
            metadata=dict(proposal.metadata),
        )
        response = self.verify_request(request, context)
        _validate_response_for_request(response, request)
        return VerificationResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            shape=proposal.shape,
            accepted_prefix_len=response.accepted_prefix_len,
            verified_tokens=list(response.verified_tokens),
            bonus_token=response.bonus_token,
            timing=dict(response.timing),
            payload=response.to_dict(),
            metadata={
                "backend_name": self.backend_name,
                **dict(self.metadata),
                **dict(response.metadata),
                "timing": dict(response.timing),
            },
        )

    def verify_request(
        self,
        request: LinearVerifyRequest,
        context: RuntimeContext | None = None,
    ) -> LinearVerifyResponse:
        """验证 HTTP/schema 层请求，供本地测试和 A100 service 共用。"""
        del context
        responses = self.verify_requests_batch([request])
        return responses[0]

    def verify_requests_batch(
        self,
        requests: list[LinearVerifyRequest],
        context: RuntimeContext | None = None,
        *,
        batch_id: str | None = None,
    ) -> list[LinearVerifyResponse]:
        """Verify multiple linear requests with single-pass target forwards.

        DiP-SD/SLED rely on the server validating a multi-user/candidate batch
        together.  This method sends each item as `prefix + draft_tokens` to the
        model boundary, so a backend can validate every draft position and the
        optional bonus token with one target forward per request batch.
        """
        del context
        if not requests:
            return []
        for request in requests:
            if not request.prefix_ids:
                raise ValueError("Linear verification requires a non-empty prefix.")
        outputs, forward_events = _timed_linear_verify_batch(
            self.model,
            requests,
            batch_id=batch_id,
        )
        return [
            _response_from_linear_forward(request, output, events)
            for request, output, events in zip(requests, outputs, forward_events)
        ]


def _proposal_prefix_ids(proposal: CandidateProposal) -> list[int]:
    """从 proposal metadata 取出 draft 前的 prefix。"""
    prefix_ids = proposal.metadata.get("prefix_ids")
    if prefix_ids is None:
        raise ValueError("Linear proposal metadata must include prefix_ids.")
    return [int(token_id) for token_id in prefix_ids]


def _eos_token_ids(proposal: CandidateProposal, context: RuntimeContext | None) -> list[int]:
    """按 proposal metadata -> context.method_config -> context.run_config 的优先级读取 EOS。"""
    raw = proposal.metadata.get("eos_token_ids")
    if raw is None and context is not None:
        raw = context.method_config.get("eos_token_ids") or context.run_config.get("eos_token_ids")
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    return [int(token_id) for token_id in raw]


def _validate_response_for_request(
    response: LinearVerifyResponse,
    request: LinearVerifyRequest,
) -> None:
    """校验 verifier 响应仍然对应当前请求，避免 HTTP/schema 漂移被吞掉。"""
    if response.request_id != request.request_id:
        raise ValueError("LinearVerifyResponse request_id does not match request.")
    if response.proposal_id != request.proposal_id:
        raise ValueError("LinearVerifyResponse proposal_id does not match request.")
    if response.accepted_prefix_len < 0:
        raise ValueError("accepted_prefix_len must be non-negative.")
    if response.accepted_prefix_len > len(request.draft_tokens):
        raise ValueError("accepted_prefix_len exceeds draft token length.")
    if len(response.verified_tokens) < response.accepted_prefix_len:
        raise ValueError("verified_tokens shorter than accepted prefix.")
    if (
        not request.allow_bonus
        and response.bonus_token is not None
        and response.accepted_prefix_len == len(request.draft_tokens)
    ):
        raise ValueError("Verifier returned bonus_token when allow_bonus is false.")


def _response_from_linear_forward(
    request: LinearVerifyRequest,
    output: LinearForwardOutput,
    target_forward_events: list[dict[str, Any]],
) -> LinearVerifyResponse:
    """把模型层 single-pass 结果转换成 schema 层响应。"""
    draft_targets = [int(token_id) for token_id in output.draft_target_tokens]

    accepted_prefix_len = 0
    verified_tokens: list[int] = []
    eos_token_ids = set(request.eos_token_ids)
    output_metadata = dict(output.metadata)
    linear_kind = output_metadata.get("linear_forward_batch_kind") or output_metadata.get("linear_forward_kind")
    is_single_pass = str(linear_kind).startswith(("linear_single_pass", "linear_tree_attention"))
    causal_safe_prefix_batch = bool(output_metadata.get("causal_safe_prefix_batch"))
    common_metadata = _linear_response_metadata(output_metadata, linear_kind, is_single_pass, causal_safe_prefix_batch)

    for index, draft_token in enumerate(request.draft_tokens):
        if index >= len(draft_targets):
            raise ValueError("linear_verify output ended before a mismatch or full acceptance.")
        target_token = int(draft_targets[index])
        verified_tokens.append(target_token)
        if target_token != int(draft_token):
            metadata = {**common_metadata, "mismatch_at": int(accepted_prefix_len)}
            return LinearVerifyResponse(
                request_id=request.request_id,
                proposal_id=request.proposal_id,
                accepted_prefix_len=accepted_prefix_len,
                verified_tokens=verified_tokens,
                bonus_token=target_token,
                metadata=metadata,
                timing=_target_timing(target_forward_events),
            )

        accepted_prefix_len += 1
        if target_token in eos_token_ids:
            metadata = {**common_metadata, "matched_eos": target_token}
            return LinearVerifyResponse(
                request_id=request.request_id,
                proposal_id=request.proposal_id,
                accepted_prefix_len=accepted_prefix_len,
                verified_tokens=verified_tokens,
                bonus_token=None,
                metadata=metadata,
                timing=_target_timing(target_forward_events),
            )

    if not request.allow_bonus:
        return LinearVerifyResponse(
            request_id=request.request_id,
            proposal_id=request.proposal_id,
            accepted_prefix_len=accepted_prefix_len,
            verified_tokens=verified_tokens,
            bonus_token=None,
            metadata={
                **common_metadata,
                "bonus_skipped": "not_allowed",
            },
            timing=_target_timing(target_forward_events),
        )

    if output.bonus_token is None:
        raise ValueError("linear_verify output did not include bonus_token.")
    bonus_token = int(output.bonus_token)
    metadata: dict[str, Any] = dict(common_metadata)
    if bonus_token in eos_token_ids:
        metadata["bonus_is_eos"] = bonus_token
    return LinearVerifyResponse(
        request_id=request.request_id,
        proposal_id=request.proposal_id,
        accepted_prefix_len=accepted_prefix_len,
        verified_tokens=verified_tokens,
        bonus_token=bonus_token,
        metadata=metadata,
        timing=_target_timing(target_forward_events),
    )


def _linear_response_metadata(
    output_metadata: dict[str, Any],
    linear_kind: Any,
    is_single_pass: bool,
    causal_safe_prefix_batch: bool,
) -> dict[str, Any]:
    metadata = {
        "linear_forward_kind": output_metadata.get("linear_forward_kind", linear_kind),
        "linear_forward_batch_kind": linear_kind,
        "single_pass_linear_verify": is_single_pass,
        "causal_safe_prefix_batch": causal_safe_prefix_batch,
    }
    for key in (
        "explicit_kv_cache",
        "cuda_graph",
        "max_graph_tokens",
        "graph_verify_token_count",
        "graph_output_token_count",
        "graph_prefill_token_count",
        "target_forward_call_count",
    ):
        if key in output_metadata:
            metadata[key] = output_metadata[key]
    return metadata


def _timed_linear_verify_batch(
    model: CausalLMRunner,
    requests: list[LinearVerifyRequest],
    *,
    batch_id: str | None,
) -> tuple[list[LinearForwardOutput], list[list[dict[str, Any]]]]:
    """运行一次模型层 linear_verify_batch，并生成每个 item 的 timing 事件。"""
    forward_inputs = [
        LinearForwardInput(
            prefix_ids=list(request.prefix_ids),
            draft_tokens=list(request.draft_tokens),
            allow_bonus=bool(request.allow_bonus),
            metadata=dict(request.metadata),
        )
        for request in requests
    ]
    start_ns = perf_counter_ns()
    outputs = model.linear_verify_batch(forward_inputs)
    end_ns = perf_counter_ns()
    if len(outputs) != len(requests):
        raise ValueError("linear_verify_batch returned a different number of outputs.")

    shared_batch_event_id = f"linear_verify_batch:{start_ns}:{len(requests)}"
    active_event_count = sum(1 for output in outputs if _linear_output_call_count(output) > 0)
    duration_ms = _duration_ms(start_ns, end_ns)
    attributed_duration_ms = duration_ms / max(1, active_event_count)
    events_by_index: list[list[dict[str, Any]]] = []
    for index, (request, output) in enumerate(zip(requests, outputs)):
        call_count = _linear_output_call_count(output)
        if call_count <= 0:
            events_by_index.append([])
            continue
        metadata = dict(output.metadata)
        batch_kind = str(
            metadata.get("linear_forward_batch_kind")
            or metadata.get("linear_forward_kind")
            or _linear_batch_kind(model)
        )
        event: dict[str, Any] = {
            "index": 0,
            "kind": "linear_verify",
            "prefix_len": len(request.prefix_ids),
            "draft_token_count": len(request.draft_tokens),
            "verified_token_count": len(output.draft_target_tokens),
            "token_ids": [int(token_id) for token_id in output.draft_target_tokens],
            "bonus_token": output.bonus_token,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ms": attributed_duration_ms,
            "shared_duration_ms": duration_ms,
            "batch_size": len(requests),
            "batch_index": int(index),
            "batch_id": batch_id,
            "linear_forward_batch_kind": batch_kind,
            "target_forward_call_count": int(call_count),
            "metadata": metadata,
        }
        metadata_shared_forward_id = metadata.get("shared_forward_id")
        if metadata_shared_forward_id:
            event["shared_batch_event_id"] = str(metadata_shared_forward_id)
        elif batch_kind == "linear_single_pass_batch":
            event["shared_batch_event_id"] = shared_batch_event_id
        else:
            event["shared_batch_event_id"] = f"linear_verify_item:{start_ns}:{index}"
        events_by_index.append([event])
    return outputs, events_by_index


def _linear_output_call_count(output: LinearForwardOutput) -> int:
    metadata = dict(output.metadata)
    if metadata.get("target_forward_call_count") is not None:
        return max(0, int(metadata["target_forward_call_count"]))
    return 1 if output.draft_target_tokens or output.bonus_token is not None else 0


def _linear_batch_kind(model: CausalLMRunner) -> str:
    capabilities = model.backend_capabilities()
    return "linear_single_pass_batch" if capabilities.supports_linear_verify_batch else "fallback_sequential"


def _target_timing(target_forward_events: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 target forward 明细，供 A100 response timing 回传。"""
    batch_kinds = {
        str(event.get("linear_forward_batch_kind"))
        for event in target_forward_events
        if event.get("linear_forward_batch_kind")
    }
    target_forward_call_count = sum(
        max(0, int(event.get("target_forward_call_count") or 1))
        for event in target_forward_events
    )
    return {
        "target_forward_events": [dict(event) for event in target_forward_events],
        "target_forward_total_ms": sum(float(event["duration_ms"]) for event in target_forward_events),
        "target_forward_call_count": target_forward_call_count,
        "linear_forward_batch_kinds": sorted(batch_kinds),
    }


def _duration_ms(start_ns: int, end_ns: int) -> float:
    """把 perf_counter_ns 差值转换为毫秒。"""
    return (int(end_ns) - int(start_ns)) / 1_000_000
