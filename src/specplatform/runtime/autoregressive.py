from __future__ import annotations

"""普通 auto-regressive baseline loop。

这个模块实现“不 speculative”的对照生成流程：每次只让一个 model runner
基于当前 prefix 生成 1 个 token，然后立刻写回 GenerationSession。
它不调用 scheduler、verifier 或 acceptance policy。
"""

from dataclasses import dataclass, field
from typing import Any

from specplatform.core import RuntimeContext
from specplatform.model import ModelForwardInput, ModelRunner
from specplatform.runtime.session import GenerationSession


@dataclass(frozen=True)
class AutoRegressiveBaselineResult:
    """baseline loop 的最小返回结果。"""

    request_id: str
    output_token_ids: list[int] = field(default_factory=list)
    step_count: int = 0
    stop_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def run_autoregressive_baseline(
    *,
    session: GenerationSession,
    model_runner: ModelRunner,
    context: RuntimeContext | None = None,
) -> AutoRegressiveBaselineResult:
    """执行普通逐 token 生成，直到 session 结束或触达长度上限。"""
    context = context or RuntimeContext()
    step_count = 0

    while not session.is_finished and len(session.prefix_ids) < session.max_len:
        token_id = _next_token(session, model_runner, context)
        emitted = session.append_tokens([token_id])
        if not emitted:
            break
        step_count += 1

    return AutoRegressiveBaselineResult(
        request_id=session.request_id,
        output_token_ids=list(session.generated_ids),
        step_count=step_count,
        stop_reason=_stop_reason(session),
        metadata={"mode": "autoregressive_baseline"},
    )


def _next_token(
    session: GenerationSession,
    model_runner: ModelRunner,
    context: RuntimeContext,
) -> int:
    """用当前 prefix 的最后一个 token 做一次 forward，并取 argmax token。"""
    output = model_runner.forward(
        ModelForwardInput(
            input_ids=[session.prefix_ids[-1]],
            position_ids=[len(session.prefix_ids) - 1],
            metadata={
                "mode": "autoregressive_baseline",
                "request_id": session.request_id,
                "seed": context.seed,
            },
        )
    )
    return _argmax(output.logits[0])


def _argmax(values: list[float]) -> int:
    """返回 logits 最大值所在的 token id。"""
    return max(range(len(values)), key=lambda index: values[index])


def _stop_reason(session: GenerationSession) -> str:
    """把 session 当前状态解释成 baseline 的停止原因。"""
    if session.generated_ids and session.eos_token_id == session.generated_ids[-1]:
        return "eos"
    if session.remaining_tokens <= 0:
        return "max_new_tokens"
    if len(session.prefix_ids) >= session.max_len:
        return "max_len"
    return "stalled"
