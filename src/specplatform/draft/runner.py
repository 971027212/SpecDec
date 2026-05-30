from __future__ import annotations

"""draft runner 的最小真实实现。

draft 层只负责运行 draft model，并把连续生成出的 token 作为原始结果返回。
它不创建 CandidateProposal、不调用 verifier，也不修改 GenerationSession；这些动作分别属于
methods、verification 和 runtime/session 边界。
"""

from dataclasses import dataclass, field
from typing import Any

from specplatform.model import CausalLMRunner


@dataclass(frozen=True)
class DraftGeneration:
    """一次 draft model 生成的原始 token 结果。

    tokens 是 draft runner 真正产出的候选 token 序列。
    timing 预留给后续 Step 11 接 timing/metrics；当前 Step 2 不在算法里使用它。
    metadata 用来携带 request_id、runner_id、原始 prefix 等调试信息，避免污染 core 数据模型。
    """

    tokens: list[int] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GreedyDraftRunner:
    """基于 CausalLMRunner 的线性 greedy draft runner。

    这里的职责非常窄：从当前 prefix 出发，连续调用 draft model 的 greedy_next_token，
    最多生成 max_tokens 个 token。它不知道 scheduler 如何给 budget，也不知道 method 后续
    如何包装 proposal，更不会判断 token 是否被 target 接受。
    """

    model: CausalLMRunner
    runner_id: str | None = None

    def generate_tokens(
        self,
        *,
        prefix_ids: list[int],
        max_tokens: int,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DraftGeneration:
        """从 prefix 之后连续生成 draft tokens。

        prefix_ids 必须非空，因为 causal LM 的下一 token 预测需要上下文。
        max_tokens 小于等于 0 时直接返回空结果，表示 scheduler 给了零预算。
        """

        if not prefix_ids:
            raise ValueError("GreedyDraftRunner.generate_tokens requires a non-empty prefix.")

        base_metadata = dict(metadata or {})
        base_metadata.update(
            {
                "request_id": request_id,
                # runner_id 优先使用 draft runner 自己的标识；没有时回退到模型标识。
                "runner_id": self.runner_id or self.model.runner_id,
                "prefix_ids": list(prefix_ids),
                "max_tokens": max_tokens,
            }
        )

        if max_tokens <= 0:
            return DraftGeneration(tokens=[], metadata=base_metadata)

        generated_tokens: list[int] = []
        # 使用副本推进局部 prefix，保证调用者传入的 GenerationSession.prefix_ids 不被修改。
        working_prefix = list(prefix_ids)

        for _ in range(max_tokens):
            next_token = self.model.greedy_next_token(working_prefix)
            generated_tokens.append(next_token)
            working_prefix.append(next_token)

        return DraftGeneration(tokens=generated_tokens, metadata=base_metadata)
