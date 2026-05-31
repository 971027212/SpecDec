from __future__ import annotations

"""最小 linear speculative decoding 策略。

methods 层只描述“如何把 draft 结果包装成候选”和“如何根据 verifier 事实做接受决策”。
它不拥有 request loop，不直接调用 verifier，也不写 GenerationSession。
"""

from dataclasses import dataclass
from typing import Any

from specplatform.core import AcceptResult, CandidateProposal, DraftBudget, RuntimeContext, VerificationResult
from specplatform.draft import DraftGeneration
from specplatform.methods.base import AcceptancePolicy, CandidateStrategy


@dataclass
class LinearCandidateStrategy(CandidateStrategy):
    """把线性 draft tokens 包装成 CandidateProposal。

    draft runner 只返回 DraftGeneration；真正跨模块传递给 verifier 的契约是 CandidateProposal。
    这层转换放在 methods 中，是为了后续可以替换成 tree、Medusa、EAGLE 等不同候选策略，
    而 runtime 仍然只看到统一的 CandidateStrategy 接口。
    """

    proposal_prefix: str = "linear"

    def propose(
        self,
        session: Any,
        draft_runner: Any,
        budget: DraftBudget,
        context: RuntimeContext,
    ) -> CandidateProposal:
        """生成一个 linear proposal；不调用 verifier，不写 session。"""
        max_tokens = min(int(budget.max_tokens), int(session.remaining_tokens))
        generation: DraftGeneration = draft_runner.generate_tokens(
            prefix_ids=session.prefix_ids,
            max_tokens=max_tokens,
            request_id=session.request_id,
            metadata={
                "draft_budget": {
                    "max_tokens": budget.max_tokens,
                    "max_branches": budget.max_branches,
                    "timeout_ms": budget.timeout_ms,
                }
            },
        )
        proposal_id = f"{self.proposal_prefix}:{session.request_id}:{session.step_idx}"
        metadata = dict(generation.metadata)
        # verifier 需要用 draft 前的 prefix 做逐 token 验证；放在 proposal metadata 中保持 core 字段稳定。
        metadata["prefix_ids"] = list(session.prefix_ids)
        metadata["remaining_tokens"] = session.remaining_tokens
        # 如果 draft 已经填满本轮剩余生成空间，verifier 不应该额外生成 bonus token。
        # 这样避免 target 多跑一次 forward，也避免 bonus 被 session 静默截断。
        metadata["allow_bonus"] = len(generation.tokens) < session.remaining_tokens
        metadata["method"] = "linear"

        return CandidateProposal(
            proposal_id=proposal_id,
            request_id=session.request_id,
            worker_id=metadata.get("runner_id"),
            shape="linear",
            tokens=list(generation.tokens),
            draft_length=len(generation.tokens),
            timing=dict(generation.timing),
            metadata=metadata,
        )


@dataclass
class GreedyPrefixAcceptancePolicy(AcceptancePolicy):
    """最小 greedy prefix acceptance。

    verifier 已经给出 accepted_prefix_len、verified_tokens 和 bonus_token。
    acceptance policy 只消费这些事实：
    - accepted_tokens 来自 proposal.tokens 的前 accepted_prefix_len 个；
    - rejected_tokens 来自 proposal.tokens 的剩余部分；
    - bonus_token 原样来自 verifier，表示 target 在拒绝点或全接受后给出的 token。
    """

    def accept(
        self,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        context: RuntimeContext,
    ) -> AcceptResult:
        """根据 verifier result 产出写回用 token 序列；不调用 verifier，不写 session。"""
        if proposal.proposal_id != verification_result.proposal_id:
            raise ValueError("VerificationResult does not belong to the given proposal.")
        if proposal.request_id != verification_result.request_id:
            raise ValueError("VerificationResult request_id does not match proposal.")

        accepted_prefix_len = int(verification_result.accepted_prefix_len or 0)
        if accepted_prefix_len < 0 or accepted_prefix_len > len(proposal.tokens):
            raise ValueError("accepted_prefix_len is outside proposal token range.")

        accepted_tokens = list(proposal.tokens[:accepted_prefix_len])
        rejected_tokens = list(proposal.tokens[accepted_prefix_len:])
        bonus_token = verification_result.bonus_token
        output_tokens = [*accepted_tokens]
        if bonus_token is not None:
            output_tokens.append(int(bonus_token))

        eos_token_ids = _eos_token_ids(context)
        stop_reason = None
        if output_tokens and output_tokens[-1] in eos_token_ids:
            stop_reason = "eos"
        elif rejected_tokens:
            stop_reason = "rejected"
        elif accepted_tokens or bonus_token is not None:
            stop_reason = "accepted"

        return AcceptResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            accepted_tokens=accepted_tokens,
            rejected_tokens=rejected_tokens,
            bonus_token=bonus_token,
            stop_reason=stop_reason,
            timing=dict(verification_result.timing),
            metadata={
                "accepted_prefix_len": accepted_prefix_len,
                "verified_tokens": list(verification_result.verified_tokens or []),
                "accepted_count": len(accepted_tokens),
                "rejected_count": len(rejected_tokens),
                "has_bonus": bonus_token is not None,
            },
        )


def _eos_token_ids(context: RuntimeContext) -> set[int]:
    """从运行配置中读取 EOS token；没有配置时返回空集合。"""
    raw = (
        context.method_config.get("eos_token_ids")
        or context.run_config.get("eos_token_ids")
        or []
    )
    if isinstance(raw, int):
        return {int(raw)}
    return {int(token_id) for token_id in raw}
