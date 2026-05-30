from __future__ import annotations

"""单个生成请求的会话状态。

GenerationSession 只维护 prompt、已生成 token 和停止条件；它不知道 draft、
verifier、acceptance policy 或 metrics。
"""

from dataclasses import dataclass, field


@dataclass
class GenerationSession:
    """一个 request 在生成过程中的最小状态容器。"""

    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    max_len: int
    eos_token_id: int | None = None
    generated_ids: list[int] = field(default_factory=list)
    step_idx: int = 0

    def __post_init__(self) -> None:
        """校验 prompt 和长度限制足以启动生成。"""
        if not self.prompt_ids:
            raise ValueError("GenerationSession requires at least one prompt token.")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if self.max_len < len(self.prompt_ids) + 1:
            raise ValueError("max_len must leave room for generated tokens.")

    @property
    def prefix_ids(self) -> list[int]:
        """当前模型可见的完整前缀：prompt + 已生成 token。"""
        return [*self.prompt_ids, *self.generated_ids]

    @property
    def remaining_tokens(self) -> int:
        """还能继续生成的新 token 数。"""
        return max(0, self.max_new_tokens - len(self.generated_ids))

    @property
    def is_finished(self) -> bool:
        """判断是否已经达到长度限制或生成 eos。"""
        if self.remaining_tokens <= 0:
            return True
        return bool(self.generated_ids and self.eos_token_id == self.generated_ids[-1])

    def append_tokens(self, token_ids: list[int]) -> list[int]:
        """按顺序写回 token；遇到结束条件会停止并返回实际写入的 token。"""
        accepted: list[int] = []
        for token_id in token_ids:
            if self.is_finished:
                break
            accepted.append(int(token_id))
            self.generated_ids.append(int(token_id))
            if self.eos_token_id is not None and int(token_id) == self.eos_token_id:
                break
        self.step_idx += 1
        return accepted
