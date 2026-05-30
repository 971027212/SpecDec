from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GenerationSession:
    request_id: str
    prompt_ids: list[int]
    max_new_tokens: int
    max_len: int
    eos_token_id: int | None = None
    generated_ids: list[int] = field(default_factory=list)
    step_idx: int = 0

    def __post_init__(self) -> None:
        if not self.prompt_ids:
            raise ValueError("GenerationSession requires at least one prompt token.")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive.")
        if self.max_len < len(self.prompt_ids) + 1:
            raise ValueError("max_len must leave room for generated tokens.")

    @property
    def prefix_ids(self) -> list[int]:
        return [*self.prompt_ids, *self.generated_ids]

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.max_new_tokens - len(self.generated_ids))

    @property
    def is_finished(self) -> bool:
        if self.remaining_tokens <= 0:
            return True
        return bool(self.generated_ids and self.eos_token_id == self.generated_ids[-1])

    def append_tokens(self, token_ids: list[int]) -> list[int]:
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
