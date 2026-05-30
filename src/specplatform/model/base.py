from __future__ import annotations

"""模型 runner 的最小抽象。

后续真实 Torch/HTTP/Graph backend 都应该适配到这个边界附近；runtime 和
method 不应该直接依赖具体深度学习框架。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelForwardInput:
    """一次模型 forward 的输入。"""

    input_ids: list[int]
    position_ids: list[int] | None = None
    cache_indices: list[int] | None = None
    attention_mask: list[list[int]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelForwardOutput:
    """一次模型 forward 的输出。"""

    logits: list[list[float]]
    timing_ms: float = 0.0
    start_ns: int | None = None
    end_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ModelRunner(ABC):
    """模型 runner 抽象基类。"""

    runner_id: str
    max_len: int

    def prefill(self, input_ids: list[int]) -> None:
        """可选 prefill 钩子；真实 runner 可在这里初始化 KV cache。"""
        return None

    @abstractmethod
    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """对输入 token 执行一次 forward，返回 logits。"""
        raise NotImplementedError

    def gather_kv(self, src_indices: list[int], dest_indices: list[int]) -> None:
        """可选 KV cache 重排钩子；真实 backend 后续再实现。"""
        return None

    def reset(self, request_id: str | None = None) -> None:
        """可选状态清理钩子；真实 runner 可按 request 清理内部状态。"""
        return None


class CausalLMRunner(ModelRunner):
    """真实因果语言模型 runner 的最小接口。

    这个接口是后续 draft model 和 target model 的共同边界：
    - draft runner 会用它基于当前 prefix 连续 greedy 生成候选 token；
    - target verifier 会用它基于 prefix 逐 token 验证 draft proposal；
    - runtime/method 不直接依赖 Transformers、Torch 或具体模型路径。

    注意：这里仍然不写完整 generation loop。完整 loop 属于 draft/runtime 层。
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """把输入文本编码成 token ids。

        真实实现可以委托给 Hugging Face tokenizer。接口返回普通 list[int]，
        是为了让 core/runtime/draft/verification 不依赖 tokenizer 类型。
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """把 token ids 解码成文本。

        这个方法主要用于 smoke test 和输出检查；算法内部仍以 token ids 为准。
        """
        raise NotImplementedError

    @abstractmethod
    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        """返回给定 prefix 下“下一 token”的 logits。

        对真实 causal LM 来说，实现通常会把完整 prefix 喂给模型，然后取最后
        一个位置的 logits。这里不做 greedy 选择，让 verifier 可以直接比较
        target 模型预测和 draft token。
        """
        raise NotImplementedError

    def greedy_next_token(self, prefix_ids: list[int]) -> int:
        """基于 next_token_logits 做最基础的 greedy 选择。

        这是最小闭环的采样策略：不做 temperature、top-p、top-k，也不做随机性。
        后续如果要扩展采样策略，应该在更高层注入策略，而不是改这个最小接口。
        """
        if not prefix_ids:
            raise ValueError("CausalLMRunner.greedy_next_token requires a non-empty prefix.")
        logits = self.next_token_logits(prefix_ids)
        if not logits:
            raise ValueError("CausalLMRunner.next_token_logits returned no logits.")
        return _argmax(logits)


def _argmax(values: list[float]) -> int:
    """返回最大 logit 所在的 token id。

    如果多个 token 分数相同，Python 的 max 会选择最先出现的 index，
    这个确定性行为便于测试和复现实验。
    """
    return max(range(len(values)), key=lambda index: values[index])
