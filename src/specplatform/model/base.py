from __future__ import annotations

"""模型 runner 的最小抽象。

后续真实 Torch/HTTP/Graph backend 都应该适配到这个边界附近；runtime 和
method 不应该直接依赖具体深度学习框架。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
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


@dataclass(frozen=True)
class TopKToken:
    """一次 next-token top-k 查询返回的候选 token。"""

    token_id: int
    logprob: float
    rank: int


@dataclass(frozen=True)
class LinearForwardInput:
    """模型层 linear speculative verification 输入。"""

    prefix_ids: list[int]
    draft_tokens: list[int]
    allow_bonus: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinearForwardOutput:
    """一次 linear verification forward 得到的 target greedy 结果。"""

    draft_target_tokens: list[int]
    bonus_token: int | None = None
    timing_ms: float = 0.0
    start_ns: int | None = None
    end_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TreeForwardNode:
    """模型层 tree_forward 使用的最小候选节点。"""

    node_id: int
    parent_id: int | None
    token_id: int
    depth: int


@dataclass(frozen=True)
class TreeForwardInput:
    """一次 target tree forward 的模型层输入。"""

    prefix_ids: list[int]
    nodes: list[TreeForwardNode]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TreeForwardChoice:
    """tree_forward 对一个 parent prefix 的 greedy target 选择。"""

    parent_node_id: int | None
    target_token_id: int
    prefix_len: int


@dataclass
class TreeForwardOutput:
    """一次 tree_forward 的输出。"""

    choices: list[TreeForwardChoice]
    timing_ms: float = 0.0
    start_ns: int | None = None
    end_ns: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelBackendCapabilities:
    """模型 backend 的平台能力声明。"""

    backend_name: str = "eager"
    backend_fallback: bool = False
    fallback_reason: str | None = None
    supports_topk: bool = True
    supports_batched_topk: bool = False
    supports_batched_next_token: bool = False
    supports_linear_verify_batch: bool = False
    supports_tree_attention: bool = False
    supports_tree_forward_batch: bool = False
    supports_kv_cache: bool = False
    supports_cuda_graph: bool = False

    def to_dict(self) -> dict[str, Any]:
        """转换成 artifact metadata 可序列化的字典。"""
        return {
            "backend_name": self.backend_name,
            "backend_fallback": bool(self.backend_fallback),
            "fallback_reason": self.fallback_reason,
            "supports_topk": bool(self.supports_topk),
            "supports_batched_topk": bool(self.supports_batched_topk),
            "supports_batched_next_token": bool(self.supports_batched_next_token),
            "supports_linear_verify_batch": bool(self.supports_linear_verify_batch),
            "supports_tree_attention": bool(self.supports_tree_attention),
            "supports_tree_forward_batch": bool(self.supports_tree_forward_batch),
            "supports_kv_cache": bool(self.supports_kv_cache),
            "supports_cuda_graph": bool(self.supports_cuda_graph),
        }


class ModelRunner(ABC):
    """模型 runner 抽象基类。"""

    runner_id: str
    max_len: int
    backend_name: str = "eager"
    backend_fallback: bool = False

    def backend_capabilities(self) -> ModelBackendCapabilities:
        """返回当前 backend 的能力声明。"""
        return ModelBackendCapabilities(
            backend_name=getattr(self, "backend_name", "eager"),
            backend_fallback=bool(getattr(self, "backend_fallback", False)),
        )

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

    def next_token_logits_batch(self, prefix_ids_batch: list[list[int]]) -> list[list[float]]:
        """返回一批 prefix 的 next-token logits。

        默认实现逐条调用 `next_token_logits`，保证所有 runner 都有正确语义；
        真正的 target backend 应覆盖这个方法，把 tree parent prefixes 合并成
        一个 batch forward。
        """
        return [self.next_token_logits(prefix_ids) for prefix_ids in prefix_ids_batch]

    def greedy_next_tokens(self, prefix_ids_batch: list[list[int]]) -> list[int]:
        """基于 batch next-token logits 做确定性 greedy 选择。"""
        logits_batch = self.next_token_logits_batch(prefix_ids_batch)
        return [_argmax(logits) for logits in logits_batch]

    def linear_verify(self, request: LinearForwardInput) -> LinearForwardOutput:
        """对一条 linear draft 做模型层验证 forward。

        默认实现按 prefix 逐步调用 next-token 接口，但语义上模拟 single-pass
        verification：后续位置始终条件在 draft token 上，而不是 target token 上。
        高性能 backend 应覆盖为一次 `prefix + draft_tokens` forward。
        """
        if not request.prefix_ids:
            raise ValueError("linear_verify requires a non-empty prefix.")
        working_prefix = list(request.prefix_ids)
        draft_target_tokens: list[int] = []
        matched_all = True
        for draft_token in request.draft_tokens:
            target_token = int(self.greedy_next_token(working_prefix))
            draft_target_tokens.append(target_token)
            if target_token != int(draft_token):
                matched_all = False
                break
            working_prefix.append(int(draft_token))
        bonus_token = (
            int(self.greedy_next_token(working_prefix))
            if request.allow_bonus and matched_all
            else None
        )
        return LinearForwardOutput(
            draft_target_tokens=draft_target_tokens,
            bonus_token=bonus_token,
            metadata={
                "linear_forward_kind": "sequential_next_token_fallback",
                "target_forward_call_count": len(draft_target_tokens) + (1 if bonus_token is not None else 0),
            },
        )

    def linear_verify_batch(self, requests: list[LinearForwardInput]) -> list[LinearForwardOutput]:
        """批量 linear verification 入口。

        默认实现保持语义正确，逐条调用 `linear_verify`。真实 target backend
        应覆盖为跨请求 padded batch single-pass forward。
        """
        outputs: list[LinearForwardOutput] = []
        for index, request in enumerate(requests):
            output = self.linear_verify(request)
            metadata = dict(output.metadata)
            metadata.update(
                {
                    "linear_forward_batch_kind": "fallback_sequential",
                    "batch_index": index,
                    "batch_size": len(requests),
                }
            )
            output.metadata = metadata
            outputs.append(output)
        return outputs

    def tree_forward(self, request: TreeForwardInput) -> TreeForwardOutput:
        """验证一棵 candidate tree 的模型层入口。

        默认实现是语义正确的 fallback：把每个 parent path 转成 prefix，然后复用
        `greedy_next_tokens`。真正的 SpecEdge backend 应覆盖这个方法，用 tree
        attention / KV index / CUDA graph 一次性完成。
        """
        if not request.prefix_ids:
            raise ValueError("tree_forward requires a non-empty prefix.")
        nodes_by_id = _tree_nodes_by_id(request.nodes)
        children_by_parent = _tree_children_by_parent(request.nodes)
        parent_ids = list(children_by_parent)
        parent_prefixes = [
            [*request.prefix_ids, *_tree_path_tokens(parent_id, nodes_by_id)]
            for parent_id in parent_ids
        ]
        token_ids = self.greedy_next_tokens(parent_prefixes)
        return TreeForwardOutput(
            choices=[
                TreeForwardChoice(
                    parent_node_id=parent_id,
                    target_token_id=int(token_id),
                    prefix_len=len(prefix_ids),
                )
                for parent_id, prefix_ids, token_id in zip(parent_ids, parent_prefixes, token_ids)
            ],
            metadata={
                "tree_forward_kind": "batched_next_token_fallback"
                if self.backend_capabilities().supports_batched_next_token
                else "sequential_next_token_fallback",
                "parent_prefix_count": len(parent_prefixes),
            },
        )

    def tree_forward_batch(self, requests: list[TreeForwardInput]) -> list[TreeForwardOutput]:
        """批量 tree_forward 入口。

        默认实现保持语义正确，逐个调用 `tree_forward`，并显式标记 batch
        fallback。后续高性能 backend 可以覆盖为真正的跨请求 packed tree attention。
        """
        outputs: list[TreeForwardOutput] = []
        for index, request in enumerate(requests):
            output = self.tree_forward(request)
            metadata = dict(output.metadata)
            metadata.update(
                {
                    "tree_forward_batch_kind": "fallback_sequential",
                    "batch_index": index,
                    "batch_size": len(requests),
                }
            )
            output.metadata = metadata
            outputs.append(output)
        return outputs

    def next_token_topk(self, prefix_ids: list[int], k: int) -> list[TopKToken]:
        """返回确定性的 next-token top-k 候选。

        默认实现基于 `next_token_logits` 做 CPU 侧 top-k/log-softmax。高性能
        backend 可以覆盖这个方法，直接在设备端返回 top-k。
        """
        if k <= 0:
            return []
        logits = self.next_token_logits(prefix_ids)
        if not logits:
            raise ValueError("CausalLMRunner.next_token_logits returned no logits.")
        max_logit = max(logits)
        logsumexp = max_logit + math.log(sum(math.exp(value - max_logit) for value in logits))
        ranked = sorted(range(len(logits)), key=lambda index: (-logits[index], index))[:k]
        return [
            TopKToken(
                token_id=int(token_id),
                logprob=float(logits[token_id] - logsumexp),
                rank=rank,
            )
            for rank, token_id in enumerate(ranked)
        ]

    def next_token_topk_batch(self, prefix_ids_batch: list[list[int]], k: int) -> list[list[TopKToken]]:
        """返回一批 prefix 的 next-token top-k 候选。

        默认实现逐条调用 `next_token_topk`，高性能 draft backend 可以覆盖成
        frontier batch/graph/KV 路径。
        """
        return [self.next_token_topk(prefix_ids, k) for prefix_ids in prefix_ids_batch]


def _argmax(values: list[float]) -> int:
    """返回最大 logit 所在的 token id。

    如果多个 token 分数相同，Python 的 max 会选择最先出现的 index，
    这个确定性行为便于测试和复现实验。
    """
    return max(range(len(values)), key=lambda index: values[index])


def _tree_nodes_by_id(nodes: list[TreeForwardNode]) -> dict[int, TreeForwardNode]:
    """返回 tree_forward node_id 索引，并做最小拓扑校验。"""
    indexed: dict[int, TreeForwardNode] = {}
    for node in nodes:
        if node.node_id in indexed:
            raise ValueError(f"Duplicate tree_forward node id: {node.node_id}")
        if node.parent_id is not None and node.parent_id not in indexed:
            raise ValueError(f"tree_forward parent must appear before child: {node.node_id}")
        indexed[node.node_id] = node
    return indexed


def _tree_children_by_parent(nodes: list[TreeForwardNode]) -> dict[int | None, list[TreeForwardNode]]:
    """按 parent_id 分组，保留 tree_forward 节点顺序。"""
    _tree_nodes_by_id(nodes)
    grouped: dict[int | None, list[TreeForwardNode]] = {}
    for node in nodes:
        grouped.setdefault(node.parent_id, []).append(node)
    return grouped


def _tree_path_tokens(parent_id: int | None, nodes_by_id: dict[int, TreeForwardNode]) -> list[int]:
    """返回从 root 到 parent_id 的 token path。"""
    if parent_id is None:
        return []
    path: list[int] = []
    current: TreeForwardNode | None = nodes_by_id[parent_id]
    while current is not None:
        path.append(current.token_id)
        current = nodes_by_id.get(current.parent_id) if current.parent_id is not None else None
    path.reverse()
    return path
