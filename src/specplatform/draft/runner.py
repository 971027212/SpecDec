from __future__ import annotations

"""draft runner 的最小真实实现。

draft 层只负责运行 draft model，并把连续生成出的 token 作为原始结果返回。
它不创建 CandidateProposal、不调用 verifier，也不修改 GenerationSession；这些动作分别属于
methods、verification 和 runtime/session 边界。
"""

from dataclasses import dataclass, field
import math
from time import perf_counter_ns
from typing import Any

from specplatform.core import CandidateNode, CandidateTree
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


@dataclass(frozen=True)
class TreeDraftGeneration:
    """一次 tree draft model 生成的原始候选树结果。"""

    tree: CandidateTree
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
    metadata: dict[str, Any] = field(default_factory=dict)

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

        base_metadata = {**dict(self.metadata), **dict(metadata or {})}
        base_metadata.update(
            {
                "request_id": request_id,
                # runner_id 优先使用 draft runner 自己的标识；没有时回退到模型标识。
                "runner_id": self.runner_id or self.model.runner_id,
                "device": getattr(self.model, "device", None),
                "backend_capabilities": self.model.backend_capabilities().to_dict(),
                "prefix_ids": list(prefix_ids),
                "max_tokens": max_tokens,
            }
        )

        if max_tokens <= 0:
            return DraftGeneration(tokens=[], metadata=base_metadata)

        generated_tokens: list[int] = []
        token_forward_events: list[dict[str, Any]] = []
        # 使用副本推进局部 prefix，保证调用者传入的 GenerationSession.prefix_ids 不被修改。
        working_prefix = list(prefix_ids)

        for index in range(max_tokens):
            start_ns = perf_counter_ns()
            next_token = self.model.greedy_next_token(working_prefix)
            end_ns = perf_counter_ns()
            generated_tokens.append(next_token)
            token_forward_events.append(
                {
                    "index": index,
                    "prefix_len": len(working_prefix),
                    "token_id": int(next_token),
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ms": (end_ns - start_ns) / 1_000_000,
                }
            )
            working_prefix.append(next_token)

        base_metadata["draft_token_forward_events"] = token_forward_events
        return DraftGeneration(tokens=generated_tokens, metadata=base_metadata)

    def generate_tokens_until_confidence_drop(
        self,
        *,
        prefix_ids: list[int],
        max_tokens: int,
        confidence_threshold: float,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DraftGeneration:
        """Generate SLED-style draft tokens until the greedy confidence drops.

        SLED's edge-side dynamic drafting decides when to request server
        verification from the draft model's token confidence.  The token that
        triggers the low-confidence stop is included in the verification
        request, matching the accumulated draft-token queue semantics.
        """

        if not prefix_ids:
            raise ValueError("GreedyDraftRunner.generate_tokens_until_confidence_drop requires a non-empty prefix.")

        base_metadata = {**dict(self.metadata), **dict(metadata or {})}
        base_metadata.update(
            {
                "request_id": request_id,
                "runner_id": self.runner_id or self.model.runner_id,
                "device": getattr(self.model, "device", None),
                "backend_capabilities": self.model.backend_capabilities().to_dict(),
                "prefix_ids": list(prefix_ids),
                "max_tokens": max_tokens,
                "dynamic_drafting": True,
                "confidence_threshold": float(confidence_threshold),
            }
        )

        if max_tokens <= 0:
            base_metadata["dynamic_stop_reason"] = "zero_budget"
            base_metadata["draft_confidences"] = []
            return DraftGeneration(tokens=[], metadata=base_metadata)

        generated_tokens: list[int] = []
        token_forward_events: list[dict[str, Any]] = []
        confidences: list[float] = []
        working_prefix = list(prefix_ids)
        stop_reason = "max_tokens"

        for index in range(max_tokens):
            start_ns = perf_counter_ns()
            topk = self.model.next_token_topk(working_prefix, 1)
            end_ns = perf_counter_ns()
            if not topk:
                raise ValueError("CausalLMRunner.next_token_topk returned no candidates.")
            candidate = topk[0]
            confidence = math.exp(float(candidate.logprob))
            next_token = int(candidate.token_id)
            generated_tokens.append(next_token)
            confidences.append(confidence)
            token_forward_events.append(
                {
                    "index": index,
                    "prefix_len": len(working_prefix),
                    "token_id": next_token,
                    "draft_logprob": float(candidate.logprob),
                    "draft_confidence": confidence,
                    "confidence_threshold": float(confidence_threshold),
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_ms": (end_ns - start_ns) / 1_000_000,
                    "dynamic_stop_triggered": confidence < float(confidence_threshold),
                }
            )
            working_prefix.append(next_token)
            if confidence < float(confidence_threshold):
                stop_reason = "confidence_below_threshold"
                break

        base_metadata["draft_token_forward_events"] = token_forward_events
        base_metadata["draft_confidences"] = confidences
        base_metadata["dynamic_stop_reason"] = stop_reason
        return DraftGeneration(tokens=generated_tokens, metadata=base_metadata)


@dataclass
class TopKTreeDraftRunner:
    """基于 next-token top-k 的确定性 tree draft runner。

    这是 SpecEdge tree 方法的可测试 eager 路径：按深度扩展 frontier，
    同一 depth 的父节点通过模型层 `next_token_topk_batch` 批量取 top-k 子节点，
    按累计 logprob 保存在 CandidateTree 中。后续 graph/cached backend 可以覆盖
    这个模型边界，runner 语义不变。
    """

    model: CausalLMRunner
    runner_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def generate_tree(
        self,
        *,
        prefix_ids: list[int],
        max_depth: int,
        max_branches: int,
        max_nodes: int,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TreeDraftGeneration:
        """从 prefix 之后扩展确定性 top-k 候选树。"""
        if not prefix_ids:
            raise ValueError("TopKTreeDraftRunner.generate_tree requires a non-empty prefix.")

        base_metadata = {**dict(self.metadata), **dict(metadata or {})}
        runner_id = str(base_metadata.get("runner_id") or self.runner_id or self.model.runner_id)
        base_metadata.update(
            {
                "request_id": request_id,
                "runner_id": runner_id,
                "device": getattr(self.model, "device", None),
                "backend_capabilities": self.model.backend_capabilities().to_dict(),
                "prefix_ids": list(prefix_ids),
                "max_depth": int(max_depth),
                "max_branches": int(max_branches),
                "max_nodes": int(max_nodes),
            }
        )

        if max_depth <= 0 or max_branches <= 0 or max_nodes <= 0:
            tree = CandidateTree(root_prefix_len=len(prefix_ids), nodes=[])
            return TreeDraftGeneration(tree=tree, metadata=base_metadata)

        graph_tree_generator = getattr(self.model, "generate_tree_topk_graph", None)
        if callable(graph_tree_generator) and not bool(base_metadata.get("disable_graph_tree_draft", False)):
            graph_result = graph_tree_generator(
                prefix_ids=list(prefix_ids),
                max_depth=int(max_depth),
                max_branches=int(max_branches),
                max_nodes=int(max_nodes),
                request_id=request_id,
                runner_id=str(runner_id),
                metadata=dict(base_metadata),
            )
            tree = graph_result["tree"]
            tree.validate()
            graph_metadata = {**base_metadata, **dict(graph_result.get("metadata") or {})}
            graph_metadata["tree_snapshot"] = tree.to_dict()
            return TreeDraftGeneration(
                tree=tree,
                metadata=graph_metadata,
            )

        nodes: list[CandidateNode] = []
        detail_events: list[dict[str, Any]] = []
        next_node_id = 0
        # frontier entries: parent node id, path tokens from root, cumulative logprob.
        frontier: list[tuple[int | None, list[int], float]] = [(None, [], 0.0)]

        for depth in range(1, max_depth + 1):
            next_frontier: list[tuple[int | None, list[int], float]] = []
            local_prefixes = [[*prefix_ids, *path_tokens] for _parent_id, path_tokens, _score in frontier]
            start_ns = perf_counter_ns()
            topk_batch = self.model.next_token_topk_batch(local_prefixes, max_branches)
            end_ns = perf_counter_ns()
            shared_batch_event_id = f"draft_topk_batch:{start_ns}:{depth}:{len(local_prefixes)}"
            if len(topk_batch) != len(frontier):
                raise ValueError("CausalLMRunner.next_token_topk_batch returned a different batch size.")
            duration_ms = (end_ns - start_ns) / 1_000_000
            attributed_duration_ms = duration_ms / max(1, len(frontier))
            for batch_index, ((parent_id, path_tokens, cumulative_logprob), local_prefix, topk_tokens) in enumerate(
                zip(frontier, local_prefixes, topk_batch)
            ):
                if len(nodes) >= max_nodes:
                    break
                detail_events.append(
                    {
                        "phase": "draft.topk",
                        "start_ns": start_ns,
                        "end_ns": end_ns,
                        "duration_ms": attributed_duration_ms,
                        "shared_duration_ms": duration_ms,
                        "batch_size": len(frontier),
                        "batch_index": batch_index,
                        "shared_batch_event_id": shared_batch_event_id,
                        "topk_batch_kind": "batched_topk"
                        if self.model.backend_capabilities().supports_batched_topk
                        else "fallback_or_cached_topk",
                        "depth": depth,
                        "parent_node_id": parent_id,
                        "prefix_len": len(local_prefix),
                        "topk": [
                            {
                                "token_id": candidate.token_id,
                                "rank": candidate.rank,
                                "logprob": candidate.logprob,
                            }
                            for candidate in topk_tokens
                        ],
                    }
                )
                for candidate in topk_tokens:
                    if len(nodes) >= max_nodes:
                        break
                    node_logprob = cumulative_logprob + float(candidate.logprob)
                    node = CandidateNode(
                        node_id=next_node_id,
                        parent_id=parent_id,
                        token_id=int(candidate.token_id),
                        depth=depth,
                        draft_logprob=node_logprob,
                        draft_worker_id=str(runner_id),
                    )
                    nodes.append(node)
                    next_frontier.append((node.node_id, [*path_tokens, node.token_id], node_logprob))
                    next_node_id += 1
            frontier = next_frontier
            if not frontier or len(nodes) >= max_nodes:
                break

        tree = CandidateTree(root_prefix_len=len(prefix_ids), nodes=nodes)
        tree.validate()
        base_metadata["draft_token_forward_events"] = detail_events
        base_metadata["tree_snapshot"] = tree.to_dict()
        return TreeDraftGeneration(tree=tree, metadata=base_metadata)

    def generate_tree_batch(self, requests: list[dict[str, Any]]) -> list[TreeDraftGeneration]:
        """Generate a batch of top-k trees, using a graph batch backend when available."""
        if not requests:
            return []

        graph_batch_generator = getattr(self.model, "generate_tree_topk_batch_graph", None)
        if callable(graph_batch_generator) and not any(
            bool(dict(request.get("metadata") or {}).get("disable_graph_tree_draft", False))
            for request in requests
        ):
            graph_requests: list[dict[str, Any]] = []
            for index, request in enumerate(requests):
                prefix_ids = [int(token_id) for token_id in request["prefix_ids"]]
                if not prefix_ids:
                    raise ValueError("TopKTreeDraftRunner.generate_tree_batch requires non-empty prefixes.")
                runner_id = str(request.get("runner_id") or self.runner_id or self.model.runner_id)
                metadata = {**dict(self.metadata), **dict(request.get("metadata") or {})}
                metadata.update(
                    {
                        "request_id": request.get("request_id"),
                        "runner_id": runner_id,
                        "device": getattr(self.model, "device", None),
                        "backend_capabilities": self.model.backend_capabilities().to_dict(),
                        "prefix_ids": list(prefix_ids),
                        "max_depth": int(request.get("max_depth", 0)),
                        "max_branches": int(request.get("max_branches", 0)),
                        "max_nodes": int(request.get("max_nodes", 0)),
                        "batch_index": index,
                        "batch_size": len(requests),
                    }
                )
                if request.get("draft_batch_index") is not None:
                    metadata["official_draft_batch_index"] = int(request["draft_batch_index"])
                graph_requests.append(
                    {
                        "prefix_ids": prefix_ids,
                        "max_depth": int(request.get("max_depth", 0)),
                        "max_branches": int(request.get("max_branches", 0)),
                        "max_nodes": int(request.get("max_nodes", 0)),
                        "draft_batch_index": request.get("draft_batch_index"),
                        "request_id": request.get("request_id"),
                        "runner_id": runner_id,
                        "metadata": metadata,
                    }
                )
            graph_results = graph_batch_generator(graph_requests)
            if len(graph_results) != len(graph_requests):
                raise ValueError("CausalLMRunner.generate_tree_topk_batch_graph returned a different batch size.")
            generations: list[TreeDraftGeneration] = []
            for graph_request, graph_result in zip(graph_requests, graph_results):
                tree = graph_result["tree"]
                tree.validate()
                graph_metadata = {**dict(graph_request["metadata"]), **dict(graph_result.get("metadata") or {})}
                graph_metadata["tree_snapshot"] = tree.to_dict()
                generations.append(TreeDraftGeneration(tree=tree, metadata=graph_metadata))
            return generations

        return [
            self.generate_tree(
                prefix_ids=[int(token_id) for token_id in request["prefix_ids"]],
                max_depth=int(request.get("max_depth", 0)),
                max_branches=int(request.get("max_branches", 0)),
                max_nodes=int(request.get("max_nodes", 0)),
                request_id=request.get("request_id"),
                metadata=dict(request.get("metadata") or {}),
            )
            for request in requests
        ]

    def grow_official_tree_batch(self, requests: list[dict[str, Any]]) -> list[TreeDraftGeneration]:
        """Grow existing official SpecEdge state trees through a model backend."""
        if not requests:
            return []

        graph_grower = getattr(self.model, "grow_official_tree_batch_graph", None)
        if callable(graph_grower) and not any(
            bool(dict(request.get("metadata") or {}).get("disable_graph_tree_draft", False))
            for request in requests
        ):
            graph_requests: list[dict[str, Any]] = []
            for index, request in enumerate(requests):
                prefix_ids = [int(token_id) for token_id in request["prefix_ids"]]
                if not prefix_ids:
                    raise ValueError("TopKTreeDraftRunner.grow_official_tree_batch requires non-empty prefixes.")
                runner_id = str(request.get("runner_id") or self.runner_id or self.model.runner_id)
                metadata = {**dict(self.metadata), **dict(request.get("metadata") or {})}
                metadata.update(
                    {
                        "request_id": request.get("request_id"),
                        "runner_id": runner_id,
                        "device": getattr(self.model, "device", None),
                        "backend_capabilities": self.model.backend_capabilities().to_dict(),
                        "prefix_ids": list(prefix_ids),
                        "max_depth": int(request.get("max_depth", 0)),
                        "max_branches": int(request.get("max_branches", 0)),
                        "max_nodes": int(request.get("max_nodes", 0)),
                        "batch_index": index,
                        "batch_size": len(requests),
                        "official_state_grow": True,
                    }
                )
                graph_requests.append(
                    {
                        "prefix_ids": prefix_ids,
                        "tree": request.get("tree"),
                        "tree_node_statuses": dict(request.get("tree_node_statuses") or {}),
                        "needs_prefix_tail_forward": bool(request.get("needs_prefix_tail_forward", False)),
                        "draft_batch_index": request.get("draft_batch_index"),
                        "max_depth": int(request.get("max_depth", 0)),
                        "max_branches": int(request.get("max_branches", 0)),
                        "max_nodes": int(request.get("max_nodes", 0)),
                        "request_id": request.get("request_id"),
                        "runner_id": runner_id,
                        "metadata": metadata,
                    }
                )
            graph_results = graph_grower(graph_requests)
            if len(graph_results) != len(graph_requests):
                raise ValueError("CausalLMRunner.grow_official_tree_batch_graph returned a different batch size.")
            generations: list[TreeDraftGeneration] = []
            for graph_request, graph_result in zip(graph_requests, graph_results):
                tree = graph_result["tree"]
                tree.validate()
                graph_metadata = {**dict(graph_request["metadata"]), **dict(graph_result.get("metadata") or {})}
                graph_metadata["tree_snapshot"] = tree.to_dict()
                generations.append(TreeDraftGeneration(tree=tree, metadata=graph_metadata))
            return generations

        return self.generate_tree_batch(
            [
                {
                    "prefix_ids": [int(token_id) for token_id in request["prefix_ids"]],
                    "max_depth": int(request.get("max_depth", 0)),
                    "max_branches": int(request.get("max_branches", 0)),
                    "max_nodes": int(request.get("max_nodes", 0)),
                    "request_id": request.get("request_id"),
                    "runner_id": request.get("runner_id"),
                    "metadata": {
                        **dict(request.get("metadata") or {}),
                        "official_state_grow_fallback": True,
                    },
                }
                for request in requests
            ]
        )

    def generate_official_proactive(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Generate official SpecEdge proactive POST_* nodes via a model backend."""
        metadata = dict(request.get("metadata") or {})
        graph_generator = getattr(self.model, "generate_official_proactive_graph", None)
        if callable(graph_generator) and not bool(metadata.get("disable_graph_tree_draft", False)):
            runner_id = str(request.get("runner_id") or self.runner_id or self.model.runner_id)
            enriched = {
                **dict(request),
                "runner_id": runner_id,
                "metadata": {
                    **dict(self.metadata),
                    **metadata,
                    "request_id": request.get("request_id"),
                    "runner_id": runner_id,
                    "device": getattr(self.model, "device", None),
                    "backend_capabilities": self.model.backend_capabilities().to_dict(),
                    "official_proactive_graph_requested": True,
                },
            }
            return graph_generator(enriched)
        return None
