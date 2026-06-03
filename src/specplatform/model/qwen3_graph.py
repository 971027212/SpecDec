from __future__ import annotations

"""Qwen3 explicit-KV CUDA graph backend selection boundary.

The real SpecEdge graph backend needs custom Qwen3 attention, KV cache indexing,
and CUDA graph capture. This module provides the platform-facing loader and an
honest fallback path: experiments can request `qwen3_graph` by default, while
artifacts explicitly say whether graph execution was actually enabled.
"""

from dataclasses import dataclass, field, replace
import math
import os
import threading
from time import perf_counter_ns
from typing import Any

from specplatform.core import CandidateNode, CandidateTree
from specplatform.model.base import (
    CausalLMRunner,
    LinearForwardInput,
    LinearForwardOutput,
    ModelBackendCapabilities,
    ModelForwardInput,
    ModelForwardOutput,
    TopKToken,
    TreeForwardChoice,
    TreeForwardInput,
    TreeForwardNode,
    TreeForwardOutput,
)
from specplatform.model.qwen3_graph_backend.graph import BatchGraphEngine, GraphEngine
from specplatform.model.qwen3_graph_backend.qwen3 import Qwen3ForCausalLM
from specplatform.model.transformers import CachedTransformersCausalLMRunner, TransformersCausalLMRunner


class Qwen3GraphBackendUnavailable(RuntimeError):
    """Raised when qwen3_graph is requested without allowed fallback."""


@dataclass
class Qwen3GraphCausalLMRunner(CausalLMRunner):
    """Official SpecEdge-style Qwen3 runner with explicit KV cache and CUDA graphs.

    Linear verification uses a single-request graph engine. SpecEdge tree
    verification uses a separate batched graph engine so the target server can
    verify many request trees with official-style explicit KV indices.
    """

    runner_id: str
    tokenizer: Any
    model: Any
    engine: GraphEngine
    max_len: int
    max_graph_tokens: int
    device: str
    batch_engine: BatchGraphEngine | None = None
    max_graph_batch_size: int = 1
    backend_name: str = "qwen3_graph"
    backend_fallback: bool = False
    fallback_reason: str | None = None
    _official_draft_batch_indices: dict[str, int] = None  # type: ignore[assignment]
    _graph_lock: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self._official_draft_batch_indices is None:
            self._official_draft_batch_indices = {}
        if self._graph_lock is None:
            self._graph_lock = getattr(self.engine, "_lock", None) or threading.RLock()

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        *,
        runner_id: str,
        device: str | None = "cuda",
        torch_dtype: str | None = "auto",
        device_map: str | None = None,
        trust_remote_code: bool = True,
        max_graph_len: int | None = None,
        max_graph_tokens: int | None = None,
        max_graph_batch_size: int | None = None,
    ) -> "Qwen3GraphCausalLMRunner":
        """Load the custom Qwen3 graph model and capture fixed-shape graphs."""
        import torch
        from transformers import AutoTokenizer

        if device is None:
            device = "cuda"
        if not torch.cuda.is_available():
            raise Qwen3GraphBackendUnavailable("CUDA is not available for qwen3_graph backend")
        if "qwen3" not in str(model_path).lower():
            raise Qwen3GraphBackendUnavailable("qwen3_graph backend only supports Qwen3 model paths")

        dtype = _resolve_torch_dtype(torch, torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        model_kwargs: dict[str, Any] = {}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        model = Qwen3ForCausalLM.from_pretrained(model_path, **model_kwargs)
        if device_map is None:
            model = model.to(device)
        model.eval()

        model_max_len = int(
            getattr(model.config, "max_position_embeddings", None)
            or getattr(model.config, "seq_length", None)
            or 32768
        )
        configured_max_len = max_graph_len or os.environ.get("SPECPLATFORM_QWEN3_GRAPH_MAX_LEN")
        max_len = min(model_max_len, int(configured_max_len)) if configured_max_len else model_max_len
        if max_len <= 0:
            raise Qwen3GraphBackendUnavailable("SPECPLATFORM_QWEN3_GRAPH_MAX_LEN must be positive")
        graph_tokens = int(max_graph_tokens or os.environ.get("SPECPLATFORM_QWEN3_GRAPH_MAX_TOKENS", "32"))
        if graph_tokens <= 0:
            raise Qwen3GraphBackendUnavailable("SPECPLATFORM_QWEN3_GRAPH_MAX_TOKENS must be positive")
        graph_batch_size = int(
            max_graph_batch_size or os.environ.get("SPECPLATFORM_QWEN3_GRAPH_MAX_BATCH_SIZE", "8")
        )
        if graph_batch_size <= 0:
            raise Qwen3GraphBackendUnavailable("SPECPLATFORM_QWEN3_GRAPH_MAX_BATCH_SIZE must be positive")
        input_device = str(next(model.parameters()).device)
        engine = GraphEngine(model=model, max_len=max_len, max_n_beams=graph_tokens)
        batch_engine = BatchGraphEngine(
            model=model,
            max_len=max_len,
            max_batch_size=graph_batch_size,
            max_n_beams=graph_tokens,
        )
        return cls(
            runner_id=runner_id,
            tokenizer=tokenizer,
            model=model,
            engine=engine,
            batch_engine=batch_engine,
            max_len=max_len,
            max_graph_tokens=graph_tokens,
            max_graph_batch_size=graph_batch_size,
            device=input_device,
        )

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """Use the graph path for a plain full-sequence forward."""
        if not request.input_ids:
            raise ValueError("Qwen3GraphCausalLMRunner.forward requires input_ids.")
        logits = self._verify_path_logits(prefix_ids=[int(request.input_ids[0])], draft_tokens=list(request.input_ids[1:]))
        return ModelForwardOutput(logits=logits.detach().float().cpu().tolist(), metadata={"runner_id": self.runner_id})

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        if not prefix_ids:
            raise ValueError("Qwen3GraphCausalLMRunner.next_token_logits requires a non-empty prefix.")
        logits = self._verify_path_logits(prefix_ids=[int(token_id) for token_id in prefix_ids], draft_tokens=[])
        return logits[0, -1].detach().float().cpu().tolist()

    def next_token_topk(self, prefix_ids: list[int], k: int) -> list[TopKToken]:
        import torch

        if k <= 0:
            return []
        logits = self._verify_path_logits(prefix_ids=[int(token_id) for token_id in prefix_ids], draft_tokens=[])[0, -1]
        k = min(int(k), int(logits.numel()))
        values, indices = torch.topk(torch.log_softmax(logits.detach().float(), dim=-1), k=k, largest=True, sorted=True)
        return [
            TopKToken(token_id=int(token_id), logprob=float(logprob), rank=rank)
            for rank, (token_id, logprob) in enumerate(zip(indices.cpu().tolist(), values.cpu().tolist()))
        ]

    def generate_tree_topk_graph(
        self,
        *,
        prefix_ids: list[int],
        max_depth: int,
        max_branches: int,
        max_nodes: int,
        request_id: str | None = None,
        runner_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Draft a top-k tree with persistent explicit KV and graph frontier steps.

        This is the draft-side counterpart of the target tree verifier: the
        prefix is prefetched once, then each frontier depth is expanded with a
        graph-captured forward over the frontier nodes and an explicit tree
        attention mask.
        """
        import torch

        if not prefix_ids:
            raise ValueError("qwen3_graph tree drafting requires a non-empty prefix.")
        max_depth = int(max_depth)
        max_branches = int(max_branches)
        max_nodes = int(max_nodes)
        if max_depth <= 0 or max_branches <= 0 or max_nodes <= 0:
            return {
                "tree": CandidateTree(root_prefix_len=len(prefix_ids), nodes=[]),
                "metadata": {"graph_tree_draft": True, "draft_token_forward_events": []},
            }
        if len(prefix_ids) + max_nodes > int(self.max_len):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph draft tree length {len(prefix_ids) + max_nodes} exceeds cache max {self.max_len}"
            )
        if max_branches > int(self.max_graph_tokens):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph max_branches {max_branches} exceeds captured max {self.max_graph_tokens}"
            )

        runner_id = str(runner_id or self.runner_id)
        base_metadata = dict(metadata or {})
        self.engine.reset()
        prefill_tokens = [int(token_id) for token_id in prefix_ids]
        prefill_len = len(prefill_tokens)
        input_ids = torch.tensor([prefill_tokens], dtype=torch.long, device=self.device)
        position_ids = torch.arange(prefill_len, dtype=torch.long, device=self.device).unsqueeze(0)
        cache_seq_indices = torch.arange(prefill_len, dtype=torch.long, device=self.device)
        attention_mask = _causal_allow_mask(
            torch,
            query_seq_indices=cache_seq_indices,
            max_len=self.max_len,
            dtype=self.model.dtype,
            device=self.device,
        )
        prefill_start_ns = perf_counter_ns()
        prefix_logits = self.engine.prefill(
            input_ids=input_ids,
            position_ids=position_ids,
            batch_idx=0,
            cache_seq_indices=cache_seq_indices,
            attention_mask=attention_mask,
        )
        prefill_end_ns = perf_counter_ns()

        nodes: list[CandidateNode] = []
        detail_events: list[dict[str, Any]] = []
        next_node_id = 0
        candidate_ids: list[int] = []

        root_topk = _topk_tokens(torch, prefix_logits[0, -1], min(max_branches, max_nodes))
        detail_events.append(
            {
                "phase": "draft.graph_prefill_topk",
                "start_ns": prefill_start_ns,
                "end_ns": prefill_end_ns,
                "duration_ms": _duration_ms(prefill_start_ns, prefill_end_ns),
                "prefix_len": prefill_len,
                "depth": 0,
                "batch_size": 1,
                "topk_batch_kind": "qwen3_graph_prefill_topk",
                "request_id": request_id,
                "topk": [candidate.__dict__ for candidate in root_topk],
            }
        )
        for candidate in root_topk:
            node = CandidateNode(
                node_id=next_node_id,
                parent_id=None,
                token_id=int(candidate.token_id),
                depth=1,
                draft_logprob=float(candidate.logprob),
                draft_worker_id=runner_id,
            )
            nodes.append(node)
            candidate_ids.append(node.node_id)
            next_node_id += 1
            if len(nodes) >= max_nodes:
                break

        for depth in range(1, max_depth):
            if not candidate_ids:
                break
            selected_candidate_ids = _top_candidate_ids(nodes, candidate_ids, max_count=int(self.max_graph_tokens))
            if not selected_candidate_ids:
                break
            selected_candidate_set = set(selected_candidate_ids)
            candidate_ids = [node_id for node_id in candidate_ids if node_id not in selected_candidate_set]
            incoming_children: list[dict[str, Any]] = []
            for chunk_start in range(0, len(selected_candidate_ids), int(self.max_graph_tokens)):
                candidate_chunk_ids = selected_candidate_ids[chunk_start : chunk_start + int(self.max_graph_tokens)]
                candidate_nodes = [_node_by_id(nodes, node_id) for node_id in candidate_chunk_ids]
                chunk_size = len(candidate_nodes)
                input_ids = torch.tensor(
                    [[int(node.token_id) for node in candidate_nodes]],
                    dtype=torch.long,
                    device=self.device,
                )
                position_ids = torch.tensor(
                    [[prefill_len + int(node.depth) - 1 for node in candidate_nodes]],
                    dtype=torch.long,
                    device=self.device,
                )
                cache_batch_indices = torch.zeros((chunk_size,), dtype=torch.long, device=self.device)
                cache_seq_indices = torch.tensor(
                    [prefill_len + _node_index(nodes, int(node.node_id)) for node in candidate_nodes],
                    dtype=torch.long,
                    device=self.device,
                )
                attention_mask = torch.zeros((1, 1, chunk_size, self.max_len), dtype=self.model.dtype, device=self.device)
                tree_forward_nodes = [
                    TreeForwardNode(
                        node_id=int(tree_node.node_id),
                        parent_id=None if tree_node.parent_id is None else int(tree_node.parent_id),
                        token_id=int(tree_node.token_id),
                        depth=int(tree_node.depth),
                    )
                    for tree_node in nodes
                ]
                for row_index, node in enumerate(candidate_nodes):
                    _fill_tree_node_allow_mask(
                        attention_mask[0, 0, row_index],
                        prefix_len=prefill_len,
                        node=TreeForwardNode(
                            node_id=int(node.node_id),
                            parent_id=None if node.parent_id is None else int(node.parent_id),
                            token_id=int(node.token_id),
                            depth=int(node.depth),
                        ),
                        node_index=_node_index(nodes, int(node.node_id)),
                        nodes=tree_forward_nodes,
                    )
                forward_start_ns = perf_counter_ns()
                logits = self.engine.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    cache_batch_indices=cache_batch_indices,
                    cache_seq_indices=cache_seq_indices,
                    attention_mask=attention_mask,
                )
                forward_end_ns = perf_counter_ns()
                shared_batch_event_id = f"draft_graph_topk_batch:{forward_start_ns}:{depth}:{chunk_size}"
                for batch_index, parent_node in enumerate(candidate_nodes):
                    topk = _topk_tokens(torch, logits[0, batch_index], max_branches)
                    detail_events.append(
                        {
                            "phase": "draft.graph_topk",
                            "start_ns": forward_start_ns,
                            "end_ns": forward_end_ns,
                            "duration_ms": _duration_ms(forward_start_ns, forward_end_ns) / max(1, chunk_size),
                            "shared_duration_ms": _duration_ms(forward_start_ns, forward_end_ns),
                            "batch_size": chunk_size,
                            "batch_index": batch_index,
                            "shared_batch_event_id": shared_batch_event_id,
                            "topk_batch_kind": "qwen3_graph_frontier_topk",
                            "depth": depth,
                            "parent_node_id": int(parent_node.node_id),
                            "prefix_len": prefill_len + int(parent_node.depth),
                            "explicit_kv_cache": True,
                            "cuda_graph": True,
                            "topk": [candidate.__dict__ for candidate in topk],
                        }
                    )
                    for candidate in topk:
                        incoming_children.append(
                            {
                                "parent_id": int(parent_node.node_id),
                                "token_id": int(candidate.token_id),
                                "depth": int(parent_node.depth) + 1,
                                "draft_logprob": float(parent_node.draft_logprob or 0.0) + float(candidate.logprob),
                                "draft_worker_id": runner_id,
                            }
                        )
            kept_children = _official_budget_children(nodes, incoming_children, max_nodes=max_nodes)
            for child in kept_children:
                node = CandidateNode(
                    node_id=next_node_id,
                    parent_id=int(child["parent_id"]),
                    token_id=int(child["token_id"]),
                    depth=int(child["depth"]),
                    draft_logprob=float(child["draft_logprob"]),
                    draft_worker_id=str(child["draft_worker_id"]),
                )
                nodes.append(node)
                candidate_ids.append(node.node_id)
                next_node_id += 1
            if len(nodes) > max_nodes:
                nodes = _trim_nodes_with_ancestors(nodes, max_nodes=max_nodes)
                active_ids = {int(node.node_id) for node in nodes}
                candidate_ids = [node_id for node_id in candidate_ids if node_id in active_ids]

        tree = CandidateTree(root_prefix_len=prefill_len, nodes=nodes)
        tree.validate()
        graph_metadata = {
            **base_metadata,
            "graph_tree_draft": True,
            "tree_draft_backend": "qwen3_graph_frontier",
            "draft_token_forward_events": detail_events,
            "max_graph_tokens": self.max_graph_tokens,
            "official_budget_pruning": True,
            "explicit_kv_cache": True,
            "cuda_graph": True,
        }
        return {"tree": tree, "metadata": graph_metadata}

    def generate_tree_topk_batch_graph(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Draft top-k trees for multiple requests with official-style BatchGraphEngine.

        This path mirrors the official edge draft loop more closely than the
        single-request helper: request slots are prefetched into one batched KV
        cache, candidate beams are selected per slot, and each growth step uses
        one fixed-shape BatchGraphEngine forward.
        """
        import torch

        if not requests:
            return []
        if self.batch_engine is None:
            raise Qwen3GraphBackendUnavailable("qwen3_graph batch tree drafting requires BatchGraphEngine")
        if len(requests) > int(self.max_graph_batch_size):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph draft batch size {len(requests)} exceeds captured max {self.max_graph_batch_size}"
            )

        normalized: list[dict[str, Any]] = []
        used_batch_indices: set[int] = set()
        for index, raw_request in enumerate(requests):
            prefix_ids = [int(token_id) for token_id in raw_request.get("prefix_ids", [])]
            if not prefix_ids:
                raise ValueError("qwen3_graph batch tree drafting requires non-empty prefixes.")
            max_depth = int(raw_request.get("max_depth", 0))
            max_branches = int(raw_request.get("max_branches", 0))
            max_nodes = int(raw_request.get("max_nodes", 0))
            if len(prefix_ids) + max(max_nodes, 0) > int(self.max_len):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph draft tree length {len(prefix_ids) + max_nodes} exceeds cache max {self.max_len}"
                )
            if max_branches > int(self.max_graph_tokens):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph max_branches {max_branches} exceeds captured max {self.max_graph_tokens}"
                )
            if max_nodes > int(self.max_graph_tokens):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph max_nodes {max_nodes} exceeds captured max {self.max_graph_tokens}"
                )
            metadata = dict(raw_request.get("metadata") or {})
            batch_index = raw_request.get("draft_batch_index")
            if batch_index is None:
                batch_index = metadata.get("official_draft_batch_index", metadata.get("batch_index", index))
            batch_idx = int(batch_index)
            if batch_idx < 0 or batch_idx >= int(self.max_graph_batch_size):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph batch tree draft index {batch_idx} exceeds captured batch size {self.max_graph_batch_size}"
                )
            if batch_idx in used_batch_indices:
                raise ValueError(f"Duplicate qwen3_graph batch tree draft index: {batch_idx}")
            used_batch_indices.add(batch_idx)
            normalized.append(
                {
                    "prefix_ids": prefix_ids,
                    "prefix_len": len(prefix_ids),
                    "max_depth": max_depth,
                    "max_branches": max_branches,
                    "max_nodes": max_nodes,
                    "request_id": raw_request.get("request_id"),
                    "runner_id": str(raw_request.get("runner_id") or self.runner_id),
                    "metadata": metadata,
                    "batch_index": batch_idx,
                    "request_order_index": index,
                }
            )
            if raw_request.get("request_id") is not None:
                self._official_draft_batch_indices[str(raw_request["request_id"])] = batch_idx

        pad_token_id = _pad_token_id(self.tokenizer, self.model)
        max_batch = int(self.max_graph_batch_size)
        max_beams = int(self.max_graph_tokens)
        dtype = self.model.dtype
        device = self.device
        score_floor = float(normalized[0]["metadata"].get("official_score_floor", -10.0))
        decay_factor = float(normalized[0]["metadata"].get("official_decay_factor", math.log(0.9)))

        if hasattr(self.batch_engine, "remove_requests"):
            self.batch_engine.remove_requests(
                torch.tensor(sorted(used_batch_indices), dtype=torch.long, device=device)
            )
        else:
            self.batch_engine.reset()
        prefix_last_logits: list[Any | None] = [None for _ in normalized]
        detail_events_by_row: list[list[dict[str, Any]]] = [[] for _ in normalized]
        nodes_by_row: list[list[CandidateNode]] = [[] for _ in normalized]
        statuses_by_row: list[dict[int, int]] = [{} for _ in normalized]
        candidate_ids_by_row: list[list[int]] = [[] for _ in normalized]
        next_node_ids = [0 for _ in normalized]

        for row_index, request in enumerate(normalized):
            batch_idx = int(request["batch_index"])
            prefix_ids = request["prefix_ids"]
            prefix_len = int(request["prefix_len"])
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            position_ids = torch.arange(prefix_len, dtype=torch.long, device=device).unsqueeze(0)
            cache_seq_indices = torch.arange(prefix_len, dtype=torch.long, device=device)
            attention_mask = _causal_allow_mask(
                torch,
                query_seq_indices=cache_seq_indices,
                max_len=self.max_len,
                dtype=dtype,
                device=device,
            )
            prefill_start_ns = perf_counter_ns()
            logits = self.batch_engine.prefill(
                input_ids=input_ids,
                position_ids=position_ids,
                batch_idx=batch_idx,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )
            prefill_end_ns = perf_counter_ns()
            prefix_last_logits[row_index] = logits[0, -1].detach()
            detail_events_by_row[row_index].append(
                {
                    "phase": "draft.batch_graph_prefill",
                    "start_ns": prefill_start_ns,
                    "end_ns": prefill_end_ns,
                    "duration_ms": _duration_ms(prefill_start_ns, prefill_end_ns),
                    "prefix_len": prefix_len,
                    "batch_size": len(normalized),
                    "batch_index": batch_idx,
                    "request_order_index": row_index,
                    "topk_batch_kind": "qwen3_batch_graph_prefill",
                    "request_id": request["request_id"],
                    "explicit_kv_cache": True,
                    "cuda_graph": True,
                }
            )

        for row_index, request in enumerate(normalized):
            max_depth = int(request["max_depth"])
            max_nodes = int(request["max_nodes"])
            max_branches = int(request["max_branches"])
            if max_depth <= 0 or max_nodes <= 0 or max_branches <= 0:
                continue
            logits = prefix_last_logits[row_index]
            if logits is None:
                continue
            root_topk = _topk_tokens(torch, logits, min(max_branches, max_nodes))
            incoming_roots = [
                {
                    "parent_id": None,
                    "token_id": int(candidate.token_id),
                    "depth": 1,
                    "draft_logprob": float(decay_factor) + float(candidate.logprob),
                    "draft_worker_id": request["runner_id"],
                }
                for candidate in root_topk
            ]
            kept_roots = _official_budget_children(
                nodes_by_row[row_index],
                incoming_roots,
                max_nodes=max_nodes,
                score_floor=score_floor,
            )
            detail_events_by_row[row_index].append(
                {
                    "phase": "draft.batch_graph_root_topk",
                    "start_ns": detail_events_by_row[row_index][-1]["start_ns"],
                    "end_ns": detail_events_by_row[row_index][-1]["end_ns"],
                    "duration_ms": detail_events_by_row[row_index][-1]["duration_ms"],
                    "prefix_len": int(request["prefix_len"]),
                    "depth": 0,
                    "batch_size": len(normalized),
                    "batch_index": int(request["batch_index"]),
                    "request_order_index": row_index,
                    "topk_batch_kind": "qwen3_batch_graph_root_topk",
                    "official_decay_factor": decay_factor,
                    "official_score_floor": score_floor,
                    "topk": [candidate.__dict__ for candidate in root_topk],
                }
            )
            for child in kept_roots:
                node = CandidateNode(
                    node_id=next_node_ids[row_index],
                    parent_id=None,
                    token_id=int(child["token_id"]),
                    depth=1,
                    draft_logprob=float(child["draft_logprob"]),
                    draft_worker_id=str(child["draft_worker_id"]),
                )
                nodes_by_row[row_index].append(node)
                statuses_by_row[row_index][int(node.node_id)] = 15
                candidate_ids_by_row[row_index].append(int(node.node_id))
                next_node_ids[row_index] += 1

        max_depth_all = max(int(request["max_depth"]) for request in normalized)
        for depth in range(1, max_depth_all):
            selected_by_row: list[list[CandidateNode]] = [[] for _ in normalized]
            for row_index, request in enumerate(normalized):
                if depth >= int(request["max_depth"]):
                    continue
                selected_ids = _top_candidate_ids(
                    nodes_by_row[row_index],
                    candidate_ids_by_row[row_index],
                    max_count=max_beams,
                )
                if not selected_ids:
                    continue
                selected_set = set(selected_ids)
                candidate_ids_by_row[row_index] = [
                    node_id
                    for node_id in candidate_ids_by_row[row_index]
                    if node_id not in selected_set
                ]
                selected_nodes = [_node_by_id(nodes_by_row[row_index], node_id) for node_id in selected_ids]
                selected_by_row[row_index] = selected_nodes
                for node in selected_nodes:
                    statuses_by_row[row_index][int(node.node_id)] = 10
            if not any(selected_by_row):
                break

            input_ids = torch.full((max_batch, max_beams), pad_token_id, dtype=torch.long, device=device)
            position_ids = torch.full((max_batch, max_beams), self.max_len - 1, dtype=torch.long, device=device)
            cache_seq_indices = torch.full((max_batch, max_beams), self.max_len - 1, dtype=torch.long, device=device)
            attention_mask = torch.zeros((max_batch, 1, max_beams, self.max_len), dtype=dtype, device=device)

            for row_index, selected_nodes in enumerate(selected_by_row):
                request = normalized[row_index]
                batch_idx = int(request["batch_index"])
                prefix_len = int(request["prefix_len"])
                tree_forward_nodes = [
                    TreeForwardNode(
                        node_id=int(tree_node.node_id),
                        parent_id=None if tree_node.parent_id is None else int(tree_node.parent_id),
                        token_id=int(tree_node.token_id),
                        depth=int(tree_node.depth),
                    )
                    for tree_node in nodes_by_row[row_index]
                ]
                for beam_index, node in enumerate(selected_nodes):
                    input_ids[batch_idx, beam_index] = int(node.token_id)
                    position_ids[batch_idx, beam_index] = prefix_len + int(node.depth) - 1
                    cache_seq_indices[batch_idx, beam_index] = prefix_len + _node_index(
                        nodes_by_row[row_index],
                        int(node.node_id),
                    )
                    _fill_tree_node_allow_mask(
                        attention_mask[batch_idx, 0, beam_index],
                        prefix_len=prefix_len,
                        node=TreeForwardNode(
                            node_id=int(node.node_id),
                            parent_id=None if node.parent_id is None else int(node.parent_id),
                            token_id=int(node.token_id),
                            depth=int(node.depth),
                        ),
                        node_index=_node_index(nodes_by_row[row_index], int(node.node_id)),
                        nodes=tree_forward_nodes,
                    )

            cache_batch_indices = torch.arange(max_batch, dtype=torch.long, device=device).repeat_interleave(max_beams)
            forward_start_ns = perf_counter_ns()
            logits = self.batch_engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices.flatten(),
                attention_mask=attention_mask,
            )
            forward_end_ns = perf_counter_ns()
            shared_batch_event_id = f"draft_batch_graph_topk_batch:{forward_start_ns}:{depth}:{len(normalized)}"

            for row_index, selected_nodes in enumerate(selected_by_row):
                if not selected_nodes:
                    continue
                request = normalized[row_index]
                batch_idx = int(request["batch_index"])
                max_nodes = int(request["max_nodes"])
                max_branches = int(request["max_branches"])
                incoming_children: list[dict[str, Any]] = []
                for beam_index, parent_node in enumerate(selected_nodes):
                    topk = _topk_tokens(torch, logits[batch_idx, beam_index], max_branches)
                    detail_events_by_row[row_index].append(
                        {
                            "phase": "draft.batch_graph_topk",
                            "start_ns": forward_start_ns,
                            "end_ns": forward_end_ns,
                            "duration_ms": _duration_ms(forward_start_ns, forward_end_ns) / max(1, len(normalized)),
                            "shared_duration_ms": _duration_ms(forward_start_ns, forward_end_ns),
                            "batch_size": len(normalized),
                            "batch_index": batch_idx,
                            "request_order_index": row_index,
                            "beam_index": beam_index,
                            "shared_batch_event_id": shared_batch_event_id,
                            "topk_batch_kind": "qwen3_batch_graph_frontier_topk",
                            "depth": depth,
                            "parent_node_id": int(parent_node.node_id),
                            "prefix_len": int(request["prefix_len"]) + int(parent_node.depth),
                            "official_decay_factor": decay_factor,
                            "official_score_floor": score_floor,
                            "explicit_kv_cache": True,
                            "cuda_graph": True,
                            "topk": [candidate.__dict__ for candidate in topk],
                        }
                    )
                    for candidate in topk:
                        incoming_children.append(
                            {
                                "parent_id": int(parent_node.node_id),
                                "token_id": int(candidate.token_id),
                                "depth": int(parent_node.depth) + 1,
                                "draft_logprob": float(parent_node.draft_logprob or 0.0)
                                + float(decay_factor)
                                + float(candidate.logprob),
                                "draft_worker_id": request["runner_id"],
                            }
                        )
                kept_children = _official_budget_children(
                    nodes_by_row[row_index],
                    incoming_children,
                    max_nodes=max_nodes,
                    score_floor=score_floor,
                )
                for child in kept_children:
                    if int(request["prefix_len"]) + len(nodes_by_row[row_index]) >= int(self.max_len):
                        break
                    node = CandidateNode(
                        node_id=next_node_ids[row_index],
                        parent_id=int(child["parent_id"]),
                        token_id=int(child["token_id"]),
                        depth=int(child["depth"]),
                        draft_logprob=float(child["draft_logprob"]),
                        draft_worker_id=str(child["draft_worker_id"]),
                    )
                    nodes_by_row[row_index].append(node)
                    statuses_by_row[row_index][int(node.node_id)] = 15
                    candidate_ids_by_row[row_index].append(int(node.node_id))
                    next_node_ids[row_index] += 1

        outputs: list[dict[str, Any]] = []
        for row_index, request in enumerate(normalized):
            batch_idx = int(request["batch_index"])
            max_nodes = int(request["max_nodes"])
            old_nodes = list(nodes_by_row[row_index])
            nodes = _trim_nodes_with_ancestors(nodes_by_row[row_index], max_nodes=max_nodes)
            budget_gathered = _gather_batch_tree_after_trim(
                torch,
                batch_engine=self.batch_engine,
                batch_idx=batch_idx,
                prefix_len=int(request["prefix_len"]),
                old_nodes=old_nodes,
                new_nodes=nodes,
                device=device,
                detail_events=detail_events_by_row[row_index],
                phase="draft.batch_graph_budget_gather",
            )
            kept_ids = {int(node.node_id) for node in nodes}
            statuses = {
                int(node_id): int(status)
                for node_id, status in statuses_by_row[row_index].items()
                if int(node_id) in kept_ids
            }
            tree = CandidateTree(root_prefix_len=int(request["prefix_len"]), nodes=nodes)
            tree.validate()
            outputs.append(
                {
                    "tree": tree,
                    "metadata": {
                        **dict(request["metadata"]),
                        "graph_tree_draft": True,
                        "official_batch_tree_draft": True,
                        "tree_draft_backend": "qwen3_batch_graph_official",
                        "draft_token_forward_events": detail_events_by_row[row_index],
                        "tree_node_statuses": {str(node_id): status for node_id, status in statuses.items()},
                        "max_graph_tokens": self.max_graph_tokens,
                        "max_graph_batch_size": self.max_graph_batch_size,
                        "official_budget_pruning": True,
                        "official_decay_factor": decay_factor,
                        "official_score_floor": score_floor,
                        "official_budget_kv_gather": bool(budget_gathered),
                        "explicit_kv_cache": True,
                        "cuda_graph": True,
                        "batch_index": batch_idx,
                        "official_draft_batch_index": batch_idx,
                        "request_order_index": row_index,
                        "official_draft_request_id": request["request_id"],
                        "batch_size": len(normalized),
                    },
                }
            )
        return outputs

    def grow_official_tree_batch_graph(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Grow persistent official SpecEdge draft trees without full prefix prefill."""
        import torch

        if not requests:
            return []
        if self.batch_engine is None:
            raise Qwen3GraphBackendUnavailable("qwen3_graph official state grow requires BatchGraphEngine")
        if len(requests) > int(self.max_graph_batch_size):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph state grow batch size {len(requests)} exceeds captured max {self.max_graph_batch_size}"
            )

        normalized: list[dict[str, Any]] = []
        missing_persistent_row = False
        used_batch_indices: set[int] = set()
        for request_order_index, raw_request in enumerate(requests):
            prefix_ids = [int(token_id) for token_id in raw_request.get("prefix_ids", [])]
            if not prefix_ids:
                raise ValueError("qwen3_graph official state grow requires non-empty prefixes.")
            metadata = dict(raw_request.get("metadata") or {})
            max_depth = int(raw_request.get("max_depth", 0))
            max_branches = int(raw_request.get("max_branches", 0))
            max_nodes = int(raw_request.get("max_nodes", 0))
            if len(prefix_ids) + max(max_nodes, 0) > int(self.max_len):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph official state grow length {len(prefix_ids) + max_nodes} exceeds cache max {self.max_len}"
                )
            if max_branches > int(self.max_graph_tokens):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph max_branches {max_branches} exceeds captured max {self.max_graph_tokens}"
                )
            if max_nodes > int(self.max_graph_tokens):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph max_nodes {max_nodes} exceeds captured max {self.max_graph_tokens}"
                )

            tree_payload = raw_request.get("tree")
            if isinstance(tree_payload, CandidateTree):
                tree = tree_payload
            elif tree_payload:
                tree = CandidateTree.from_dict(dict(tree_payload))
            else:
                tree = CandidateTree(root_prefix_len=len(prefix_ids), nodes=[])
            if tree.root_prefix_len != len(prefix_ids):
                raise ValueError("official state grow tree root_prefix_len must match prefix_ids length.")
            tree.validate()
            nodes = [
                CandidateNode(
                    node_id=int(node.node_id),
                    parent_id=None if node.parent_id is None else int(node.parent_id),
                    token_id=int(node.token_id),
                    depth=int(node.depth),
                    draft_logprob=node.draft_logprob,
                    draft_worker_id=str(node.draft_worker_id or raw_request.get("runner_id") or self.runner_id),
                )
                for node in tree.nodes
            ]
            statuses = _official_status_map(
                raw_request.get("tree_node_statuses") or metadata.get("tree_node_statuses") or {},
                nodes=nodes,
            )

            batch_index = raw_request.get("draft_batch_index")
            if batch_index is None:
                batch_index = metadata.get("official_draft_batch_index", metadata.get("batch_index"))
            if batch_index is None and raw_request.get("request_id") is not None:
                batch_index = self._official_draft_batch_indices.get(str(raw_request["request_id"]))
            if batch_index is None:
                missing_persistent_row = True
                batch_idx: int | None = None
            else:
                batch_idx = int(batch_index)
                if batch_idx < 0 or batch_idx >= int(self.max_graph_batch_size):
                    raise Qwen3GraphBackendUnavailable(
                        f"qwen3_graph official state grow batch index {batch_idx} exceeds captured batch size {self.max_graph_batch_size}"
                    )
                if batch_idx in used_batch_indices:
                    raise ValueError(f"Duplicate official draft batch index in grow batch: {batch_idx}")
                used_batch_indices.add(batch_idx)
                if raw_request.get("request_id") is not None:
                    self._official_draft_batch_indices[str(raw_request["request_id"])] = batch_idx

            normalized.append(
                {
                    "prefix_ids": prefix_ids,
                    "prefix_len": len(prefix_ids),
                    "nodes": nodes,
                    "statuses": statuses,
                    "needs_prefix_tail_forward": bool(raw_request.get("needs_prefix_tail_forward", False)),
                    "max_depth": max_depth,
                    "max_branches": max_branches,
                    "max_nodes": max_nodes,
                    "request_id": raw_request.get("request_id"),
                    "runner_id": str(raw_request.get("runner_id") or self.runner_id),
                    "metadata": metadata,
                    "batch_index": batch_idx,
                    "request_order_index": request_order_index,
                    "next_node_id": (max((int(node.node_id) for node in nodes), default=-1) + 1),
                }
            )

        if missing_persistent_row:
            fallback_results = self.generate_tree_topk_batch_graph(
                [
                    {
                        "prefix_ids": list(request["prefix_ids"]),
                        "max_depth": int(request["max_depth"]),
                        "max_branches": int(request["max_branches"]),
                        "max_nodes": int(request["max_nodes"]),
                        "request_id": request["request_id"],
                        "runner_id": request["runner_id"],
                        "metadata": {
                            **dict(request["metadata"]),
                            "official_state_grow_fallback": True,
                            "official_persistent_kv_reused": False,
                            "official_persistent_kv_reuse_reason": "missing_draft_batch_index",
                        },
                    }
                    for request in normalized
                ]
            )
            for result in fallback_results:
                result["metadata"] = {
                    **dict(result.get("metadata") or {}),
                    "official_state_grow": True,
                    "official_state_grow_fallback": True,
                    "official_persistent_kv_reused": False,
                    "official_persistent_kv_reuse_reason": "missing_draft_batch_index",
                    "official_needs_prefix_tail_forward": False,
                }
            return fallback_results

        pad_token_id = _pad_token_id(self.tokenizer, self.model)
        max_batch = int(self.max_graph_batch_size)
        max_beams = int(self.max_graph_tokens)
        dtype = self.model.dtype
        device = self.device
        score_floor = float(normalized[0]["metadata"].get("official_score_floor", -10.0))
        decay_factor = float(normalized[0]["metadata"].get("official_decay_factor", math.log(0.9)))
        detail_events_by_request: list[list[dict[str, Any]]] = [[] for _ in normalized]

        max_depth_all = max(int(request["max_depth"]) for request in normalized)
        for grow_step in range(max(0, max_depth_all)):
            selected_by_request: list[list[dict[str, Any]]] = [[] for _ in normalized]
            for request_index, request in enumerate(normalized):
                max_depth = int(request["max_depth"])
                if max_depth <= 0:
                    continue
                sources: list[dict[str, Any]] = []
                if bool(request["needs_prefix_tail_forward"]):
                    sources.append(
                        {
                            "kind": "prefix_tail",
                            "token_id": int(request["prefix_ids"][-1]),
                            "score": 0.0,
                            "node_id": -1,
                            "depth": 0,
                        }
                    )
                for node in request["nodes"]:
                    if int(request["statuses"].get(int(node.node_id), 15)) != 15:
                        continue
                    if int(node.depth) >= max_depth:
                        continue
                    sources.append(
                        {
                            "kind": "node",
                            "node": node,
                            "token_id": int(node.token_id),
                            "score": float("-inf") if node.draft_logprob is None else float(node.draft_logprob),
                            "node_id": int(node.node_id),
                            "depth": int(node.depth),
                        }
                    )
                sources.sort(key=lambda item: (float(item["score"]), -int(item["node_id"])), reverse=True)
                selected = sources[:max_beams]
                if not selected:
                    continue
                for source in selected:
                    if source["kind"] == "prefix_tail":
                        request["needs_prefix_tail_forward"] = False
                    else:
                        request["statuses"][int(source["node"].node_id)] = 10
                selected_by_request[request_index] = selected
            if not any(selected_by_request):
                break

            input_ids = torch.full((max_batch, max_beams), pad_token_id, dtype=torch.long, device=device)
            position_ids = torch.full((max_batch, max_beams), self.max_len - 1, dtype=torch.long, device=device)
            cache_seq_indices = torch.full((max_batch, max_beams), self.max_len - 1, dtype=torch.long, device=device)
            attention_mask = torch.zeros((max_batch, 1, max_beams, self.max_len), dtype=dtype, device=device)

            for request_index, selected_sources in enumerate(selected_by_request):
                if not selected_sources:
                    continue
                request = normalized[request_index]
                batch_idx = int(request["batch_index"])
                prefix_len = int(request["prefix_len"])
                tree_forward_nodes = [
                    TreeForwardNode(
                        node_id=int(tree_node.node_id),
                        parent_id=None if tree_node.parent_id is None else int(tree_node.parent_id),
                        token_id=int(tree_node.token_id),
                        depth=int(tree_node.depth),
                    )
                    for tree_node in request["nodes"]
                ]
                for beam_index, source in enumerate(selected_sources):
                    input_ids[batch_idx, beam_index] = int(source["token_id"])
                    if source["kind"] == "prefix_tail":
                        position_ids[batch_idx, beam_index] = prefix_len - 1
                        cache_seq_indices[batch_idx, beam_index] = prefix_len - 1
                        attention_mask[batch_idx, 0, beam_index, :prefix_len] = 1
                    else:
                        node = source["node"]
                        node_index = _node_index(request["nodes"], int(node.node_id))
                        position_ids[batch_idx, beam_index] = prefix_len + int(node.depth) - 1
                        cache_seq_indices[batch_idx, beam_index] = prefix_len + node_index
                        _fill_tree_node_allow_mask(
                            attention_mask[batch_idx, 0, beam_index],
                            prefix_len=prefix_len,
                            node=TreeForwardNode(
                                node_id=int(node.node_id),
                                parent_id=None if node.parent_id is None else int(node.parent_id),
                                token_id=int(node.token_id),
                                depth=int(node.depth),
                            ),
                            node_index=node_index,
                            nodes=tree_forward_nodes,
                        )

            cache_batch_indices = torch.arange(max_batch, dtype=torch.long, device=device).repeat_interleave(max_beams)
            forward_start_ns = perf_counter_ns()
            logits = self.batch_engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices.flatten(),
                attention_mask=attention_mask,
            )
            forward_end_ns = perf_counter_ns()
            shared_batch_event_id = f"draft_batch_graph_official_grow:{forward_start_ns}:{grow_step}:{len(normalized)}"

            for request_index, selected_sources in enumerate(selected_by_request):
                if not selected_sources:
                    continue
                request = normalized[request_index]
                batch_idx = int(request["batch_index"])
                max_nodes = int(request["max_nodes"])
                max_branches = int(request["max_branches"])
                incoming_children: list[dict[str, Any]] = []
                for beam_index, source in enumerate(selected_sources):
                    topk = _topk_tokens(torch, logits[batch_idx, beam_index], max_branches)
                    parent_node = source.get("node")
                    parent_node_id = None if parent_node is None else int(parent_node.node_id)
                    parent_depth = 0 if parent_node is None else int(parent_node.depth)
                    parent_logprob = 0.0 if parent_node is None else float(parent_node.draft_logprob or 0.0)
                    detail_events_by_request[request_index].append(
                        {
                            "phase": "draft.batch_graph_official_grow",
                            "start_ns": forward_start_ns,
                            "end_ns": forward_end_ns,
                            "duration_ms": _duration_ms(forward_start_ns, forward_end_ns) / max(1, len(normalized)),
                            "shared_duration_ms": _duration_ms(forward_start_ns, forward_end_ns),
                            "batch_size": len(normalized),
                            "batch_index": batch_idx,
                            "request_order_index": int(request["request_order_index"]),
                            "beam_index": beam_index,
                            "shared_batch_event_id": shared_batch_event_id,
                            "topk_batch_kind": "qwen3_batch_graph_official_grow",
                            "grow_step": grow_step,
                            "source_kind": str(source["kind"]),
                            "parent_node_id": parent_node_id,
                            "prefix_len": int(request["prefix_len"]) + parent_depth,
                            "official_decay_factor": decay_factor,
                            "official_score_floor": score_floor,
                            "official_persistent_kv_reused": True,
                            "explicit_kv_cache": True,
                            "cuda_graph": True,
                            "topk": [candidate.__dict__ for candidate in topk],
                        }
                    )
                    for candidate in topk:
                        incoming_children.append(
                            {
                                "parent_id": parent_node_id,
                                "token_id": int(candidate.token_id),
                                "depth": parent_depth + 1,
                                "draft_logprob": parent_logprob + float(decay_factor) + float(candidate.logprob),
                                "draft_worker_id": request["runner_id"],
                            }
                        )
                kept_children = _official_budget_children(
                    request["nodes"],
                    incoming_children,
                    max_nodes=max_nodes,
                    score_floor=score_floor,
                )
                for child in kept_children:
                    if int(request["prefix_len"]) + len(request["nodes"]) >= int(self.max_len):
                        break
                    node = CandidateNode(
                        node_id=int(request["next_node_id"]),
                        parent_id=None if child["parent_id"] is None else int(child["parent_id"]),
                        token_id=int(child["token_id"]),
                        depth=int(child["depth"]),
                        draft_logprob=float(child["draft_logprob"]),
                        draft_worker_id=str(child["draft_worker_id"]),
                    )
                    request["nodes"].append(node)
                    request["statuses"][int(node.node_id)] = 15
                    request["next_node_id"] = int(request["next_node_id"]) + 1

                if len(request["nodes"]) > max_nodes:
                    old_nodes = list(request["nodes"])
                    trimmed = _trim_nodes_with_ancestors(request["nodes"], max_nodes=max_nodes)
                    request_index_for_events = int(request["request_order_index"])
                    _gather_batch_tree_after_trim(
                        torch,
                        batch_engine=self.batch_engine,
                        batch_idx=int(request["batch_index"]),
                        prefix_len=int(request["prefix_len"]),
                        old_nodes=old_nodes,
                        new_nodes=trimmed,
                        device=device,
                        detail_events=detail_events_by_request[request_index_for_events],
                        phase="draft.batch_graph_official_grow_budget_gather",
                    )
                    kept_ids = {int(node.node_id) for node in trimmed}
                    request["nodes"] = trimmed
                    request["statuses"] = {
                        int(node_id): int(status)
                        for node_id, status in request["statuses"].items()
                        if int(node_id) in kept_ids
                    }

        outputs: list[dict[str, Any]] = []
        for request in normalized:
            max_nodes = int(request["max_nodes"])
            old_nodes = list(request["nodes"])
            nodes = _trim_nodes_with_ancestors(request["nodes"], max_nodes=max_nodes)
            request_index = int(request["request_order_index"])
            budget_gathered = _gather_batch_tree_after_trim(
                torch,
                batch_engine=self.batch_engine,
                batch_idx=int(request["batch_index"]),
                prefix_len=int(request["prefix_len"]),
                old_nodes=old_nodes,
                new_nodes=nodes,
                device=device,
                detail_events=detail_events_by_request[request_index],
                phase="draft.batch_graph_official_grow_budget_gather",
            )
            kept_ids = {int(node.node_id) for node in nodes}
            statuses = {
                int(node_id): int(status)
                for node_id, status in request["statuses"].items()
                if int(node_id) in kept_ids
            }
            tree = CandidateTree(root_prefix_len=int(request["prefix_len"]), nodes=nodes)
            tree.validate()
            outputs.append(
                {
                    "tree": tree,
                    "metadata": {
                        **dict(request["metadata"]),
                        "graph_tree_draft": True,
                        "official_batch_tree_draft": True,
                        "official_state_grow": True,
                        "official_state_grow_fallback": False,
                        "official_persistent_kv_reused": True,
                        "official_needs_prefix_tail_forward": bool(request["needs_prefix_tail_forward"]),
                        "tree_draft_backend": "qwen3_batch_graph_official_grow",
                        "draft_token_forward_events": detail_events_by_request[request_index],
                        "tree_node_statuses": {str(node_id): status for node_id, status in statuses.items()},
                        "max_graph_tokens": self.max_graph_tokens,
                        "max_graph_batch_size": self.max_graph_batch_size,
                        "official_budget_pruning": True,
                        "official_decay_factor": decay_factor,
                        "official_score_floor": score_floor,
                        "official_budget_kv_gather": bool(budget_gathered),
                        "explicit_kv_cache": True,
                        "cuda_graph": True,
                        "batch_index": int(request["batch_index"]),
                        "official_draft_batch_index": int(request["batch_index"]),
                        "official_draft_request_id": request["request_id"],
                        "batch_size": len(normalized),
                    },
                }
            )
        return outputs

    def generate_official_proactive_graph(self, request: dict[str, Any]) -> dict[str, Any] | None:
        """Generate official proactive POST_* nodes in-place on the draft batch KV row."""
        import torch

        if self.batch_engine is None:
            return None
        prefix_ids = [int(token_id) for token_id in request.get("prefix_ids", [])]
        if not prefix_ids:
            raise ValueError("qwen3_graph official proactive requires a non-empty prefix.")
        tree_payload = request.get("tree")
        if isinstance(tree_payload, CandidateTree):
            tree = tree_payload
        elif tree_payload:
            tree = CandidateTree.from_dict(dict(tree_payload))
        else:
            return None
        if tree.root_prefix_len != len(prefix_ids):
            raise ValueError("official proactive tree root_prefix_len must match prefix_ids length.")
        tree.validate()
        if not tree.nodes:
            return None

        metadata = dict(request.get("metadata") or {})
        batch_index = request.get("draft_batch_index")
        if batch_index is None:
            batch_index = metadata.get("official_draft_batch_index", metadata.get("batch_index"))
        if batch_index is None and request.get("request_id") is not None:
            batch_index = self._official_draft_batch_indices.get(str(request["request_id"]))
        if batch_index is None:
            return {
                "metadata": {
                    **metadata,
                    "official_proactive_graph": False,
                    "official_persistent_kv_reused": False,
                    "official_persistent_kv_reuse_reason": "missing_draft_batch_index",
                }
            }
        batch_idx = int(batch_index)
        if batch_idx < 0 or batch_idx >= int(self.max_graph_batch_size):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph official proactive batch index {batch_idx} exceeds captured batch size {self.max_graph_batch_size}"
            )
        if request.get("request_id") is not None:
            self._official_draft_batch_indices[str(request["request_id"])] = batch_idx

        max_depth = int(request.get("max_depth", 0))
        max_branches = int(request.get("max_branches", 0))
        max_nodes = int(request.get("max_nodes", 0))
        max_leaf_beams = int(request.get("max_leaf_beams", self.max_graph_tokens))
        root_top_k = int(request.get("root_top_k", 1))
        decay_factor = float(request.get("decay_factor", metadata.get("official_proactive_decay_factor", math.log(0.95))))
        runner_id = str(request.get("runner_id") or self.runner_id)
        if max_nodes <= 0 or max_leaf_beams <= 0 or root_top_k <= 0:
            return None
        if max_branches > int(self.max_graph_tokens):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph proactive max_branches {max_branches} exceeds captured max {self.max_graph_tokens}"
            )
        if len(prefix_ids) + len(tree.nodes) + max_nodes > int(self.max_len):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph proactive tree length {len(prefix_ids) + len(tree.nodes) + max_nodes} exceeds cache max {self.max_len}"
            )

        nodes = [
            CandidateNode(
                node_id=int(node.node_id),
                parent_id=None if node.parent_id is None else int(node.parent_id),
                token_id=int(node.token_id),
                depth=int(node.depth),
                draft_logprob=node.draft_logprob,
                draft_worker_id=str(node.draft_worker_id or runner_id),
            )
            for node in tree.nodes
        ]
        statuses = _official_status_map(
            request.get("tree_node_statuses") or metadata.get("tree_node_statuses") or {},
            nodes=nodes,
        )
        children_by_parent = tree.children_by_parent()
        leaves = [node for node in nodes if int(node.node_id) not in children_by_parent]
        if not leaves:
            return None
        leaves.sort(
            key=lambda node: (
                float("-inf") if node.draft_logprob is None else float(node.draft_logprob),
                -int(node.node_id),
            ),
            reverse=True,
        )
        selected_leaves = leaves[: min(max_leaf_beams, int(self.max_graph_tokens))]

        pad_token_id = _pad_token_id(self.tokenizer, self.model)
        max_batch = int(self.max_graph_batch_size)
        max_beams = int(self.max_graph_tokens)
        dtype = self.model.dtype
        device = self.device
        detail_events: list[dict[str, Any]] = []

        input_ids = torch.full((max_batch, max_beams), pad_token_id, dtype=torch.long, device=device)
        position_ids = torch.full((max_batch, max_beams), self.max_len - 1, dtype=torch.long, device=device)
        cache_seq_indices = torch.full((max_batch, max_beams), self.max_len - 1, dtype=torch.long, device=device)
        attention_mask = torch.zeros((max_batch, 1, max_beams, self.max_len), dtype=dtype, device=device)
        tree_forward_nodes = [
            TreeForwardNode(
                node_id=int(node.node_id),
                parent_id=None if node.parent_id is None else int(node.parent_id),
                token_id=int(node.token_id),
                depth=int(node.depth),
            )
            for node in nodes
        ]
        prefix_len = len(prefix_ids)
        for beam_index, node in enumerate(selected_leaves):
            node_index = _node_index(nodes, int(node.node_id))
            input_ids[batch_idx, beam_index] = int(node.token_id)
            position_ids[batch_idx, beam_index] = prefix_len + int(node.depth) - 1
            cache_seq_indices[batch_idx, beam_index] = prefix_len + node_index
            _fill_tree_node_allow_mask(
                attention_mask[batch_idx, 0, beam_index],
                prefix_len=prefix_len,
                node=TreeForwardNode(
                    node_id=int(node.node_id),
                    parent_id=None if node.parent_id is None else int(node.parent_id),
                    token_id=int(node.token_id),
                    depth=int(node.depth),
                ),
                node_index=node_index,
                nodes=tree_forward_nodes,
            )

        cache_batch_indices = torch.arange(max_batch, dtype=torch.long, device=device).repeat_interleave(max_beams)
        select_start_ns = perf_counter_ns()
        logits = self.batch_engine.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices.flatten(),
            attention_mask=attention_mask,
        )
        select_end_ns = perf_counter_ns()
        best: tuple[float, int, int, CandidateNode, int, list[int]] | None = None
        nodes_by_id = {int(node.node_id): node for node in nodes}
        for leaf_order, leaf in enumerate(selected_leaves):
            leaf_score = float(leaf.draft_logprob or 0.0)
            leaf_path = _candidate_path_tokens(leaf, nodes_by_id)
            topk = _topk_tokens(torch, logits[batch_idx, leaf_order], root_top_k)
            detail_events.append(
                {
                    "phase": "draft.official_proactive_root",
                    "start_ns": select_start_ns,
                    "end_ns": select_end_ns,
                    "duration_ms": _duration_ms(select_start_ns, select_end_ns) / max(1, len(selected_leaves)),
                    "shared_duration_ms": _duration_ms(select_start_ns, select_end_ns),
                    "batch_index": batch_idx,
                    "leaf_node_id": int(leaf.node_id),
                    "leaf_path": list(leaf_path),
                    "topk_batch_kind": "qwen3_batch_graph_official_proactive_root",
                    "official_persistent_kv_reused": True,
                    "explicit_kv_cache": True,
                    "cuda_graph": True,
                    "topk": [candidate.__dict__ for candidate in topk],
                }
            )
            for candidate in topk:
                score = leaf_score + float(decay_factor) + float(candidate.logprob)
                key = (score, -leaf_order, -int(candidate.rank), leaf, int(candidate.token_id), leaf_path)
                if best is None or key[:3] > best[:3]:
                    best = key
        if best is None:
            return None

        root_logprob, _leaf_order, _rank_order, parent_node, root_token_id, leaf_path = best
        prompt_len = int(request.get("prompt_len", len(prefix_ids)))
        max_new_tokens = int(request.get("max_new_tokens", 10**9))
        prospective_new_tokens = (len(prefix_ids) + len(leaf_path) + 1) - prompt_len
        max_depth = min(max_depth, max(0, max_new_tokens - int(prospective_new_tokens)))
        next_node_id = max((int(node.node_id) for node in nodes), default=-1) + 1
        root_node = CandidateNode(
            node_id=next_node_id,
            parent_id=int(parent_node.node_id),
            token_id=int(root_token_id),
            depth=int(parent_node.depth) + 1,
            draft_logprob=float(root_logprob),
            draft_worker_id=runner_id,
        )
        nodes.append(root_node)
        statuses[int(root_node.node_id)] = 20
        proactive_ids: list[int] = [int(root_node.node_id)]
        next_node_id += 1

        for grow_step in range(max(0, max_depth)):
            if max_branches <= 0:
                break
            candidates = [
                node
                for node in nodes
                if int(node.node_id) in proactive_ids and int(statuses.get(int(node.node_id), 15)) == 20
            ]
            candidates.sort(
                key=lambda node: (
                    float("-inf") if node.draft_logprob is None else float(node.draft_logprob),
                    -int(node.node_id),
                ),
                reverse=True,
            )
            selected = candidates[:max_beams]
            if not selected:
                break

            input_ids = torch.full((max_batch, max_beams), pad_token_id, dtype=torch.long, device=device)
            position_ids = torch.full((max_batch, max_beams), self.max_len - 1, dtype=torch.long, device=device)
            cache_seq_indices = torch.full((max_batch, max_beams), self.max_len - 1, dtype=torch.long, device=device)
            attention_mask = torch.zeros((max_batch, 1, max_beams, self.max_len), dtype=dtype, device=device)
            tree_forward_nodes = [
                TreeForwardNode(
                    node_id=int(node.node_id),
                    parent_id=None if node.parent_id is None else int(node.parent_id),
                    token_id=int(node.token_id),
                    depth=int(node.depth),
                )
                for node in nodes
            ]
            for beam_index, node in enumerate(selected):
                node_index = _node_index(nodes, int(node.node_id))
                input_ids[batch_idx, beam_index] = int(node.token_id)
                position_ids[batch_idx, beam_index] = prefix_len + int(node.depth) - 1
                cache_seq_indices[batch_idx, beam_index] = prefix_len + node_index
                _fill_tree_node_allow_mask(
                    attention_mask[batch_idx, 0, beam_index],
                    prefix_len=prefix_len,
                    node=TreeForwardNode(
                        node_id=int(node.node_id),
                        parent_id=None if node.parent_id is None else int(node.parent_id),
                        token_id=int(node.token_id),
                        depth=int(node.depth),
                    ),
                    node_index=node_index,
                    nodes=tree_forward_nodes,
                )

            forward_start_ns = perf_counter_ns()
            logits = self.batch_engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices.flatten(),
                attention_mask=attention_mask,
            )
            forward_end_ns = perf_counter_ns()
            incoming_children: list[dict[str, Any]] = []
            for beam_index, parent in enumerate(selected):
                statuses[int(parent.node_id)] = 25
                topk = _topk_tokens(torch, logits[batch_idx, beam_index], max_branches)
                detail_events.append(
                    {
                        "phase": "draft.official_proactive_grow",
                        "start_ns": forward_start_ns,
                        "end_ns": forward_end_ns,
                        "duration_ms": _duration_ms(forward_start_ns, forward_end_ns) / max(1, len(selected)),
                        "shared_duration_ms": _duration_ms(forward_start_ns, forward_end_ns),
                        "batch_index": batch_idx,
                        "beam_index": beam_index,
                        "grow_step": grow_step,
                        "parent_node_id": int(parent.node_id),
                        "topk_batch_kind": "qwen3_batch_graph_official_proactive_grow",
                        "official_persistent_kv_reused": True,
                        "explicit_kv_cache": True,
                        "cuda_graph": True,
                        "topk": [candidate.__dict__ for candidate in topk],
                    }
                )
                for candidate in topk:
                    incoming_children.append(
                        {
                            "parent_id": int(parent.node_id),
                            "token_id": int(candidate.token_id),
                            "depth": int(parent.depth) + 1,
                            "draft_logprob": float(parent.draft_logprob or 0.0)
                            + float(decay_factor)
                            + float(candidate.logprob),
                            "draft_worker_id": runner_id,
                        }
                    )
            existing_proactive_nodes = [node for node in nodes if int(node.node_id) in proactive_ids]
            kept_children = _official_budget_children(
                existing_proactive_nodes,
                incoming_children,
                max_nodes=max_nodes,
                score_floor=float("-inf"),
            )
            for child_spec in kept_children:
                if len(prefix_ids) + len(nodes) >= int(self.max_len):
                    raise Qwen3GraphBackendUnavailable(
                        f"qwen3_graph proactive tree length exceeds cache max {self.max_len}"
                    )
                child = CandidateNode(
                    node_id=next_node_id,
                    parent_id=int(child_spec["parent_id"]),
                    token_id=int(child_spec["token_id"]),
                    depth=int(child_spec["depth"]),
                    draft_logprob=float(child_spec["draft_logprob"]),
                    draft_worker_id=str(child_spec["draft_worker_id"]),
                )
                nodes.append(child)
                statuses[int(child.node_id)] = 20
                proactive_ids.append(int(child.node_id))
                next_node_id += 1

        descendants = [
            node
            for node in nodes
            if int(node.node_id) in proactive_ids and int(node.node_id) != int(root_node.node_id)
        ]
        id_map = {int(node.node_id): index for index, node in enumerate(descendants)}
        subtree_nodes = [
            CandidateNode(
                node_id=id_map[int(node.node_id)],
                parent_id=None if int(node.parent_id) == int(root_node.node_id) else id_map[int(node.parent_id)],
                token_id=int(node.token_id),
                depth=max(1, int(node.depth) - int(root_node.depth)),
                draft_logprob=node.draft_logprob,
                draft_worker_id=str(node.draft_worker_id),
            )
            for node in descendants
        ]
        subtree = CandidateTree(root_prefix_len=len(prefix_ids) + len(leaf_path) + 1, nodes=subtree_nodes)
        subtree.validate()
        subtree_statuses = {
            str(id_map[int(node.node_id)]): _normal_official_status_int(int(statuses.get(int(node.node_id), 20)))
            for node in descendants
        }
        return {
            "parent_node_id": int(parent_node.node_id),
            "root_token_id": int(root_token_id),
            "root_logprob": float(root_logprob),
            "root_status": int(statuses.get(int(root_node.node_id), 20)),
            "leaf_path": list(leaf_path),
            "subtree": subtree,
            "subtree_statuses": subtree_statuses,
            "metadata": {
                **metadata,
                "official_proactive_graph": True,
                "official_persistent_kv_reused": True,
                "official_proactive_parent_node_id": int(parent_node.node_id),
                "official_proactive_root_token_id": int(root_token_id),
                "official_proactive_root_status": int(statuses.get(int(root_node.node_id), 20)),
                "official_proactive_leaf_path": list(leaf_path),
                "official_proactive_budget_pruning": True,
                "draft_token_forward_events": detail_events,
                "tree_draft_backend": "qwen3_batch_graph_official_proactive",
                "official_draft_batch_index": batch_idx,
                "explicit_kv_cache": True,
                "cuda_graph": True,
            },
        }

    def official_specedge_commit_acceptance(
        self,
        *,
        request_id: str,
        batch_index: int | None,
        source_seq_indices: list[int],
        dest_seq_indices: list[int],
        prefix_ids: list[int],
        retained_tree: dict[str, Any] | None = None,
        reused_proactive_tree: bool = False,
    ) -> dict[str, Any]:
        """Apply official accept gather to the draft BatchGraphEngine KV cache."""
        if self.batch_engine is None:
            return {"official_draft_kv_gather": False, "reason": "no_batch_engine"}
        if batch_index is None:
            batch_index = self._official_draft_batch_indices.get(str(request_id))
        if batch_index is None:
            return {"official_draft_kv_gather": False, "reason": "unknown_batch_index"}
        if not source_seq_indices:
            return {"official_draft_kv_gather": False, "reason": "empty_source_indices"}
        if len(source_seq_indices) != len(dest_seq_indices):
            raise ValueError("official_specedge_commit_acceptance requires matching source/dest index lengths.")
        import torch

        src = torch.tensor([int(index) for index in source_seq_indices], dtype=torch.long, device=self.device)
        dst = torch.tensor([int(index) for index in dest_seq_indices], dtype=torch.long, device=self.device)
        self.batch_engine.gather(int(batch_index), src, dst)
        self._official_draft_batch_indices[str(request_id)] = int(batch_index)
        return {
            "official_draft_kv_gather": True,
            "request_id": str(request_id),
            "batch_index": int(batch_index),
            "source_seq_indices": [int(index) for index in source_seq_indices],
            "dest_seq_indices": [int(index) for index in dest_seq_indices],
            "prefix_len": len(prefix_ids),
            "retained_tree_node_count": 0 if retained_tree is None else len(retained_tree.get("nodes", [])),
            "reused_proactive_tree": bool(reused_proactive_tree),
        }

    def linear_verify_batch(self, requests: list[LinearForwardInput]) -> list[LinearForwardOutput]:
        """Verify linear drafts with the packed tree-attention graph path when possible."""
        if not requests:
            return []
        if any(len(request.draft_tokens) > int(self.max_graph_tokens) for request in requests):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph verify token count exceeds captured max {self.max_graph_tokens}"
            )
        if self.batch_engine is not None:
            return self._linear_verify_batch_tree_attention(requests)
        return [
            self._linear_verify_graph(request, batch_index=index, batch_size=len(requests))
            for index, request in enumerate(requests)
        ]

    def reset(self, request_id: str | None = None) -> None:
        del request_id
        self.engine.reset()
        if self.batch_engine is not None:
            self.batch_engine.reset()

    def backend_capabilities(self) -> ModelBackendCapabilities:
        return ModelBackendCapabilities(
            backend_name=self.backend_name,
            backend_fallback=self.backend_fallback,
            fallback_reason=self.fallback_reason,
            supports_topk=True,
            supports_batched_topk=False,
            supports_batched_next_token=False,
            supports_linear_verify_batch=self.batch_engine is not None,
            supports_tree_attention=self.batch_engine is not None,
            supports_tree_forward_batch=self.batch_engine is not None,
            supports_kv_cache=True,
            supports_cuda_graph=True,
        )

    def tree_forward(self, request: TreeForwardInput) -> TreeForwardOutput:
        """Verify one tree with the same batched graph path used by server batches."""
        return self.tree_forward_batch([request])[0]

    def tree_forward_batch(self, requests: list[TreeForwardInput]) -> list[TreeForwardOutput]:
        """Run official-style tree attention verification with BatchGraphEngine.

        The prefix for each request is prefetched into the explicit KV cache.
        Then all tree nodes are packed into a fixed-shape graph batch. Root
        choices come from the prefill logits; non-root choices come from the
        corresponding parent node's graph output.
        """
        if not requests:
            return []
        if len(requests) > int(self.max_graph_batch_size):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph tree batch size {len(requests)} exceeds captured max {self.max_graph_batch_size}"
            )
        if self.batch_engine is None:
            raise Qwen3GraphBackendUnavailable("qwen3_graph tree verification requires BatchGraphEngine")
        for request in requests:
            if not request.prefix_ids:
                raise ValueError("qwen3_graph tree_forward_batch requires non-empty prefixes.")
            _validate_tree_forward_nodes(request.nodes)
            if len(request.prefix_ids) + len(request.nodes) > int(self.max_len):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph tree length {len(request.prefix_ids) + len(request.nodes)} exceeds cache max {self.max_len}"
                )
            if len(request.nodes) > int(self.max_graph_tokens):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph tree node count {len(request.nodes)} exceeds captured max {self.max_graph_tokens}"
                )
        return self._tree_forward_batch_graph(requests)

    def _linear_verify_graph(
        self,
        request: LinearForwardInput,
        *,
        batch_index: int,
        batch_size: int,
    ) -> LinearForwardOutput:
        if not request.prefix_ids:
            raise ValueError("qwen3_graph linear verification requires a non-empty prefix.")
        draft_tokens = [int(token_id) for token_id in request.draft_tokens]
        if len(draft_tokens) > int(self.max_graph_tokens):
            raise Qwen3GraphBackendUnavailable(
                f"qwen3_graph verify token count {len(draft_tokens)} exceeds captured max {self.max_graph_tokens}"
            )

        working_prefix = [int(token_id) for token_id in request.prefix_ids]
        draft_target_tokens: list[int] = []
        matched_all = True
        target_forward_call_count = 0
        for draft_token in draft_tokens:
            target_token = self._greedy_next_token_graph(working_prefix)
            target_forward_call_count += 1
            draft_target_tokens.append(target_token)
            if int(target_token) != int(draft_token):
                matched_all = False
                break
            working_prefix.append(int(draft_token))
        bonus_token = None
        if request.allow_bonus and matched_all:
            bonus_token = self._greedy_next_token_graph(working_prefix)
            target_forward_call_count += 1
        return LinearForwardOutput(
            draft_target_tokens=draft_target_tokens,
            bonus_token=bonus_token,
            metadata={
                "linear_forward_kind": "linear_prefix_step_qwen3_graph",
                "linear_forward_batch_kind": "linear_prefix_step_qwen3_graph",
                "batch_index": batch_index,
                "batch_size": batch_size,
                "draft_token_count": len(draft_tokens),
                "graph_verify_token_count": len(draft_target_tokens),
                "graph_output_token_count": len(draft_target_tokens) + (1 if bonus_token is not None else 0),
                "graph_prefill_token_count": len(request.prefix_ids),
                "target_forward_call_count": target_forward_call_count,
                "single_pass_linear_verify": False,
                "causal_safe_prefix_batch": True,
                "explicit_kv_cache": True,
                "cuda_graph": True,
                "max_graph_tokens": self.max_graph_tokens,
            },
        )

    def _linear_verify_batch_tree_attention(self, requests: list[LinearForwardInput]) -> list[LinearForwardOutput]:
        """Map each linear draft to a chain tree and verify the batch in one graph step."""
        shared_forward_id = f"linear_tree_attention_qwen3_graph:{perf_counter_ns()}:{len(requests)}"
        tree_inputs = [_linear_request_to_tree_input(request) for request in requests]
        tree_outputs = self.tree_forward_batch(tree_inputs)
        outputs: list[LinearForwardOutput] = []
        for index, (request, tree_output) in enumerate(zip(requests, tree_outputs)):
            output = _linear_output_from_tree_output(
                request,
                tree_output,
                batch_index=index,
                batch_size=len(requests),
                shared_forward_id=shared_forward_id,
            )
            outputs.append(output)
        return outputs

    def _greedy_next_token_graph(self, prefix_ids: list[int]) -> int:
        import torch

        logits = self._verify_path_logits(prefix_ids=[int(token_id) for token_id in prefix_ids], draft_tokens=[])[0, -1]
        return int(torch.argmax(logits.detach().float()).item())

    def _tree_forward_batch_graph(self, requests: list[TreeForwardInput]) -> list[TreeForwardOutput]:
        import torch

        pad_token_id = _pad_token_id(self.tokenizer, self.model)
        max_batch = int(self.max_graph_batch_size)
        max_nodes = int(self.max_graph_tokens)
        batch_size = len(requests)
        dtype = self.model.dtype
        device = self.device

        self.batch_engine.reset()
        prefix_last_logits: list[Any] = []
        for row_index, request in enumerate(requests):
            prefix_ids = [int(token_id) for token_id in request.prefix_ids]
            prefix_len = len(prefix_ids)
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=device)
            position_ids = torch.arange(prefix_len, dtype=torch.long, device=device).unsqueeze(0)
            cache_seq_indices = torch.arange(prefix_len, dtype=torch.long, device=device)
            attention_mask = _causal_allow_mask(
                torch,
                query_seq_indices=cache_seq_indices,
                max_len=self.max_len,
                dtype=dtype,
                device=device,
            )
            logits = self.batch_engine.prefill(
                input_ids=input_ids,
                position_ids=position_ids,
                batch_idx=row_index,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )
            prefix_last_logits.append(logits[0, -1].detach())

        input_ids = torch.full((max_batch, max_nodes), pad_token_id, dtype=torch.long, device=device)
        position_ids = torch.full((max_batch, max_nodes), self.max_len - 1, dtype=torch.long, device=device)
        cache_seq_indices = torch.full((max_batch, max_nodes), self.max_len - 1, dtype=torch.long, device=device)
        attention_mask = torch.zeros((max_batch, 1, max_nodes, self.max_len), dtype=dtype, device=device)
        attention_mask[:, :, :, 0] = 1

        for row_index, request in enumerate(requests):
            prefix_len = len(request.prefix_ids)
            for node_index, node in enumerate(request.nodes):
                cache_index = prefix_len + node_index
                input_ids[row_index, node_index] = int(node.token_id)
                position_ids[row_index, node_index] = prefix_len + int(node.depth) - 1
                cache_seq_indices[row_index, node_index] = cache_index
                _fill_tree_node_allow_mask(
                    attention_mask[row_index, 0, node_index],
                    prefix_len=prefix_len,
                    node=node,
                    node_index=node_index,
                    nodes=request.nodes,
                )

        cache_batch_indices = torch.arange(max_batch, dtype=torch.long, device=device).repeat_interleave(max_nodes)
        graph_logits = self.batch_engine.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            cache_batch_indices=cache_batch_indices,
            cache_seq_indices=cache_seq_indices.flatten(),
            attention_mask=attention_mask,
        )

        outputs: list[TreeForwardOutput] = []
        for row_index, request in enumerate(requests):
            parent_ids = _tree_parent_ids_with_extra(request.nodes, request.metadata)
            node_index_by_id = {int(node.node_id): index for index, node in enumerate(request.nodes)}
            choices: list[TreeForwardChoice] = []
            for parent_id in parent_ids:
                logits = (
                    prefix_last_logits[row_index]
                    if parent_id is None
                    else graph_logits[row_index, node_index_by_id[int(parent_id)]].detach()
                )
                choices.append(
                    TreeForwardChoice(
                        parent_node_id=parent_id,
                        target_token_id=int(torch.argmax(logits.detach().float()).item()),
                        prefix_len=_tree_parent_prefix_len(len(request.prefix_ids), request.nodes, parent_id),
                    )
                )
            outputs.append(
                TreeForwardOutput(
                    choices=choices,
                    metadata={
                        "tree_forward_kind": "tree_attention_qwen3_graph",
                        "tree_forward_batch_kind": "tree_attention_batch_qwen3_graph",
                        "batch_index": row_index,
                        "batch_size": batch_size,
                        "active_batch_size": batch_size,
                        "node_count": len(request.nodes),
                        "choice_count": len(parent_ids),
                        "padded_token_count": max_nodes,
                        "graph_prefill_token_count": len(request.prefix_ids),
                        "graph_verify_token_count": len(request.nodes),
                        "target_forward_call_count": 1,
                        "single_pass_tree_verify": True,
                        "explicit_kv_cache": True,
                        "cuda_graph": True,
                        "max_graph_tokens": self.max_graph_tokens,
                        "max_graph_batch_size": self.max_graph_batch_size,
                    },
                )
            )
        return outputs

    def _verify_path_logits(self, *, prefix_ids: list[int], draft_tokens: list[int]) -> Any:
        """Return logits for predictions after prefix and each draft token."""
        import torch

        with self._graph_lock:
            if not prefix_ids:
                raise ValueError("qwen3_graph verification requires a non-empty prefix.")
            draft_tokens = [int(token_id) for token_id in draft_tokens]
            graph_tail_tokens = list(draft_tokens)
            if len(prefix_ids) + len(graph_tail_tokens) > int(self.max_len):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph verify length {len(prefix_ids) + len(graph_tail_tokens)} exceeds cache max {self.max_len}"
                )
            if len(graph_tail_tokens) > int(self.max_graph_tokens):
                raise Qwen3GraphBackendUnavailable(
                    f"qwen3_graph verify token count {len(graph_tail_tokens)} exceeds captured max {self.max_graph_tokens}"
                )

            self.engine.reset()
            prefill_tokens = [int(token_id) for token_id in prefix_ids]
            prefill_len = len(prefill_tokens)
            input_ids = torch.tensor([prefill_tokens], dtype=torch.long, device=self.device)
            position_ids = torch.arange(prefill_len, dtype=torch.long, device=self.device).unsqueeze(0)
            cache_seq_indices = torch.arange(prefill_len, dtype=torch.long, device=self.device)
            attention_mask = _causal_allow_mask(
                torch,
                query_seq_indices=cache_seq_indices,
                max_len=self.max_len,
                dtype=self.model.dtype,
                device=self.device,
            )
            prefix_logits = self.engine.prefill(
                input_ids=input_ids,
                position_ids=position_ids,
                batch_idx=0,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )
            first_prediction_logits = prefix_logits[:, -1:, :]
            if not graph_tail_tokens:
                return first_prediction_logits

            start_pos = prefill_len
            input_ids = torch.tensor([graph_tail_tokens], dtype=torch.long, device=self.device)
            position_ids = torch.arange(
                start_pos,
                start_pos + len(graph_tail_tokens),
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0)
            cache_batch_indices = torch.zeros((len(graph_tail_tokens),), dtype=torch.long, device=self.device)
            cache_seq_indices = torch.arange(
                start_pos,
                start_pos + len(graph_tail_tokens),
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = _causal_allow_mask(
                torch,
                query_seq_indices=cache_seq_indices,
                max_len=self.max_len,
                dtype=self.model.dtype,
                device=self.device,
            )
            tail_logits = self.engine.forward(
                input_ids=input_ids,
                position_ids=position_ids,
                cache_batch_indices=cache_batch_indices,
                cache_seq_indices=cache_seq_indices,
                attention_mask=attention_mask,
            )
            return torch.cat([first_prediction_logits, tail_logits], dim=1)


def load_qwen3_graph_or_fallback(
    model_path: str,
    *,
    runner_id: str,
    device: str | None = "cuda",
    torch_dtype: str | None = "auto",
    device_map: str | None = None,
    attn_implementation: str | None = None,
    trust_remote_code: bool = True,
    allow_fallback: bool = True,
    max_graph_len: int | None = None,
    max_graph_tokens: int | None = None,
    max_graph_batch_size: int | None = None,
) -> CausalLMRunner:
    """Load a Qwen3 graph backend, or a cached fallback with explicit metadata.

    When graph loading fails and `allow_fallback` is true, returns the explicit
    HF cached fallback with visible fallback metadata. Set `allow_fallback=False`
    for paper-facing experiments so graph setup errors fail fast.
    """
    reason = _graph_unavailable_reason(model_path)
    if reason is not None:
        if not allow_fallback:
            raise Qwen3GraphBackendUnavailable(reason)
        runner = CachedTransformersCausalLMRunner.from_pretrained(
            model_path,
            runner_id=runner_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        return replace(
            runner,
            backend_name="qwen3_graph_fallback_hf_cached",
            backend_fallback=True,
            fallback_reason=reason,
        )
    del attn_implementation
    try:
        return Qwen3GraphCausalLMRunner.from_pretrained(
            model_path,
            runner_id=runner_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            max_graph_len=max_graph_len,
            max_graph_tokens=max_graph_tokens,
            max_graph_batch_size=max_graph_batch_size,
        )
    except Exception as exc:
        if not allow_fallback:
            raise
        runner = CachedTransformersCausalLMRunner.from_pretrained(
            model_path,
            runner_id=runner_id,
            device=device,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        return replace(
            runner,
            backend_name="qwen3_graph_fallback_hf_cached",
            backend_fallback=True,
            fallback_reason=f"qwen3_graph backend load failed: {exc}",
        )


def qwen3_graph_fallback_capabilities(reason: str) -> ModelBackendCapabilities:
    """Return capabilities used by tests and dry-run config validation."""
    return ModelBackendCapabilities(
        backend_name="qwen3_graph_fallback_hf_cached",
        backend_fallback=True,
        fallback_reason=reason,
        supports_topk=True,
        supports_batched_topk=False,
        supports_batched_next_token=True,
        supports_linear_verify_batch=True,
        supports_tree_attention=True,
        supports_tree_forward_batch=True,
        supports_kv_cache=True,
        supports_cuda_graph=False,
    )


def _graph_unavailable_reason(model_path: str) -> str | None:
    """Return a concrete reason if graph backend cannot be used."""
    if "qwen3" not in str(model_path).lower():
        return "qwen3_graph backend only supports Qwen3 model paths"
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on environment
        return f"torch import failed: {exc}"
    if not torch.cuda.is_available():
        return "CUDA is not available for qwen3_graph backend"
    # CUDA is present, but custom Qwen3 graph execution is still not wired into
    # specplatform. The loader will attempt graph setup and fallback only if
    # allow_fallback=True.
    return None


def _resolve_torch_dtype(torch: Any, torch_dtype: str | None) -> Any | None:
    """Convert config dtype strings into torch dtype values."""
    if torch_dtype in (None, "auto"):
        return None if torch_dtype is None else "auto"
    if torch_dtype == "bf16":
        return torch.bfloat16
    if torch_dtype == "fp16":
        return torch.float16
    if torch_dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


def _causal_allow_mask(
    torch: Any,
    *,
    query_seq_indices: Any,
    max_len: int,
    dtype: Any,
    device: str,
) -> Any:
    """Build the 1/0 allow mask expected by the vendored GraphEngine."""
    col_indices = torch.arange(int(max_len), dtype=torch.long, device=device)
    allowed = col_indices.unsqueeze(0) <= query_seq_indices.unsqueeze(1)
    return allowed.to(dtype=dtype).unsqueeze(0).unsqueeze(0)


def _pad_token_id(tokenizer: Any, model: Any) -> int:
    """Infer a harmless token id for ignored fixed-shape graph slots."""
    candidates = [
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "eos_token_id", None),
        getattr(getattr(model, "config", None), "pad_token_id", None),
        getattr(getattr(model, "config", None), "eos_token_id", None),
    ]
    for candidate in candidates:
        if isinstance(candidate, (list, tuple)):
            candidate = candidate[0] if candidate else None
        if candidate is not None:
            return int(candidate)
    return 0


def _validate_tree_forward_nodes(nodes: list[TreeForwardNode]) -> None:
    """Validate the parent-before-child topology expected by graph tree verify."""
    seen: set[int] = set()
    for node in nodes:
        node_id = int(node.node_id)
        if node_id in seen:
            raise ValueError(f"Duplicate tree_forward node id: {node_id}")
        if node.parent_id is None:
            if int(node.depth) != 1:
                raise ValueError(f"Root child depth must be 1: {node_id}")
        else:
            parent_id = int(node.parent_id)
            if parent_id not in seen:
                raise ValueError(f"tree_forward parent must appear before child: {node_id}")
        seen.add(node_id)


def _tree_parent_ids(nodes: list[TreeForwardNode]) -> list[int | None]:
    """Return parent ids that need target next-token choices, preserving order."""
    parent_ids: list[int | None] = []
    for node in nodes:
        parent_id = None if node.parent_id is None else int(node.parent_id)
        if parent_id not in parent_ids:
            parent_ids.append(parent_id)
    return parent_ids


def _tree_parent_ids_with_extra(nodes: list[TreeForwardNode], metadata: dict[str, Any]) -> list[int | None]:
    """Return normal tree parent choices plus explicit extra parent choices."""
    parent_ids = _tree_parent_ids(nodes)
    for raw_parent_id in list(dict(metadata or {}).get("extra_choice_parent_ids") or []):
        parent_id = None if raw_parent_id is None else int(raw_parent_id)
        if parent_id not in parent_ids:
            parent_ids.append(parent_id)
    return parent_ids


def _linear_request_to_tree_input(request: LinearForwardInput) -> TreeForwardInput:
    """Represent a linear draft as a single chain tree for graph verification."""
    draft_tokens = [int(token_id) for token_id in request.draft_tokens]
    nodes: list[TreeForwardNode] = []
    parent_id: int | None = None
    for index, token_id in enumerate(draft_tokens):
        node_id = int(index)
        nodes.append(
            TreeForwardNode(
                node_id=node_id,
                parent_id=parent_id,
                token_id=token_id,
                depth=index + 1,
            )
        )
        parent_id = node_id
    extra_choice_parent_ids: list[int | None] = []
    if request.allow_bonus:
        extra_choice_parent_ids.append(None if not nodes else int(nodes[-1].node_id))
    return TreeForwardInput(
        prefix_ids=[int(token_id) for token_id in request.prefix_ids],
        nodes=nodes,
        metadata={
            **dict(request.metadata or {}),
            "linear_chain_tree": True,
            "extra_choice_parent_ids": extra_choice_parent_ids,
        },
    )


def _linear_output_from_tree_output(
    request: LinearForwardInput,
    tree_output: TreeForwardOutput,
    *,
    batch_index: int,
    batch_size: int,
    shared_forward_id: str,
) -> LinearForwardOutput:
    """Convert chain-tree target choices back into a linear verification output."""
    choices_by_parent = {
        (None if choice.parent_node_id is None else int(choice.parent_node_id)): int(choice.target_token_id)
        for choice in tree_output.choices
    }
    draft_tokens = [int(token_id) for token_id in request.draft_tokens]
    draft_target_tokens: list[int] = []
    for index, _draft_token in enumerate(draft_tokens):
        parent_id = None if index == 0 else index - 1
        if parent_id not in choices_by_parent:
            raise ValueError(f"qwen3_graph linear tree output missing parent choice: {parent_id}")
        draft_target_tokens.append(int(choices_by_parent[parent_id]))

    matched_all = all(int(target) == int(draft) for target, draft in zip(draft_target_tokens, draft_tokens))
    bonus_token = None
    if request.allow_bonus and matched_all:
        bonus_parent_id = None if not draft_tokens else len(draft_tokens) - 1
        if bonus_parent_id not in choices_by_parent:
            raise ValueError(f"qwen3_graph linear tree output missing bonus parent choice: {bonus_parent_id}")
        bonus_token = int(choices_by_parent[bonus_parent_id])

    tree_metadata = dict(tree_output.metadata or {})
    return LinearForwardOutput(
        draft_target_tokens=draft_target_tokens,
        bonus_token=bonus_token,
        metadata={
            **tree_metadata,
            "linear_forward_kind": "linear_tree_attention_qwen3_graph",
            "linear_forward_batch_kind": "linear_tree_attention_batch_qwen3_graph",
            "batch_index": int(batch_index),
            "batch_size": int(batch_size),
            "active_batch_size": int(tree_metadata.get("active_batch_size") or batch_size),
            "draft_token_count": len(draft_tokens),
            "graph_verify_token_count": len(draft_tokens),
            "graph_output_token_count": len(draft_target_tokens) + (1 if bonus_token is not None else 0),
            "target_forward_call_count": max(0, int(tree_metadata.get("target_forward_call_count") or 1)),
            "shared_forward_id": shared_forward_id,
            "single_pass_linear_verify": True,
            "causal_safe_prefix_batch": True,
            "explicit_kv_cache": bool(tree_metadata.get("explicit_kv_cache", True)),
            "cuda_graph": bool(tree_metadata.get("cuda_graph", True)),
        },
    )


def _tree_parent_prefix_len(prefix_len: int, nodes: list[TreeForwardNode], parent_id: int | None) -> int:
    """Return token length of the prefix represented by a tree parent node."""
    if parent_id is None:
        return int(prefix_len)
    nodes_by_id = {int(node.node_id): node for node in nodes}
    depth = 0
    current = nodes_by_id[int(parent_id)]
    while current is not None:
        depth += 1
        current = nodes_by_id.get(int(current.parent_id)) if current.parent_id is not None else None
    return int(prefix_len) + depth


def _fill_tree_node_allow_mask(
    row: Any,
    *,
    prefix_len: int,
    node: TreeForwardNode,
    node_index: int,
    nodes: list[TreeForwardNode],
) -> None:
    """Fill one fixed-length allow-mask row for a tree node query."""
    row.zero_()
    row[: int(prefix_len)] = 1
    node_positions = {int(tree_node.node_id): int(prefix_len) + index for index, tree_node in enumerate(nodes)}
    nodes_by_id = {int(tree_node.node_id): tree_node for tree_node in nodes}
    ancestors: list[int] = []
    current = nodes_by_id.get(int(node.parent_id)) if node.parent_id is not None else None
    while current is not None:
        ancestors.append(int(current.node_id))
        current = nodes_by_id.get(int(current.parent_id)) if current.parent_id is not None else None
    for ancestor_id in reversed(ancestors):
        row[node_positions[ancestor_id]] = 1
    row[int(prefix_len) + int(node_index)] = 1


def _topk_tokens(torch: Any, logits: Any, k: int) -> list[TopKToken]:
    """Return sorted log-softmax top-k tokens from a device logits row."""
    if k <= 0:
        return []
    k = min(int(k), int(logits.numel()))
    values, indices = torch.topk(torch.log_softmax(logits.detach().float(), dim=-1), k=k, largest=True, sorted=True)
    return [
        TopKToken(token_id=int(token_id), logprob=float(logprob), rank=rank)
        for rank, (token_id, logprob) in enumerate(zip(indices.detach().cpu().tolist(), values.detach().cpu().tolist()))
    ]


def _node_by_id(nodes: list[CandidateNode], node_id: int) -> CandidateNode:
    for node in nodes:
        if int(node.node_id) == int(node_id):
            return node
    raise KeyError(f"Unknown candidate node id: {node_id}")


def _node_index(nodes: list[CandidateNode], node_id: int) -> int:
    for index, node in enumerate(nodes):
        if int(node.node_id) == int(node_id):
            return index
    raise KeyError(f"Unknown candidate node id: {node_id}")


def _top_candidate_ids(nodes: list[CandidateNode], candidate_ids: list[int], *, max_count: int) -> list[int]:
    """Select official-style highest-logprob candidate beams for one graph step."""
    if max_count <= 0:
        return []
    nodes_by_id = {int(node.node_id): node for node in nodes}
    candidates = [nodes_by_id[int(node_id)] for node_id in candidate_ids if int(node_id) in nodes_by_id]
    candidates.sort(
        key=lambda node: (
            float("-inf") if node.draft_logprob is None else float(node.draft_logprob),
            -int(node.node_id),
        ),
        reverse=True,
    )
    return [int(node.node_id) for node in candidates[: int(max_count)]]


def _official_status_map(raw_statuses: Any, *, nodes: list[CandidateNode]) -> dict[int, int]:
    """Normalize official SpecEdge node statuses, defaulting unknown nodes to CANDIDATE."""
    statuses = {
        int(node_id): int(status)
        for node_id, status in dict(raw_statuses or {}).items()
    }
    for node in nodes:
        statuses.setdefault(int(node.node_id), 15)
    known_ids = {int(node.node_id) for node in nodes}
    return {
        int(node_id): int(status)
        for node_id, status in statuses.items()
        if int(node_id) in known_ids
    }


def _gather_batch_tree_after_trim(
    torch: Any,
    *,
    batch_engine: Any,
    batch_idx: int,
    prefix_len: int,
    old_nodes: list[CandidateNode],
    new_nodes: list[CandidateNode],
    device: str,
    detail_events: list[dict[str, Any]] | None = None,
    phase: str = "draft.batch_graph_budget_gather",
) -> bool:
    """Compact BatchGraphEngine KV positions after tree budget trimming."""
    if not new_nodes:
        return False
    old_ids = [int(node.node_id) for node in old_nodes]
    new_ids = [int(node.node_id) for node in new_nodes]
    if old_ids == new_ids:
        return False
    old_index_by_id = {int(node.node_id): index for index, node in enumerate(old_nodes)}
    source_seq_indices = [int(prefix_len) + old_index_by_id[int(node.node_id)] for node in new_nodes]
    dest_seq_indices = [int(prefix_len) + index for index, _node in enumerate(new_nodes)]
    if source_seq_indices == dest_seq_indices and len(new_nodes) == len(old_nodes):
        return False

    src = torch.tensor(source_seq_indices, dtype=torch.long, device=device)
    dst = torch.tensor(dest_seq_indices, dtype=torch.long, device=device)
    start_ns = perf_counter_ns()
    batch_engine.gather(int(batch_idx), src, dst)
    end_ns = perf_counter_ns()
    if detail_events is not None:
        detail_events.append(
            {
                "phase": phase,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": _duration_ms(start_ns, end_ns),
                "batch_index": int(batch_idx),
                "prefix_len": int(prefix_len),
                "source_seq_indices": [int(index) for index in source_seq_indices],
                "dest_seq_indices": [int(index) for index in dest_seq_indices],
                "old_node_ids": old_ids,
                "new_node_ids": new_ids,
                "official_budget_kv_gather": True,
                "explicit_kv_cache": True,
            }
        )
    return True


def _candidate_path_tokens(node: CandidateNode, nodes_by_id: dict[int, CandidateNode]) -> list[int]:
    path: list[int] = []
    current: CandidateNode | None = node
    while current is not None:
        path.append(int(current.token_id))
        current = nodes_by_id.get(int(current.parent_id)) if current.parent_id is not None else None
    path.reverse()
    return path


def _normal_official_status_int(status: int) -> int:
    if int(status) == 25:
        return 10
    if int(status) == 20:
        return 15
    return int(status)


def _official_budget_children(
    nodes: list[CandidateNode],
    children: list[dict[str, Any]],
    *,
    max_nodes: int,
    score_floor: float = -10.0,
) -> list[dict[str, Any]]:
    """Filter incoming draft children with the official budget-bucket rule.

    Official SpecEdge compares existing tree logprobs and new child logprobs,
    keeps new children above the current top-budget threshold, and later trims
    the tree to the fixed budget.  Ties are intentionally kept here; the final
    closure-preserving trim resolves over-budget cases deterministically.
    """
    if not children:
        return []
    if len(nodes) + len(children) <= int(max_nodes):
        return list(children)
    scores = [
        float("-inf") if node.draft_logprob is None else float(node.draft_logprob)
        for node in nodes
    ]
    scores.extend(float(child["draft_logprob"]) for child in children)
    scores.sort(reverse=True)
    threshold_index = min(max(1, int(max_nodes)), len(scores)) - 1
    threshold = max(float(scores[threshold_index]), float(score_floor))
    return [child for child in children if float(child["draft_logprob"]) >= threshold]


def _trim_nodes_with_ancestors(nodes: list[CandidateNode], *, max_nodes: int) -> list[CandidateNode]:
    """Trim to budget while preserving parent-before-child topology."""
    if len(nodes) <= int(max_nodes):
        return list(nodes)
    nodes_by_id = {int(node.node_id): node for node in nodes}
    ranked = sorted(
        nodes,
        key=lambda node: (
            float("-inf") if node.draft_logprob is None else float(node.draft_logprob),
            -int(node.node_id),
        ),
        reverse=True,
    )
    selected: set[int] = set()
    for node in ranked:
        lineage: list[CandidateNode] = []
        current: CandidateNode | None = node
        while current is not None and int(current.node_id) not in selected:
            lineage.append(current)
            current = nodes_by_id.get(int(current.parent_id)) if current.parent_id is not None else None
        missing = [item for item in reversed(lineage) if int(item.node_id) not in selected]
        if len(selected) + len(missing) > int(max_nodes):
            continue
        selected.update(int(item.node_id) for item in missing)
        if len(selected) >= int(max_nodes):
            break
    return [node for node in nodes if int(node.node_id) in selected]


def _duration_ms(start_ns: int, end_ns: int) -> float:
    return (int(end_ns) - int(start_ns)) / 1_000_000
