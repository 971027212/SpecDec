from __future__ import annotations

"""Hugging Face Transformers causal LM runners.

model 层只适配 encode/decode/next-token 查询，不写 generation loop。draft runner
和 verifier 会复用这个接口，但不会依赖 Transformers 的具体类型。
"""

from dataclasses import dataclass, field
import copy
import math
from typing import Any

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
    TreeForwardOutput,
)


@dataclass
class TransformersCausalLMRunner(CausalLMRunner):
    """把真实 Transformers causal LM 包装成 CausalLMRunner。"""

    runner_id: str
    tokenizer: Any
    model: Any
    max_len: int
    device: str | None = None
    backend_name: str = "hf_eager"
    backend_fallback: bool = False
    fallback_reason: str | None = None

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
        attn_implementation: str | None = None,
    ) -> "TransformersCausalLMRunner":
        """加载本地 Hugging Face 权重。

        这里延迟 import torch/transformers，保证普通单元测试不需要安装大模型依赖。
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        dtype = _resolve_torch_dtype(torch, torch_dtype)
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            model_kwargs["torch_dtype"] = dtype
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if device_map is None and device is not None:
            model = model.to(device)
        model.eval()

        max_len = int(
            getattr(model.config, "max_position_embeddings", None)
            or getattr(model.config, "seq_length", None)
            or 32768
        )
        input_device = _first_parameter_device(model, fallback=device)
        return cls(
            runner_id=runner_id,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            device=input_device,
        )

    def encode(self, text: str) -> list[int]:
        """把文本编码成 token ids；不在这里启动生成循环。"""
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def decode(self, token_ids: list[int]) -> str:
        """把 token ids 解码成文本，用于 smoke 输出检查。"""
        return str(self.tokenizer.decode(token_ids, skip_special_tokens=False))

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        """执行一次模型 forward，返回每个位置的 logits。"""
        import torch

        if not request.input_ids:
            raise ValueError("TransformersCausalLMRunner.forward requires input_ids.")
        input_ids = torch.tensor([request.input_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids)
        logits = output.logits[0].detach().float().cpu().tolist()
        return ModelForwardOutput(logits=logits, metadata={"runner_id": self.runner_id})

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        """取最后一个 prefix 位置的 logits，供 greedy_next_token/verifier 使用。"""
        import torch

        if not prefix_ids:
            raise ValueError("TransformersCausalLMRunner.next_token_logits requires a non-empty prefix.")
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids)
        return output.logits[0, -1].detach().float().cpu().tolist()

    def next_token_logits_batch(self, prefix_ids_batch: list[list[int]]) -> list[list[float]]:
        """用按长度分桶的 batch forward 返回多条 prefix 的 next-token logits。

        Some HF/Qwen paths can produce different next-token logits for padded
        heterogeneous-length batches than for the same prefixes run alone.  The
        verifier relies on single-prefix greedy equivalence for correctness, so
        we batch only prefixes with identical lengths and avoid padding here.
        """
        import torch

        if not prefix_ids_batch:
            return []
        if any(not prefix_ids for prefix_ids in prefix_ids_batch):
            raise ValueError("TransformersCausalLMRunner.next_token_logits_batch requires non-empty prefixes.")
        buckets: dict[int, list[tuple[int, list[int]]]] = {}
        for index, prefix_ids in enumerate(prefix_ids_batch):
            buckets.setdefault(len(prefix_ids), []).append((index, [int(token_id) for token_id in prefix_ids]))

        logits_by_index: list[list[float] | None] = [None] * len(prefix_ids_batch)
        for _length, items in buckets.items():
            input_ids = torch.tensor([prefix_ids for _index, prefix_ids in items], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                output = self.model(input_ids=input_ids)
                logits = output.logits[:, -1].detach().float().cpu().tolist()
            for row_index, (original_index, _prefix_ids) in enumerate(items):
                logits_by_index[original_index] = logits[row_index]
        if any(logits is None for logits in logits_by_index):
            raise RuntimeError("Missing logits for at least one batch item.")
        return [list(logits) for logits in logits_by_index if logits is not None]

    def linear_verify(self, request: LinearForwardInput) -> LinearForwardOutput:
        """验证一条 linear draft，复用 batch verifier 的 causal-safe 路径。"""
        return self.linear_verify_batch([request])[0]

    def linear_verify_batch(self, requests: list[LinearForwardInput]) -> list[LinearForwardOutput]:
        """用 causal-safe prefix batches 验证多条 linear drafts。

        Full `prefix + draft_tokens` single-pass verification should be valid for
        an ideal causal backend, but the current HF/Qwen eager path can disagree
        with prefix-only greedy logits at historical positions.  This fallback
        batches only the actually visible prefixes for each prediction point.
        """
        if not requests:
            return []
        if any(not request.prefix_ids for request in requests):
            raise ValueError("TransformersCausalLMRunner.linear_verify_batch requires non-empty prefixes.")

        outputs_by_index: dict[int, LinearForwardOutput] = {}
        active_items: list[tuple[int, LinearForwardInput]] = []
        for index, request in enumerate(requests):
            if not request.draft_tokens and not request.allow_bonus:
                outputs_by_index[index] = LinearForwardOutput(
                    draft_target_tokens=[],
                    bonus_token=None,
                    metadata={
                        "linear_forward_kind": "linear_single_pass_noop",
                        "linear_forward_batch_kind": "linear_single_pass_noop",
                        "batch_index": index,
                        "batch_size": len(requests),
                        "target_forward_call_count": 0,
                    },
                )
                continue
            active_items.append((index, request))

        if not active_items:
            return [outputs_by_index[index] for index in range(len(requests))]

        prediction_points: list[tuple[int, int, str, int | None, list[int]]] = []
        for active_index, (original_index, request) in enumerate(active_items):
            draft_tokens = [int(token_id) for token_id in request.draft_tokens]
            for draft_index in range(len(draft_tokens)):
                prediction_points.append(
                    (
                        active_index,
                        original_index,
                        "draft",
                        draft_index,
                        [*request.prefix_ids, *draft_tokens[:draft_index]],
                    )
                )
            if request.allow_bonus:
                prediction_points.append(
                    (
                        active_index,
                        original_index,
                        "bonus",
                        None,
                        [*request.prefix_ids, *draft_tokens],
                    )
                )

        token_ids = self.greedy_next_tokens([point[4] for point in prediction_points])
        draft_targets_by_index: dict[int, list[int]] = {original_index: [] for original_index, _request in active_items}
        bonus_by_index: dict[int, int | None] = {original_index: None for original_index, _request in active_items}
        prefix_lengths_by_index: dict[int, set[int]] = {original_index: set() for original_index, _request in active_items}
        for (_active_index, original_index, kind, _draft_index, prefix_ids), token_id in zip(prediction_points, token_ids):
            prefix_lengths_by_index[original_index].add(len(prefix_ids))
            if kind == "draft":
                draft_targets_by_index[original_index].append(int(token_id))
            else:
                bonus_by_index[original_index] = int(token_id)

        global_prefix_length_count = len({len(point[4]) for point in prediction_points})
        for active_index, (original_index, request) in enumerate(active_items):
            draft_len = len(request.draft_tokens)
            prefix_lengths = sorted(prefix_lengths_by_index[original_index])
            outputs_by_index[original_index] = LinearForwardOutput(
                draft_target_tokens=draft_targets_by_index[original_index],
                bonus_token=bonus_by_index[original_index],
                metadata={
                    "linear_forward_kind": "linear_prefix_batch",
                    "linear_forward_batch_kind": "linear_prefix_batch",
                    "batch_index": original_index,
                    "batch_size": len(requests),
                    "active_batch_size": len(active_items),
                    "active_batch_index": active_index,
                    "draft_token_count": draft_len,
                    "prediction_prefix_count": draft_len + (1 if request.allow_bonus else 0),
                    "length_bucket_count": global_prefix_length_count,
                    "request_length_bucket_count": len(prefix_lengths),
                    "prediction_prefix_lengths": prefix_lengths,
                    "bonus_computed": bool(request.allow_bonus),
                    "target_forward_call_count": max(1, len(prefix_lengths)),
                    "causal_safe_prefix_batch": True,
                },
            )
        return [outputs_by_index[index] for index in range(len(requests))]

    def tree_forward(self, request: TreeForwardInput) -> TreeForwardOutput:
        """用 packed tree attention 一次 forward 计算所有 parent choice。"""
        try:
            return self._tree_forward_attention(request)
        except Exception as exc:  # pragma: no cover - 真实 HF backend 差异较大
            fallback = super().tree_forward(request)
            fallback.metadata.update(
                {
                    "tree_forward_kind": "tree_attention_fallback_to_batched_next_token",
                    "tree_attention_error": str(exc),
                }
            )
            return fallback

    def tree_forward_batch(self, requests: list[TreeForwardInput]) -> list[TreeForwardOutput]:
        """用一次 heterogeneous padded tree-attention batch forward 验证多棵树。"""
        if not requests:
            return []
        try:
            return self._tree_forward_attention_batch(requests)
        except Exception as exc:  # pragma: no cover - 真实 HF backend 差异较大
            outputs = super().tree_forward_batch(requests)
            for output in outputs:
                output.metadata.update(
                    {
                        "tree_forward_batch_kind": "tree_attention_batch_fallback_to_sequential",
                        "tree_attention_batch_error": str(exc),
                    }
                )
            return outputs

    def _tree_forward_attention(self, request: TreeForwardInput) -> TreeForwardOutput:
        """HF 4D attention-mask tree forward 实现。"""
        import torch

        if not request.prefix_ids:
            raise ValueError("TransformersCausalLMRunner.tree_forward requires a non-empty prefix.")
        if not request.nodes:
            return TreeForwardOutput(
                choices=[],
                metadata={
                    "tree_forward_kind": "tree_attention",
                    "node_count": 0,
                    "packed_token_count": len(request.prefix_ids),
                },
            )
        _validate_tree_forward_nodes(request.nodes)
        prefix_len = len(request.prefix_ids)
        input_ids = [*request.prefix_ids, *[node.token_id for node in request.nodes]]
        position_ids = [
            *range(prefix_len),
            *[prefix_len + int(node.depth) - 1 for node in request.nodes],
        ]
        allow_mask = _tree_attention_allow_mask(prefix_len, request.nodes)
        mask_dtype = _first_parameter_dtype(self.model, torch, fallback=torch.float32)
        additive_mask = torch.zeros(
            (1, 1, len(input_ids), len(input_ids)),
            dtype=mask_dtype,
            device=self.device,
        )
        additive_mask.masked_fill_(
            ~torch.tensor(allow_mask, dtype=torch.bool, device=self.device),
            _attention_mask_min(torch, mask_dtype),
        )
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        position_tensor = torch.tensor([position_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_tensor,
                attention_mask=additive_mask,
                position_ids=position_tensor,
                use_cache=False,
            )
            parent_positions = _tree_parent_positions(prefix_len, request.nodes)
            logits = output.logits[0, torch.tensor(parent_positions, device=output.logits.device)].detach().float()
            target_token_ids = torch.argmax(logits, dim=-1).cpu().tolist()
        parent_ids = _tree_parent_ids(request.nodes)
        return TreeForwardOutput(
            choices=[
                TreeForwardChoice(
                    parent_node_id=parent_id,
                    target_token_id=int(token_id),
                    prefix_len=_tree_parent_prefix_len(prefix_len, request.nodes, parent_id),
                )
                for parent_id, token_id in zip(parent_ids, target_token_ids)
            ],
            metadata={
                "tree_forward_kind": "tree_attention",
                "node_count": len(request.nodes),
                "packed_token_count": len(input_ids),
                "choice_count": len(parent_ids),
                "position_ids": position_ids,
            },
        )

    def _tree_forward_attention_batch(self, requests: list[TreeForwardInput]) -> list[TreeForwardOutput]:
        """HF 4D attention-mask heterogeneous tree batch 实现。"""
        import torch

        if any(not request.prefix_ids for request in requests):
            raise ValueError("TransformersCausalLMRunner.tree_forward_batch requires non-empty prefixes.")

        empty_outputs: dict[int, TreeForwardOutput] = {}
        active_items: list[tuple[int, TreeForwardInput]] = []
        for index, request in enumerate(requests):
            if not request.nodes:
                empty_outputs[index] = TreeForwardOutput(
                    choices=[],
                    metadata={
                        "tree_forward_kind": "tree_attention_batch",
                        "tree_forward_batch_kind": "tree_attention_batch",
                        "batch_index": index,
                        "batch_size": len(requests),
                        "node_count": 0,
                        "packed_token_count": len(request.prefix_ids),
                        "choice_count": 0,
                    },
                )
                continue
            _validate_tree_forward_nodes(request.nodes)
            active_items.append((index, request))
        if not active_items:
            return [empty_outputs[index] for index in range(len(requests))]

        pad_token_id = _pad_token_id(self.tokenizer, self.model)
        packed_rows: list[list[int]] = []
        position_rows: list[list[int]] = []
        allow_masks: list[list[list[bool]]] = []
        packed_lens: list[int] = []
        for _index, request in active_items:
            prefix_len = len(request.prefix_ids)
            input_ids = [*request.prefix_ids, *[node.token_id for node in request.nodes]]
            position_ids = [
                *range(prefix_len),
                *[prefix_len + int(node.depth) - 1 for node in request.nodes],
            ]
            packed_rows.append(input_ids)
            position_rows.append(position_ids)
            allow_masks.append(_tree_attention_allow_mask(prefix_len, request.nodes))
            packed_lens.append(len(input_ids))

        max_len = max(packed_lens)
        padded_rows: list[list[int]] = []
        padded_positions: list[list[int]] = []
        padded_masks: list[list[list[bool]]] = []
        for input_ids, position_ids, allow_mask in zip(packed_rows, position_rows, allow_masks):
            pad_count = max_len - len(input_ids)
            padded_rows.append([*input_ids, *([pad_token_id] * pad_count)])
            padded_positions.append([*position_ids, *([0] * pad_count)])
            padded_masks.append(_pad_tree_attention_allow_mask(allow_mask, max_len))

        mask_dtype = _first_parameter_dtype(self.model, torch, fallback=torch.float32)
        additive_mask = torch.zeros(
            (len(active_items), 1, max_len, max_len),
            dtype=mask_dtype,
            device=self.device,
        )
        additive_mask.masked_fill_(
            ~torch.tensor(padded_masks, dtype=torch.bool, device=self.device).unsqueeze(1),
            _attention_mask_min(torch, mask_dtype),
        )
        input_tensor = torch.tensor(padded_rows, dtype=torch.long, device=self.device)
        position_tensor = torch.tensor(padded_positions, dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(
                input_ids=input_tensor,
                attention_mask=additive_mask,
                position_ids=position_tensor,
                use_cache=False,
            )

        outputs_by_index: dict[int, TreeForwardOutput] = dict(empty_outputs)
        for row_index, (original_index, request) in enumerate(active_items):
            prefix_len = len(request.prefix_ids)
            parent_positions = _tree_parent_positions(prefix_len, request.nodes)
            logits = output.logits[
                row_index,
                torch.tensor(parent_positions, device=output.logits.device),
            ].detach().float()
            target_token_ids = torch.argmax(logits, dim=-1).cpu().tolist()
            parent_ids = _tree_parent_ids(request.nodes)
            outputs_by_index[original_index] = TreeForwardOutput(
                choices=[
                    TreeForwardChoice(
                        parent_node_id=parent_id,
                        target_token_id=int(token_id),
                        prefix_len=_tree_parent_prefix_len(prefix_len, request.nodes, parent_id),
                    )
                    for parent_id, token_id in zip(parent_ids, target_token_ids)
                ],
                metadata={
                    "tree_forward_kind": "tree_attention_batch",
                    "tree_forward_batch_kind": "tree_attention_batch",
                    "batch_index": original_index,
                    "batch_size": len(requests),
                    "active_batch_size": len(active_items),
                    "node_count": len(request.nodes),
                    "packed_token_count": len(packed_rows[row_index]),
                    "padded_token_count": max_len,
                    "choice_count": len(parent_ids),
                    "position_ids": position_rows[row_index],
                },
            )
        return [outputs_by_index[index] for index in range(len(requests))]

    def next_token_topk(self, prefix_ids: list[int], k: int) -> list[TopKToken]:
        """在设备端做 top-k，只把 k 个 token/logprob 搬回 CPU。"""
        import torch

        if k <= 0:
            return []
        if not prefix_ids:
            raise ValueError("TransformersCausalLMRunner.next_token_topk requires a non-empty prefix.")
        input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model(input_ids=input_ids)
            logits = output.logits[0, -1].detach().float()
            k = min(int(k), int(logits.numel()))
            values, indices = torch.topk(torch.log_softmax(logits, dim=-1), k=k, largest=True, sorted=True)
        return [
            TopKToken(
                token_id=int(token_id),
                logprob=float(logprob),
                rank=rank,
            )
            for rank, (token_id, logprob) in enumerate(zip(indices.cpu().tolist(), values.cpu().tolist()))
        ]

    def next_token_topk_batch(self, prefix_ids_batch: list[list[int]], k: int) -> list[list[TopKToken]]:
        """用一个 padded batch forward 返回多条 prefix 的 top-k。"""
        if k <= 0:
            return [[] for _prefix_ids in prefix_ids_batch]
        logits_batch = self.next_token_logits_batch(prefix_ids_batch)
        return [_topk_from_logits(logits, k) for logits in logits_batch]

    def backend_capabilities(self) -> ModelBackendCapabilities:
        """Hugging Face eager backend 的能力声明。"""
        return ModelBackendCapabilities(
            backend_name=self.backend_name,
            backend_fallback=self.backend_fallback,
            fallback_reason=self.fallback_reason,
            supports_topk=True,
            supports_batched_topk=True,
            supports_batched_next_token=True,
            supports_linear_verify_batch=True,
            supports_tree_attention=True,
            supports_tree_forward_batch=True,
            supports_kv_cache=False,
            supports_cuda_graph=False,
        )


@dataclass
class CachedTransformersCausalLMRunner(TransformersCausalLMRunner):
    """使用 Hugging Face past_key_values 的增量 KV-cache runner。

    这个 backend 不实现旧 SpecEdge 的 tree attention 或 CUDA graph capture，
    但它是真实可运行的 KV-cache 路径：连续扩展 prefix 时只 forward 新 token。
    """

    backend_name: str = "hf_cached"
    max_cache_entries: int = 256
    _prefix_cache: dict[tuple[int, ...], tuple[Any, Any]] = field(default_factory=dict, init=False, repr=False)

    def reset(self, request_id: str | None = None) -> None:
        """清空 prefix KV cache。"""
        del request_id
        self._prefix_cache.clear()

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        """用 prefix KV cache 取下一 token logits。"""
        logits = self._next_token_logits_tensor(prefix_ids)
        return logits.detach().float().cpu().tolist()

    def next_token_topk(self, prefix_ids: list[int], k: int) -> list[TopKToken]:
        """用 prefix KV cache 在设备端做 top-k。"""
        import torch

        if k <= 0:
            return []
        logits = self._next_token_logits_tensor(prefix_ids).detach().float()
        k = min(int(k), int(logits.numel()))
        values, indices = torch.topk(torch.log_softmax(logits, dim=-1), k=k, largest=True, sorted=True)
        return [
            TopKToken(
                token_id=int(token_id),
                logprob=float(logprob),
                rank=rank,
            )
            for rank, (token_id, logprob) in enumerate(zip(indices.cpu().tolist(), values.cpu().tolist()))
        ]

    def next_token_topk_batch(self, prefix_ids_batch: list[list[int]], k: int) -> list[list[TopKToken]]:
        """用 KV cache 路径处理一批 top-k；真实 graph backend 后续可覆盖为融合 batch。"""
        return [self.next_token_topk(prefix_ids, k) for prefix_ids in prefix_ids_batch]

    def linear_verify_batch(self, requests: list[LinearForwardInput]) -> list[LinearForwardOutput]:
        """用 prefix KV cache 验证 linear drafts，只 forward draft tail。"""
        if not requests:
            return []
        if any(not request.prefix_ids for request in requests):
            raise ValueError("CachedTransformersCausalLMRunner.linear_verify_batch requires non-empty prefixes.")
        return [
            self._linear_verify_cached(request, batch_index=index, batch_size=len(requests))
            for index, request in enumerate(requests)
        ]

    def backend_capabilities(self) -> ModelBackendCapabilities:
        """HF cached backend 的能力声明。"""
        return ModelBackendCapabilities(
            backend_name=self.backend_name,
            backend_fallback=self.backend_fallback,
            fallback_reason=self.fallback_reason,
            supports_topk=True,
            supports_batched_topk=False,
            supports_batched_next_token=True,
            supports_linear_verify_batch=True,
            supports_tree_attention=True,
            supports_tree_forward_batch=True,
            supports_kv_cache=True,
            supports_cuda_graph=False,
        )

    def _next_token_logits_tensor(self, prefix_ids: list[int]) -> Any:
        """返回 prefix 的下一 token logits tensor，并缓存 past_key_values。"""
        if not prefix_ids:
            raise ValueError("CachedTransformersCausalLMRunner requires a non-empty prefix.")
        key = tuple(int(token_id) for token_id in prefix_ids)
        logits, _past_key_values, _cache_hit, _forward_calls = self._prefix_cache_entry(key)
        return logits

    def _linear_verify_cached(
        self,
        request: LinearForwardInput,
        *,
        batch_index: int,
        batch_size: int,
    ) -> LinearForwardOutput:
        """验证单条 linear draft，并把完整 draft 路径写入 prefix cache。"""
        import torch

        prefix_key = tuple(int(token_id) for token_id in request.prefix_ids)
        prefix_logits, prefix_past, prefix_cache_hit, prefix_forward_calls = self._prefix_cache_entry(prefix_key)
        draft_tokens = [int(token_id) for token_id in request.draft_tokens]
        draft_target_tokens: list[int] = []
        bonus_token: int | None = None
        tail_forward_calls = 0
        tail_token_count = 0

        if not draft_tokens:
            if request.allow_bonus:
                bonus_token = int(torch.argmax(prefix_logits.detach().float()).item())
            return LinearForwardOutput(
                draft_target_tokens=[],
                bonus_token=bonus_token,
                metadata={
                    "linear_forward_kind": "linear_single_pass_kv_cache",
                    "linear_forward_batch_kind": "linear_single_pass_kv_cache",
                    "batch_index": batch_index,
                    "batch_size": batch_size,
                    "prefix_cache_hit": prefix_cache_hit,
                    "prefix_forward_call_count": prefix_forward_calls,
                    "tail_forward_call_count": 0,
                    "tail_token_count": 0,
                    "target_forward_call_count": prefix_forward_calls,
                    "kv_cache": True,
                },
            )

        draft_target_tokens.append(int(torch.argmax(prefix_logits.detach().float()).item()))
        needs_tail_forward = len(draft_tokens) > 1 or bool(request.allow_bonus)
        if needs_tail_forward:
            input_ids = torch.tensor([draft_tokens], dtype=torch.long, device=self.device)
            tail_past = _clone_past_key_values(prefix_past)
            position_ids, cache_position, attention_mask = _cached_tail_position_inputs(
                torch,
                prefix_len=len(prefix_key),
                token_count=len(draft_tokens),
                device=self.device,
                mask_dtype=_first_parameter_dtype(self.model, torch, fallback=torch.float32),
            )
            with torch.inference_mode():
                output = _cached_forward(
                    self.model,
                    input_ids=input_ids,
                    past_key_values=tail_past,
                    use_cache=True,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    cache_position=cache_position,
                )
            tail_logits = output.logits[0].detach().float()
            tail_forward_calls = 1
            tail_token_count = len(draft_tokens)
            if len(draft_tokens) > 1:
                next_logits = tail_logits[: len(draft_tokens) - 1]
                draft_target_tokens.extend(
                    int(token_id)
                    for token_id in torch.argmax(next_logits, dim=-1).cpu().tolist()
                )
            if request.allow_bonus:
                bonus_token = int(torch.argmax(tail_logits[len(draft_tokens) - 1]).item())
            full_key = (*prefix_key, *draft_tokens)
            full_past = _past_key_values_from_output(output)
            self._remember_prefix(full_key, full_past, tail_logits[-1].detach())

        return LinearForwardOutput(
            draft_target_tokens=draft_target_tokens,
            bonus_token=bonus_token,
            metadata={
                "linear_forward_kind": "linear_single_pass_kv_cache",
                "linear_forward_batch_kind": "linear_single_pass_kv_cache",
                "batch_index": batch_index,
                "batch_size": batch_size,
                "draft_token_count": len(draft_tokens),
                "prefix_cache_hit": prefix_cache_hit,
                "prefix_forward_call_count": prefix_forward_calls,
                "tail_forward_call_count": tail_forward_calls,
                "tail_token_count": tail_token_count,
                "target_forward_call_count": prefix_forward_calls + tail_forward_calls,
                "kv_cache": True,
            },
        )

    def _prefix_cache_entry(self, key: tuple[int, ...]) -> tuple[Any, Any, bool, int]:
        """返回 key 的 next-token logits 和 past cache；必要时只 forward 缺失 tail。"""
        import torch

        if not key:
            raise ValueError("CachedTransformersCausalLMRunner requires a non-empty prefix.")
        cached = self._prefix_cache.get(key)
        if cached is not None:
            return cached[1], cached[0], True, 0

        base_key = self._longest_cached_prefix(key)
        if base_key:
            past_key_values = _clone_past_key_values(self._prefix_cache[base_key][0])
            input_tail = list(key[len(base_key) :])
        else:
            past_key_values = None
            input_tail = list(key)
        base_len = len(key) - len(input_tail)
        input_ids = torch.tensor([input_tail], dtype=torch.long, device=self.device)
        position_ids, cache_position, attention_mask = _cached_tail_position_inputs(
            torch,
            prefix_len=base_len,
            token_count=len(input_tail),
            device=self.device,
            mask_dtype=_first_parameter_dtype(self.model, torch, fallback=torch.float32),
        )
        with torch.inference_mode():
            output = _cached_forward(
                self.model,
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
            )
        logits = output.logits[0, -1].detach()
        past = _past_key_values_from_output(output)
        self._remember_prefix(key, past, logits)
        return logits, past, False, 1

    def _longest_cached_prefix(self, key: tuple[int, ...]) -> tuple[int, ...] | None:
        """寻找 key 的最长已缓存前缀。"""
        for length in range(len(key) - 1, 0, -1):
            candidate = key[:length]
            if candidate in self._prefix_cache:
                return candidate
        return None

    def _remember_prefix(self, key: tuple[int, ...], past_key_values: Any, logits: Any) -> None:
        """记录 prefix cache，并按插入顺序限制容量。"""
        if past_key_values is None:
            return
        self._prefix_cache[key] = (past_key_values, logits)
        while len(self._prefix_cache) > int(self.max_cache_entries):
            oldest = next(iter(self._prefix_cache))
            self._prefix_cache.pop(oldest)


def _resolve_torch_dtype(torch: Any, torch_dtype: str | None) -> Any | None:
    """把命令行 dtype 字符串转换成 torch dtype。"""
    if torch_dtype in (None, "auto"):
        return None if torch_dtype is None else "auto"
    if torch_dtype == "bf16":
        return torch.bfloat16
    if torch_dtype == "fp16":
        return torch.float16
    if torch_dtype == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {torch_dtype}")


def _first_parameter_device(model: Any, *, fallback: str | None) -> str | None:
    """推断模型输入应该放到哪个 device。"""
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return fallback


def _first_parameter_dtype(model: Any, torch: Any, *, fallback: Any) -> Any:
    """推断模型 attention mask 应使用的浮点 dtype。"""
    try:
        dtype = next(model.parameters()).dtype
    except (AttributeError, StopIteration):
        return fallback
    probe = torch.empty((), dtype=dtype)
    return dtype if probe.is_floating_point() else fallback


def _past_key_values_from_output(output: Any) -> Any:
    """兼容 dataclass/dict 风格 Transformers output。"""
    if hasattr(output, "past_key_values"):
        return output.past_key_values
    if isinstance(output, dict):
        return output.get("past_key_values")
    return None


def _cached_tail_position_inputs(
    torch: Any,
    *,
    prefix_len: int,
    token_count: int,
    device: str | None,
    mask_dtype: Any,
) -> tuple[Any, Any, Any]:
    """Build explicit position/cache inputs for a cached tail forward."""
    if token_count <= 0:
        raise ValueError("cached tail forward requires at least one input token.")
    start = int(prefix_len)
    stop = start + int(token_count)
    cache_position = torch.arange(start, stop, dtype=torch.long, device=device)
    position_ids = cache_position.unsqueeze(0)
    allowed = torch.ones((int(token_count), stop), dtype=torch.bool, device=device)
    if token_count > 1:
        tail_future = torch.triu(
            torch.ones((int(token_count), int(token_count)), dtype=torch.bool, device=device),
            diagonal=1,
        )
        allowed[:, start:stop] = ~tail_future
    attention_mask = torch.zeros((1, 1, int(token_count), stop), dtype=mask_dtype, device=device)
    attention_mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), _attention_mask_min(torch, mask_dtype))
    return position_ids, cache_position, attention_mask


def _cached_forward(
    model: Any,
    *,
    input_ids: Any,
    past_key_values: Any,
    use_cache: bool,
    attention_mask: Any,
    position_ids: Any,
    cache_position: Any,
) -> Any:
    """Call a cached HF model with only the kwargs its forward accepts."""
    kwargs = {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "use_cache": use_cache,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "cache_position": cache_position,
    }
    accepted = _accepted_model_kwargs(model)
    if accepted is not None:
        kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return model(**kwargs)


def _accepted_model_kwargs(model: Any) -> set[str] | None:
    """Return accepted keyword names, or None when the callable accepts **kwargs."""
    import inspect

    callable_obj = getattr(model, "forward", None) or getattr(model, "__call__")
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return None
    accepted: set[str] = set()
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return None
        if parameter.kind in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            accepted.add(parameter.name)
    return accepted


def _clone_past_key_values(past_key_values: Any) -> Any:
    """Clone cached KV state before reuse so mutable HF caches cannot corrupt prefixes."""
    if past_key_values is None:
        return None
    clone = getattr(past_key_values, "clone", None)
    if callable(clone):
        try:
            return clone()
        except TypeError:
            pass
    if hasattr(past_key_values, "detach") and hasattr(past_key_values, "clone"):
        return past_key_values.detach().clone()
    if isinstance(past_key_values, tuple):
        return tuple(_clone_past_key_values(value) for value in past_key_values)
    if isinstance(past_key_values, list):
        return [_clone_past_key_values(value) for value in past_key_values]
    if isinstance(past_key_values, dict):
        return {key: _clone_past_key_values(value) for key, value in past_key_values.items()}
    return copy.deepcopy(past_key_values)


def _topk_from_logits(logits: list[float], k: int) -> list[TopKToken]:
    """在 CPU list logits 上生成稳定 top-k/logprob。"""
    if k <= 0:
        return []
    if not logits:
        raise ValueError("top-k requires non-empty logits.")
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


def _pad_token_id(tokenizer: Any, model: Any) -> int:
    """推断 batch padding 使用的 token id。"""
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


def _validate_tree_forward_nodes(nodes: list[Any]) -> None:
    """校验 tree_forward nodes 是父节点先于子节点的拓扑序。"""
    seen: set[int] = set()
    for node in nodes:
        if node.node_id in seen:
            raise ValueError(f"Duplicate tree_forward node id: {node.node_id}")
        if node.parent_id is None:
            if int(node.depth) != 1:
                raise ValueError(f"Root child depth must be 1: {node.node_id}")
        else:
            if node.parent_id not in seen:
                raise ValueError(f"tree_forward parent must appear before child: {node.node_id}")
        seen.add(int(node.node_id))


def _tree_parent_ids(nodes: list[Any]) -> list[int | None]:
    """返回所有需要 target choice 的 parent ids。"""
    parent_ids: list[int | None] = []
    for node in nodes:
        if node.parent_id not in parent_ids:
            parent_ids.append(node.parent_id)
    return parent_ids


def _tree_parent_positions(prefix_len: int, nodes: list[Any]) -> list[int]:
    """返回每个 parent 对应的 logits 位置。"""
    node_positions = {int(node.node_id): prefix_len + index for index, node in enumerate(nodes)}
    positions: list[int] = []
    for parent_id in _tree_parent_ids(nodes):
        positions.append(prefix_len - 1 if parent_id is None else node_positions[int(parent_id)])
    return positions


def _tree_parent_prefix_len(prefix_len: int, nodes: list[Any], parent_id: int | None) -> int:
    """返回 parent prefix 的 token 长度。"""
    if parent_id is None:
        return prefix_len
    nodes_by_id = {int(node.node_id): node for node in nodes}
    depth = 0
    current = nodes_by_id[int(parent_id)]
    while current is not None:
        depth += 1
        current = nodes_by_id.get(int(current.parent_id)) if current.parent_id is not None else None
    return prefix_len + depth


def _tree_attention_allow_mask(prefix_len: int, nodes: list[Any]) -> list[list[bool]]:
    """构造 packed tree attention 的可见性矩阵。"""
    seq_len = prefix_len + len(nodes)
    mask = [[False for _ in range(seq_len)] for _ in range(seq_len)]
    for query_pos in range(prefix_len):
        for key_pos in range(query_pos + 1):
            mask[query_pos][key_pos] = True

    node_positions = {int(node.node_id): prefix_len + index for index, node in enumerate(nodes)}
    ancestors_by_node = _tree_ancestors_by_node(nodes)
    for node in nodes:
        query_pos = node_positions[int(node.node_id)]
        for key_pos in range(prefix_len):
            mask[query_pos][key_pos] = True
        for ancestor_id in ancestors_by_node[int(node.node_id)]:
            mask[query_pos][node_positions[int(ancestor_id)]] = True
        mask[query_pos][query_pos] = True
    return mask


def _pad_tree_attention_allow_mask(mask: list[list[bool]], target_len: int) -> list[list[bool]]:
    """把单棵树的可见性矩阵 padding 到 batch max_len。"""
    source_len = len(mask)
    padded = [[False for _ in range(target_len)] for _ in range(target_len)]
    for row_index, row in enumerate(mask):
        for column_index, value in enumerate(row):
            padded[row_index][column_index] = bool(value)
    for pad_index in range(source_len, target_len):
        padded[pad_index][pad_index] = True
    return padded


def _tree_ancestors_by_node(nodes: list[Any]) -> dict[int, list[int]]:
    """返回每个节点从 root child 到 parent 的 ancestor node ids。"""
    nodes_by_id = {int(node.node_id): node for node in nodes}
    ancestors: dict[int, list[int]] = {}
    for node in nodes:
        path: list[int] = []
        current = nodes_by_id.get(int(node.parent_id)) if node.parent_id is not None else None
        while current is not None:
            path.append(int(current.node_id))
            current = nodes_by_id.get(int(current.parent_id)) if current.parent_id is not None else None
        ancestors[int(node.node_id)] = list(reversed(path))
    return ancestors


def _attention_mask_min(torch: Any, dtype: Any) -> float:
    """4D additive attention mask 使用的阻断值。"""
    del torch, dtype
    return float("-inf")
