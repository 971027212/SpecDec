from __future__ import annotations

"""Tree target verifier for SpecEdge-style speculative decoding."""

from dataclasses import dataclass, field, replace
from time import perf_counter_ns
from typing import Any

from specplatform.core import CandidateNode, CandidateProposal, RuntimeContext, VerificationResult
from specplatform.model import CausalLMRunner, TreeForwardChoice, TreeForwardInput, TreeForwardNode
from specplatform.verification.base import VerifierBackend
from specplatform.verification.schema import TreeVerifyRequest, TreeVerifyResponse


@dataclass
class TreeVerifier(VerifierBackend):
    """用 target CausalLMRunner 对 tree draft proposal 做 greedy 验证。"""

    model: CausalLMRunner
    backend_name: str = "tree_local"
    metadata: dict[str, Any] = field(default_factory=dict)

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        """验证单个 tree proposal；只返回 verifier 看到的事实。"""
        if proposal.shape != "tree" or proposal.tree is None:
            raise ValueError("TreeVerifier only supports tree proposals.")
        request = TreeVerifyRequest(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            prefix_ids=_proposal_prefix_ids(proposal),
            tree=proposal.tree,
            eos_token_ids=_eos_token_ids(proposal, context),
            allow_bonus=bool(proposal.metadata.get("allow_bonus", True)),
            metadata=dict(proposal.metadata),
        )
        response = self.verify_request(request, context)
        _validate_response_for_request(response, request)
        accepted_tokens = _tokens_for_node_ids(response.accepted_node_ids, request)
        return VerificationResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            shape=proposal.shape,
            accepted_prefix_len=len(response.accepted_node_ids),
            verified_tokens=accepted_tokens,
            bonus_token=response.bonus_token,
            timing=dict(response.timing),
            payload=response.to_dict(),
            metadata={
                "backend_name": self.backend_name,
                **dict(self.metadata),
                **dict(response.metadata),
                "timing": dict(response.timing),
            },
        )

    def verify_batch(
        self,
        proposals: list[CandidateProposal],
        context: RuntimeContext | None = None,
    ) -> list[VerificationResult]:
        """批量验证 tree proposals。"""
        if not proposals:
            return []
        requests = [
            TreeVerifyRequest(
                request_id=proposal.request_id,
                proposal_id=proposal.proposal_id,
                prefix_ids=_proposal_prefix_ids(proposal),
                tree=proposal.tree,
                eos_token_ids=_eos_token_ids(proposal, context),
                allow_bonus=bool(proposal.metadata.get("allow_bonus", True)),
                metadata=dict(proposal.metadata),
            )
            for proposal in proposals
            if proposal.shape == "tree" and proposal.tree is not None
        ]
        if len(requests) != len(proposals):
            raise ValueError("TreeVerifier only supports tree proposals.")
        responses = self.verify_requests_batch(
            requests,
            context,
            batch_id="local-tree-batch:" + ",".join(proposal.proposal_id for proposal in proposals),
        )
        results: list[VerificationResult] = []
        for proposal, request, response in zip(proposals, requests, responses):
            _validate_response_for_request(response, request)
            accepted_tokens = _tokens_for_node_ids(response.accepted_node_ids, request)
            results.append(
                VerificationResult(
                    request_id=proposal.request_id,
                    proposal_id=proposal.proposal_id,
                    shape=proposal.shape,
                    accepted_prefix_len=len(response.accepted_node_ids),
                    verified_tokens=accepted_tokens,
                    bonus_token=response.bonus_token,
                    timing=dict(response.timing),
                    payload=response.to_dict(),
                    metadata={
                        "backend_name": self.backend_name,
                        **dict(self.metadata),
                        **dict(response.metadata),
                        "timing": dict(response.timing),
                    },
                )
            )
        return results

    def verify_requests_batch(
        self,
        requests: list[TreeVerifyRequest],
        context: RuntimeContext | None = None,
        *,
        batch_id: str | None = None,
    ) -> list[TreeVerifyResponse]:
        """批量验证 schema 层 tree 请求，优先走模型层 fused tree_forward_batch。"""
        del context
        if not requests:
            return []
        if len(requests) == 1 or not self.model.backend_capabilities().supports_tree_forward_batch:
            responses = [self.verify_request(request) for request in requests]
            return [
                _with_batch_timing(
                    response,
                    batch_id=batch_id,
                    batch_size=len(requests),
                    batch_index=index,
                    tree_forward_batch_kind="fallback_sequential",
                )
                for index, response in enumerate(responses)
            ]

        for request in requests:
            _validate_tree_request(request)

        batch_choices, batch_events = _timed_tree_forward_batch(
            self.model,
            requests,
            start_index=0,
        )
        responses: list[TreeVerifyResponse] = []
        for index, request in enumerate(requests):
            nodes_by_id = request.tree.nodes_by_id()
            children_by_parent = request.tree.children_by_parent()
            choice_specs = [
                (choice.parent_node_id, choice.prefix_len, choice.target_token_id)
                for choice in batch_choices[index]
            ]
            choice_specs, guard_events = _guard_tree_forward_root_choice(
                self.model,
                request,
                nodes_by_id,
                children_by_parent,
                choice_specs,
                start_index=1,
            )
            response = _response_from_choice_specs(
                self.model,
                request,
                choice_specs,
                [batch_events[index], *guard_events],
            )
            response = _with_batch_timing(
                response,
                batch_id=batch_id,
                batch_size=len(requests),
                batch_index=index,
                tree_forward_batch_kind=str(
                    batch_events[index]
                    .get("metadata", {})
                    .get("tree_forward_batch_kind", batch_events[index].get("kind", "tree_attention_batch"))
                ),
            )
            responses.append(response)
        return responses

    def verify_request(
        self,
        request: TreeVerifyRequest,
        context: RuntimeContext | None = None,
    ) -> TreeVerifyResponse:
        """验证 HTTP/schema 层 tree 请求，供本地测试和 A100 service 共用。"""
        del context
        _validate_tree_request(request)

        nodes_by_id = request.tree.nodes_by_id()
        children_by_parent = request.tree.children_by_parent()
        forward_events: list[dict[str, Any]] = []

        if self.model.backend_capabilities().supports_tree_attention:
            tree_choices, forward_event = _timed_tree_forward(
                self.model,
                request,
                start_index=len(forward_events),
            )
            forward_events.append(forward_event)
            choice_specs = [
                (
                    choice.parent_node_id,
                    choice.prefix_len,
                    choice.target_token_id,
                )
                for choice in tree_choices
            ]
            choice_specs, guard_events = _guard_tree_forward_root_choice(
                self.model,
                request,
                nodes_by_id,
                children_by_parent,
                choice_specs,
                start_index=len(forward_events),
            )
            forward_events.extend(guard_events)
        else:
            parent_ids = list(children_by_parent)
            parent_prefixes = [
                [*request.prefix_ids, *_path_tokens(parent_id, nodes_by_id)]
                for parent_id in parent_ids
            ]
            target_tokens, choice_forward_events = _timed_tree_choice_tokens(
                self.model,
                parent_prefixes,
                parent_ids=parent_ids,
                start_index=len(forward_events),
            )
            forward_events.extend(choice_forward_events)
            choice_specs = [
                (parent_id, len(parent_prefix), target_token)
                for parent_id, parent_prefix, target_token in zip(parent_ids, parent_prefixes, target_tokens)
            ]

        return _response_from_choice_specs(self.model, request, choice_specs, forward_events)


def _validate_tree_request(request: TreeVerifyRequest) -> None:
    if not request.prefix_ids:
        raise ValueError("Tree verification requires a non-empty prefix.")
    if request.tree.root_prefix_len != len(request.prefix_ids):
        raise ValueError("Tree root_prefix_len must match prefix_ids length.")
    request.tree.validate()


def _response_from_choice_specs(
    model: CausalLMRunner,
    request: TreeVerifyRequest,
    choice_specs: list[tuple[int | None, int, int]],
    forward_events: list[dict[str, Any]],
) -> TreeVerifyResponse:
    """把 target choice facts 转成 tree verify response。"""
    children_by_parent = request.tree.children_by_parent()
    eos_token_ids = set(request.eos_token_ids)
    target_choices: list[dict[str, Any]] = []
    choices_by_parent: dict[int | None, dict[str, Any]] = {}

    for parent_id, prefix_len, target_token in choice_specs:
        children = children_by_parent[parent_id]
        matched_child = _select_matching_child(children, target_token)
        choice = {
            "parent_node_id": parent_id,
            "target_token_id": int(target_token),
            "candidate_node_ids": [child.node_id for child in children],
            "matched_node_id": None if matched_child is None else matched_child.node_id,
            "prefix_len": int(prefix_len),
        }
        target_choices.append(choice)
        choices_by_parent[parent_id] = choice

    accepted_node_ids: list[int] = []
    bonus_token: int | None = None
    current_parent: int | None = None
    while True:
        accepted_tokens = _tokens_for_node_ids(accepted_node_ids, request)
        if accepted_tokens and accepted_tokens[-1] in eos_token_ids:
            break

        children = children_by_parent.get(current_parent, [])
        if not children:
            if request.allow_bonus:
                bonus_token, forward_event = _timed_greedy_next_token(
                    model,
                    [*request.prefix_ids, *accepted_tokens],
                    index=len(target_choices),
                    kind="bonus",
                    metadata={"parent_node_id": current_parent},
                )
                forward_events.append(forward_event)
            break

        choice = choices_by_parent[current_parent]
        matched_node_id = choice.get("matched_node_id")
        if matched_node_id is None:
            # 和 linear 一样，allow_bonus=False 只禁止 full-match 后的额外 token，
            # 不禁止 mismatch 处的 target 纠偏 token。
            bonus_token = int(choice["target_token_id"])
            break

        matched_node_id = int(matched_node_id)
        accepted_node_ids.append(matched_node_id)
        current_parent = matched_node_id

    accepted_set = set(accepted_node_ids)
    rejected_node_ids = [
        node.node_id
        for node in request.tree.nodes
        if node.node_id not in accepted_set
    ]
    metadata = {
        "accepted_count": len(accepted_node_ids),
        "tree_node_count": len(request.tree.nodes),
        "full_tree_path_matched": _is_leaf(accepted_node_ids[-1], children_by_parent)
        if accepted_node_ids
        else False,
    }
    return TreeVerifyResponse(
        request_id=request.request_id,
        proposal_id=request.proposal_id,
        accepted_node_ids=accepted_node_ids,
        target_choices=target_choices,
        bonus_token=bonus_token,
        rejected_node_ids=rejected_node_ids,
        metadata=metadata,
        timing=_target_timing(forward_events),
    )


def _proposal_prefix_ids(proposal: CandidateProposal) -> list[int]:
    """从 proposal metadata 取出 draft 前的 prefix。"""
    prefix_ids = proposal.metadata.get("prefix_ids")
    if prefix_ids is None:
        raise ValueError("Tree proposal metadata must include prefix_ids.")
    return [int(token_id) for token_id in prefix_ids]


def _eos_token_ids(proposal: CandidateProposal, context: RuntimeContext | None) -> list[int]:
    """按 proposal metadata -> context.method_config -> context.run_config 的优先级读取 EOS。"""
    raw = proposal.metadata.get("eos_token_ids")
    if raw is None and context is not None:
        raw = context.method_config.get("eos_token_ids") or context.run_config.get("eos_token_ids")
    if raw is None:
        return []
    if isinstance(raw, int):
        return [int(raw)]
    return [int(token_id) for token_id in raw]


def _validate_response_for_request(
    response: TreeVerifyResponse,
    request: TreeVerifyRequest,
) -> None:
    """校验 verifier 响应仍然对应当前请求，避免 HTTP/schema 漂移被吞掉。"""
    if response.request_id != request.request_id:
        raise ValueError("TreeVerifyResponse request_id does not match request.")
    if response.proposal_id != request.proposal_id:
        raise ValueError("TreeVerifyResponse proposal_id does not match request.")
    known = {node.node_id for node in request.tree.nodes}
    unknown_accepted = [node_id for node_id in response.accepted_node_ids if node_id not in known]
    unknown_rejected = [node_id for node_id in response.rejected_node_ids if node_id not in known]
    if unknown_accepted or unknown_rejected:
        raise ValueError(
            "TreeVerifyResponse contains unknown node ids: "
            f"accepted={unknown_accepted}, rejected={unknown_rejected}."
        )
    if (
        not request.allow_bonus
        and response.bonus_token is not None
        and len(response.accepted_node_ids) == len(request.tree.nodes)
    ):
        raise ValueError("Verifier returned full-match bonus_token when allow_bonus is false.")


def _path_tokens(parent_id: int | None, nodes_by_id: dict[int, CandidateNode]) -> list[int]:
    """返回从根到 parent_id 的 token 路径。"""
    if parent_id is None:
        return []
    path: list[int] = []
    current: CandidateNode | None = nodes_by_id[parent_id]
    while current is not None:
        path.append(current.token_id)
        current = nodes_by_id.get(current.parent_id) if current.parent_id is not None else None
    path.reverse()
    return path


def _tokens_for_node_ids(node_ids: list[int], request: TreeVerifyRequest) -> list[int]:
    """按 accepted_node_ids 顺序取出 token ids。"""
    nodes_by_id = request.tree.nodes_by_id()
    return [int(nodes_by_id[int(node_id)].token_id) for node_id in node_ids]


def _select_matching_child(children: list[CandidateNode], target_token: int) -> CandidateNode | None:
    """在同一 parent 的 children 中选择匹配 target token 的确定性节点。"""
    matches = [child for child in children if int(child.token_id) == int(target_token)]
    if not matches:
        return None
    return sorted(
        matches,
        key=lambda node: (
            -(node.draft_logprob if node.draft_logprob is not None else float("-inf")),
            node.node_id,
        ),
    )[0]


def _is_leaf(node_id: int, children_by_parent: dict[int | None, list[CandidateNode]]) -> bool:
    """判断节点是否是当前 tree 的 leaf。"""
    return node_id not in children_by_parent


def _timed_greedy_next_token(
    model: CausalLMRunner,
    prefix_ids: list[int],
    *,
    index: int,
    kind: str,
    metadata: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    """运行一次 target greedy forward，并记录服务端细粒度耗时。"""
    start_ns = perf_counter_ns()
    token_id = int(model.greedy_next_token(prefix_ids))
    end_ns = perf_counter_ns()
    event = {
        "index": int(index),
        "kind": kind,
        "prefix_len": len(prefix_ids),
        "token_id": token_id,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ms": _duration_ms(start_ns, end_ns),
    }
    event.update(dict(metadata or {}))
    return token_id, event


def _timed_tree_forward(
    model: CausalLMRunner,
    request: TreeVerifyRequest,
    *,
    start_index: int,
) -> tuple[list[TreeForwardChoice], dict[str, Any]]:
    """运行一次解耦的 target tree_forward，并记录真实耗时。"""
    tree_input = TreeForwardInput(
        prefix_ids=list(request.prefix_ids),
        nodes=_tree_forward_nodes(request),
        metadata=dict(request.metadata),
    )
    start_ns = perf_counter_ns()
    output = model.tree_forward(tree_input)
    end_ns = perf_counter_ns()
    metadata = dict(output.metadata or {})
    event = {
        "index": int(start_index),
        "kind": str(metadata.get("tree_forward_kind", "tree_forward")),
        "choice_count": len(output.choices),
        "node_count": len(request.tree.nodes),
        "prefix_len": len(request.prefix_ids),
        "parent_node_ids": [choice.parent_node_id for choice in output.choices],
        "token_ids": [int(choice.target_token_id) for choice in output.choices],
        "prefix_lens": [int(choice.prefix_len) for choice in output.choices],
        "start_ns": start_ns,
        "end_ns": end_ns,
        "duration_ms": _duration_ms(start_ns, end_ns),
        "metadata": metadata,
    }
    for key in ("packed_token_count", "parent_prefix_count"):
        if key in metadata:
            event[key] = metadata[key]
    return list(output.choices), event


def _timed_tree_forward_batch(
    model: CausalLMRunner,
    requests: list[TreeVerifyRequest],
    *,
    start_index: int,
) -> tuple[list[list[TreeForwardChoice]], list[dict[str, Any]]]:
    """运行一次模型层 heterogeneous tree_forward_batch，并生成每个 item 的事件。"""
    tree_inputs = [
        TreeForwardInput(
            prefix_ids=list(request.prefix_ids),
            nodes=_tree_forward_nodes(request),
            metadata=dict(request.metadata),
        )
        for request in requests
    ]
    start_ns = perf_counter_ns()
    outputs = model.tree_forward_batch(tree_inputs)
    end_ns = perf_counter_ns()
    if len(outputs) != len(requests):
        raise ValueError("tree_forward_batch returned a different number of outputs.")
    batch_duration_ms = _duration_ms(start_ns, end_ns)
    attributed_duration_ms = batch_duration_ms / max(1, len(requests))
    batch_event_id = f"tree_forward_batch:{start_ns}:{len(requests)}"
    choices_by_request: list[list[TreeForwardChoice]] = []
    events: list[dict[str, Any]] = []
    for batch_index, (request, output) in enumerate(zip(requests, outputs)):
        metadata = dict(output.metadata or {})
        metadata.setdefault("tree_forward_batch_kind", "tree_attention_batch")
        metadata["shared_batch_event_id"] = batch_event_id
        event = {
            "index": int(start_index),
            "kind": str(metadata.get("tree_forward_kind", metadata.get("tree_forward_batch_kind", "tree_forward_batch"))),
            "choice_count": len(output.choices),
            "node_count": len(request.tree.nodes),
            "prefix_len": len(request.prefix_ids),
            "parent_node_ids": [choice.parent_node_id for choice in output.choices],
            "token_ids": [int(choice.target_token_id) for choice in output.choices],
            "prefix_lens": [int(choice.prefix_len) for choice in output.choices],
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ms": attributed_duration_ms,
            "shared_duration_ms": batch_duration_ms,
            "batch_index": batch_index,
            "batch_size": len(requests),
            "shared_batch_event_id": batch_event_id,
            "metadata": metadata,
        }
        for key in ("packed_token_count", "padded_token_count", "parent_prefix_count"):
            if key in metadata:
                event[key] = metadata[key]
        choices_by_request.append(list(output.choices))
        events.append(event)
    return choices_by_request, events


def _with_batch_timing(
    response: TreeVerifyResponse,
    *,
    batch_id: str | None,
    batch_size: int,
    batch_index: int,
    tree_forward_batch_kind: str,
) -> TreeVerifyResponse:
    timing = dict(response.timing)
    timing.update(
        {
            "batch_id": batch_id,
            "batch_size": int(batch_size),
            "batch_index": int(batch_index),
            "queue_wait_ms": float(timing.get("queue_wait_ms") or 0.0),
            "batch_wait_ms": float(timing.get("batch_wait_ms") or 0.0),
            "tree_forward_batch_kind": tree_forward_batch_kind,
        }
    )
    return replace(response, timing=timing)


def _guard_tree_forward_root_choice(
    model: CausalLMRunner,
    request: TreeVerifyRequest,
    nodes_by_id: dict[int, CandidateNode],
    children_by_parent: dict[int | None, list[CandidateNode]],
    choice_specs: list[tuple[int | None, int, int]],
    *,
    start_index: int,
) -> tuple[list[tuple[int | None, int, int]], list[dict[str, Any]]]:
    """Guard tree attention root choice with a cheap safe check when it looks impossible.

    If tree attention predicts a root token that is not among draft root candidates,
    it may be a legitimate mismatch or a backend mask issue. We run one safe greedy
    root query only in that anomaly case. When safe root disagrees and is in the
    tree, fall back to the batched verifier for the whole proposal.
    """
    root_choice = next((choice for choice in choice_specs if choice[0] is None), None)
    root_children = children_by_parent.get(None, [])
    if root_choice is None or not root_children:
        return choice_specs, []
    root_token = int(root_choice[2])
    root_candidate_tokens = {int(child.token_id) for child in root_children}
    force_root_guard = bool(request.metadata.get("force_root_guard", False))
    if root_token in root_candidate_tokens and not force_root_guard:
        return choice_specs, []

    precomputed_guard = dict(request.metadata.get("precomputed_root_guard_event") or {})
    if precomputed_guard:
        safe_root_token = int(precomputed_guard["token_id"])
        guard_event = {
            **precomputed_guard,
            "index": int(start_index),
            "kind": "tree_root_guard",
            "tree_forward_root_token_id": root_token,
        }
    else:
        safe_root_token, guard_event = _timed_greedy_next_token(
            model,
            list(request.prefix_ids),
            index=start_index,
            kind="tree_root_guard",
            metadata={"tree_forward_root_token_id": root_token},
        )
    events = [guard_event]
    if precomputed_guard and safe_root_token != root_token:
        confirmed_root_token, confirm_event = _timed_greedy_next_token(
            model,
            list(request.prefix_ids),
            index=start_index + 1,
            kind="tree_root_guard_confirm",
            metadata={
                "tree_forward_root_token_id": root_token,
                "precomputed_root_guard_token_id": safe_root_token,
                "precomputed_root_guard_disagreed": True,
            },
        )
        events.append(confirm_event)
        safe_root_token = confirmed_root_token
    if safe_root_token == root_token:
        return choice_specs, events

    if safe_root_token not in root_candidate_tokens:
        guarded_specs = [
            (parent_id, prefix_len, safe_root_token if parent_id is None else target_token)
            for parent_id, prefix_len, target_token in choice_specs
        ]
        return guarded_specs, events

    parent_ids = list(children_by_parent)
    parent_prefixes = [
        [*request.prefix_ids, *_path_tokens(parent_id, nodes_by_id)]
        for parent_id in parent_ids
    ]
    target_tokens, fallback_events = _timed_tree_choice_tokens(
        model,
        parent_prefixes,
        parent_ids=parent_ids,
        start_index=start_index + len(events),
    )
    for event in fallback_events:
        metadata = dict(event.get("metadata") or {})
        metadata["fallback_reason"] = "tree_root_guard_mismatch"
        metadata["tree_forward_root_token_id"] = root_token
        metadata["safe_root_token_id"] = safe_root_token
        event["metadata"] = metadata
    fallback_specs = [
        (parent_id, len(parent_prefix), target_token)
        for parent_id, parent_prefix, target_token in zip(parent_ids, parent_prefixes, target_tokens)
    ]
    return fallback_specs, [*events, *fallback_events]


def _timed_tree_choice_tokens(
    model: CausalLMRunner,
    prefix_ids_batch: list[list[int]],
    *,
    parent_ids: list[int | None],
    start_index: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    """对 tree parent prefixes 做 batched greedy 查询，并记录真实 forward 事件。

    支持 batch next-token 的 backend 只记录一次 batched tree choice forward；
    不支持的测试/降级 backend 保留逐 prefix 事件，避免 timing 把多次调用伪装成一次。
    """
    if not prefix_ids_batch:
        return [], []
    capabilities = model.backend_capabilities()
    if capabilities.supports_batched_next_token:
        start_ns = perf_counter_ns()
        token_ids = [int(token_id) for token_id in model.greedy_next_tokens(prefix_ids_batch)]
        end_ns = perf_counter_ns()
        return token_ids, [
            {
                "index": int(start_index),
                "kind": "tree_choice_batch",
                "batch_size": len(prefix_ids_batch),
                "prefix_lens": [len(prefix_ids) for prefix_ids in prefix_ids_batch],
                "parent_node_ids": list(parent_ids),
                "token_ids": list(token_ids),
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": _duration_ms(start_ns, end_ns),
            }
        ]

    token_ids: list[int] = []
    events: list[dict[str, Any]] = []
    for offset, (prefix_ids, parent_id) in enumerate(zip(prefix_ids_batch, parent_ids)):
        token_id, event = _timed_greedy_next_token(
            model,
            prefix_ids,
            index=start_index + offset,
            kind="tree_choice",
            metadata={"parent_node_id": parent_id},
        )
        token_ids.append(token_id)
        events.append(event)
    return token_ids, events


def _target_timing(forward_events: list[dict[str, Any]]) -> dict[str, Any]:
    """汇总 target tree forward 明细，供 A100 response timing 回传。"""
    return {
        "target_tree_forward_events": [dict(event) for event in forward_events],
        "target_tree_forward_total_ms": sum(float(event["duration_ms"]) for event in forward_events),
        "target_tree_forward_event_count": len(forward_events),
        "target_tree_choice_forward_count": sum(
            int(event.get("choice_count", event.get("batch_size", 1)))
            for event in forward_events
            if str(event.get("kind", "")).startswith("tree_choice")
            or str(event.get("kind", "")).startswith("tree_attention")
            or str(event.get("kind", "")).startswith("tree_forward")
        ),
        "kv_events": [],
    }


def _duration_ms(start_ns: int, end_ns: int) -> float:
    """把 perf_counter_ns 差值转换为毫秒。"""
    return (int(end_ns) - int(start_ns)) / 1_000_000


def _tree_forward_nodes(request: TreeVerifyRequest) -> list[TreeForwardNode]:
    """把 verification schema 的 CandidateNode 转成模型层 tree_forward node。"""
    return [
        TreeForwardNode(
            node_id=int(node.node_id),
            parent_id=None if node.parent_id is None else int(node.parent_id),
            token_id=int(node.token_id),
            depth=int(node.depth),
        )
        for node in request.tree.nodes
    ]
