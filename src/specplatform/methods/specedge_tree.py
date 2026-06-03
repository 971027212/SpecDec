from __future__ import annotations

"""SpecEdge tree speculative decoding 策略。

这一层只负责把 draft runner 的 tree 结果包装成 CandidateProposal，并把
tree verifier 的事实转换成 session 可写回的 token；不调用 verifier、不写 session。
"""

from dataclasses import dataclass, replace
from typing import Any

from specplatform.core import (
    AcceptResult,
    CandidateNode,
    CandidateProposal,
    CandidateTree,
    DraftBudget,
    DraftJob,
    PlanHints,
    RuntimeContext,
    VerificationResult,
)
from specplatform.draft import TreeDraftGeneration
from specplatform.methods.base import (
    AcceptancePolicy,
    CandidateStrategy,
    PlanningPolicy,
    ProactiveDraftPolicy,
    ReconcilePolicy,
    ReconcileResult,
)
from specplatform.methods.specedge_official import OfficialSpecEdgeDraftState, OfficialTreeStatus


@dataclass
class SpecEdgeTreeCandidateStrategy(CandidateStrategy):
    """把确定性 top-k draft tree 包装成 tree CandidateProposal。"""

    proposal_prefix: str = "specedge-tree"
    default_max_budget: int = 16
    default_max_branch_width: int = 2

    def propose(
        self,
        session: Any,
        draft_runner: Any,
        budget: DraftBudget,
        context: RuntimeContext,
    ) -> CandidateProposal:
        """生成一个 tree proposal；不调用 verifier，不写 session。"""
        max_depth = min(int(budget.max_tokens), int(session.remaining_tokens))
        max_branches = _config_int(
            context,
            "max_branch_width",
            int(budget.max_branches) if int(budget.max_branches) > 1 else self.default_max_branch_width,
        )
        max_nodes = _config_int(context, "max_budget", self.default_max_budget)
        generation: TreeDraftGeneration = draft_runner.generate_tree(
            prefix_ids=session.prefix_ids,
            max_depth=max_depth,
            max_branches=max_branches,
            max_nodes=max_nodes,
            request_id=session.request_id,
            metadata={
                "draft_budget": {
                    "max_tokens": budget.max_tokens,
                    "max_branches": budget.max_branches,
                    "timeout_ms": budget.timeout_ms,
                    "max_budget": max_nodes,
                }
            },
        )
        metadata = dict(generation.metadata)
        metadata["prefix_ids"] = list(session.prefix_ids)
        metadata["remaining_tokens"] = session.remaining_tokens
        metadata["allow_bonus"] = False if _config_bool(context, "disable_bonus", False) else max_depth < session.remaining_tokens
        metadata["method"] = "specedge_tree"
        metadata["tree_node_count"] = len(generation.tree.nodes)
        metadata["tree_max_depth"] = max((node.depth for node in generation.tree.nodes), default=0)
        metadata["force_root_guard"] = _config_bool(context, "force_root_guard", False)
        runner_id = str(metadata.get("runner_id") or "draft")
        proposal_id = f"{self.proposal_prefix}:{session.request_id}:{session.step_idx}:{runner_id}"

        return CandidateProposal(
            proposal_id=proposal_id,
            request_id=session.request_id,
            worker_id=metadata.get("runner_id"),
            shape="tree",
            tokens=[node.token_id for node in generation.tree.nodes],
            tree=generation.tree,
            draft_length=len(generation.tree.nodes),
            timing=dict(generation.timing),
            metadata=metadata,
        )


@dataclass
class SpecEdgeOfficialCandidateStrategy(CandidateStrategy):
    """Official-style SpecEdge BatchTree/BatchGraphEngine proposal strategy."""

    state: OfficialSpecEdgeDraftState
    proposal_prefix: str = "specedge-official"
    default_max_budget: int = 20
    default_max_branch_width: int = 8

    def propose(
        self,
        session: Any,
        draft_runner: Any,
        budget: DraftBudget,
        context: RuntimeContext,
    ) -> CandidateProposal:
        """Single-job compatibility wrapper for runtimes that do not batch draft jobs."""
        runner_id = str(getattr(draft_runner, "runner_id", None) or getattr(getattr(draft_runner, "model", None), "runner_id", "draft"))
        job = DraftJob(
            request_id=str(session.request_id),
            worker_id=runner_id,
            budget=budget,
        )
        return self.propose_batch(
            jobs=[job],
            sessions_by_id={str(session.request_id): session},
            draft_runners={runner_id: draft_runner},
            context=context,
        )[0]

    def propose_batch(
        self,
        *,
        jobs: list[Any],
        sessions_by_id: dict[str, Any],
        draft_runners: dict[str, Any],
        context: RuntimeContext,
    ) -> list[CandidateProposal]:
        """Generate official BatchTree proposals, grouped by backing model."""
        if not jobs:
            return []
        proposals_by_index: dict[int, CandidateProposal] = {}
        pending_jobs: list[tuple[int, Any]] = []
        grow_jobs: list[tuple[int, Any]] = []
        for job_index, job in enumerate(jobs):
            session = sessions_by_id[job.request_id]
            slot = self.state.slots.get(str(session.request_id))
            if (
                _config_bool(context, "official_reuse_state_tree", True)
                and slot is not None
                and list(slot.prefix_ids) == list(session.prefix_ids)
                and (slot.nodes or slot.needs_prefix_tail_forward)
            ):
                runner = draft_runners.get(job.worker_id)
                grower = getattr(runner, "grow_official_tree_batch", None)
                if _config_bool(context, "official_grow_reused_state_tree", True) and callable(grower):
                    grow_jobs.append((job_index, job))
                elif slot.nodes:
                    metadata = _official_reused_state_metadata(
                        session=session,
                        job=job,
                        slot=slot,
                        context=context,
                    )
                    proposal_id = f"{self.proposal_prefix}:{session.request_id}:{session.step_idx}:{job.worker_id}:reused"
                    proposals_by_index[job_index] = slot.to_candidate_proposal(
                        proposal_id=proposal_id,
                        allow_bonus=bool(metadata["allow_bonus"]),
                        metadata=metadata,
                    )
                else:
                    pending_jobs.append((job_index, job))
            else:
                pending_jobs.append((job_index, job))
        for group in _official_draft_job_groups(grow_jobs, draft_runners):
            group_requests: list[dict[str, Any]] = []
            for _job_index, job in group:
                session = sessions_by_id[job.request_id]
                slot = self.state.slot(str(session.request_id))
                max_depth = min(int(job.budget.max_tokens), int(session.remaining_tokens))
                max_branches = _config_int(
                    context,
                    "max_branch_width",
                    int(job.budget.max_branches)
                    if int(job.budget.max_branches) > 1
                    else self.default_max_branch_width,
                )
                max_nodes = _config_int(context, "max_budget", self.default_max_budget)
                group_requests.append(
                    {
                        "prefix_ids": list(session.prefix_ids),
                        "tree": slot.to_candidate_tree().to_dict(),
                        "tree_node_statuses": {str(node_id): int(status) for node_id, status in slot.statuses.items()},
                        "needs_prefix_tail_forward": bool(slot.needs_prefix_tail_forward),
                        "draft_batch_index": slot.draft_batch_index,
                        "max_depth": max_depth,
                        "max_branches": max_branches,
                        "max_nodes": max_nodes,
                        "request_id": session.request_id,
                        "runner_id": str(job.worker_id),
                        "metadata": {
                            "draft_budget": {
                                "max_tokens": job.budget.max_tokens,
                                "max_branches": job.budget.max_branches,
                                "timeout_ms": job.budget.timeout_ms,
                                "max_budget": max_nodes,
                            },
                            "runner_id": str(job.worker_id),
                            "official_specedge_state": True,
                            "official_batch_tree": True,
                            "official_state_reused_tree": True,
                            "official_state_grow_requested": True,
                        },
                    }
                )

            runner = draft_runners[group[0][1].worker_id]
            grower = getattr(runner, "grow_official_tree_batch")
            generations = grower(group_requests)
            if len(generations) != len(group):
                raise ValueError("Official SpecEdge state grow returned a different batch size.")

            for generation, (job_index, job), request_spec in zip(generations, group, group_requests):
                session = sessions_by_id[job.request_id]
                metadata = _official_generation_metadata(
                    generation,
                    session=session,
                    context=context,
                    request_spec=request_spec,
                    group_size=len(group),
                )
                metadata["official_state_reused_tree"] = True
                metadata["official_state_grown_tree"] = True
                slot = self.state.ensure_request(
                    session.request_id,
                    list(session.prefix_ids),
                    draft_worker_id=str(job.worker_id),
                    draft_batch_index=_request_draft_batch_index(request_spec, metadata),
                )
                slot.replace_tree(
                    generation.tree,
                    prefix_ids=list(session.prefix_ids),
                    statuses=_tree_statuses_from_metadata(metadata),
                )
                slot.needs_prefix_tail_forward = bool(metadata.get("official_needs_prefix_tail_forward", False))
                proposal_id = f"{self.proposal_prefix}:{session.request_id}:{session.step_idx}:{job.worker_id}:grown"
                proposals_by_index[job_index] = slot.to_candidate_proposal(
                    proposal_id=proposal_id,
                    allow_bonus=bool(metadata["allow_bonus"]),
                    metadata=metadata,
                )
        for group in _official_draft_job_groups(pending_jobs, draft_runners):
            group_requests: list[dict[str, Any]] = []
            for _job_index, job in group:
                session = sessions_by_id[job.request_id]
                slot = self.state.ensure_request(
                    session.request_id,
                    list(session.prefix_ids),
                    draft_worker_id=str(job.worker_id),
                )
                max_depth = min(int(job.budget.max_tokens), int(session.remaining_tokens))
                max_branches = _config_int(
                    context,
                    "max_branch_width",
                    int(job.budget.max_branches)
                    if int(job.budget.max_branches) > 1
                    else self.default_max_branch_width,
                )
                max_nodes = _config_int(context, "max_budget", self.default_max_budget)
                group_requests.append(
                    {
                        "prefix_ids": list(session.prefix_ids),
                        "max_depth": max_depth,
                        "max_branches": max_branches,
                        "max_nodes": max_nodes,
                        "draft_batch_index": slot.draft_batch_index,
                        "request_id": session.request_id,
                        "runner_id": str(job.worker_id),
                        "metadata": {
                            "draft_budget": {
                                "max_tokens": job.budget.max_tokens,
                                "max_branches": job.budget.max_branches,
                                "timeout_ms": job.budget.timeout_ms,
                                "max_budget": max_nodes,
                            },
                            "runner_id": str(job.worker_id),
                            "official_specedge_state": True,
                            "official_batch_tree": True,
                            "official_draft_batch_index": slot.draft_batch_index,
                        },
                    }
                )

            runner = draft_runners[group[0][1].worker_id]
            batch_generator = getattr(runner, "generate_tree_batch", None)
            if callable(batch_generator):
                generations = batch_generator(group_requests)
            else:
                generations = []
                for request, (_job_index, job) in zip(group_requests, group):
                    fallback_request = dict(request)
                    fallback_request.pop("runner_id", None)
                    generations.append(draft_runners[job.worker_id].generate_tree(**fallback_request))
            if len(generations) != len(group):
                raise ValueError("Official SpecEdge batch draft returned a different batch size.")

            for generation, (job_index, job), request_spec in zip(generations, group, group_requests):
                session = sessions_by_id[job.request_id]
                metadata = _official_generation_metadata(
                    generation,
                    session=session,
                    context=context,
                    request_spec=request_spec,
                    group_size=len(group),
                )
                slot = self.state.ensure_request(
                    session.request_id,
                    list(session.prefix_ids),
                    draft_worker_id=str(job.worker_id),
                    draft_batch_index=_request_draft_batch_index(request_spec, metadata),
                )
                slot.replace_tree(
                    generation.tree,
                    prefix_ids=list(session.prefix_ids),
                    statuses=_tree_statuses_from_metadata(metadata),
                )
                slot.needs_prefix_tail_forward = bool(metadata.get("official_needs_prefix_tail_forward", False))
                proposal_id = f"{self.proposal_prefix}:{session.request_id}:{session.step_idx}:{job.worker_id}"
                proposals_by_index[job_index] = slot.to_candidate_proposal(
                    proposal_id=proposal_id,
                    allow_bonus=bool(metadata["allow_bonus"]),
                    metadata=metadata,
                )
        return [proposals_by_index[index] for index in range(len(jobs))]


@dataclass
class SpecEdgeProactiveDraftPolicy(ProactiveDraftPolicy):
    """SpecEdge proactive single-head draft 原型。"""

    proposal_prefix: str = "specedge-proactive"
    default_max_depth: int = 2
    default_max_branch_width: int = 8
    default_max_budget: int = 20

    def propose_proactive(
        self,
        session: Any,
        proposal: CandidateProposal,
        draft_runner: Any,
        context: RuntimeContext,
    ) -> CandidateProposal | None:
        """沿初始 tree 中累计 logprob 最高的 leaf path 继续扩展。"""
        if proposal.shape != "tree" or proposal.tree is None:
            return None
        prefix_ids = [int(token_id) for token_id in proposal.metadata.get("prefix_ids", [])]
        if not prefix_ids:
            return None
        expansion_path = _best_leaf_path_tokens(proposal)
        proactive_prefix = [*prefix_ids, *expansion_path]
        remaining_tokens = int(session.max_new_tokens) - (len(proactive_prefix) - len(session.prompt_ids))
        if remaining_tokens <= 0:
            return None

        max_depth = min(
            _config_int(context, "proactive_max_depth", self.default_max_depth),
            remaining_tokens,
        )
        max_branches = _config_int(context, "proactive_branch_width", self.default_max_branch_width)
        max_nodes = _config_int(context, "proactive_max_budget", self.default_max_budget)
        generation: TreeDraftGeneration = draft_runner.generate_tree(
            prefix_ids=proactive_prefix,
            max_depth=max_depth,
            max_branches=max_branches,
            max_nodes=max_nodes,
            request_id=session.request_id,
            metadata={
                "proactive": True,
                "parent_proposal_id": proposal.proposal_id,
                "expansion_head_tokens": list(expansion_path),
            },
        )
        proposal_id = f"{self.proposal_prefix}:{session.request_id}:{session.step_idx}:{proposal.proposal_id}"
        metadata = dict(generation.metadata)
        metadata.update(
            {
                "prefix_ids": proactive_prefix,
                "remaining_tokens": remaining_tokens,
                "allow_bonus": False if _config_bool(context, "disable_bonus", False) else max_depth < remaining_tokens,
                "method": "specedge_pipeline",
                "tree_node_count": len(generation.tree.nodes),
                "tree_max_depth": max((node.depth for node in generation.tree.nodes), default=0),
                "proactive_base_tokens": list(expansion_path),
                "force_root_guard": _config_bool(context, "proactive_force_root_guard", True),
            }
        )
        return CandidateProposal(
            proposal_id=proposal_id,
            request_id=session.request_id,
            worker_id=metadata.get("runner_id"),
            shape="tree",
            tokens=[node.token_id for node in generation.tree.nodes],
            tree=generation.tree,
            draft_length=len(generation.tree.nodes),
            timing=dict(generation.timing),
            metadata=metadata,
        )


@dataclass
class SpecEdgeOfficialProactiveDraftPolicy(ProactiveDraftPolicy):
    """Official-style proactive draft using POST_CANDIDATE state."""

    state: OfficialSpecEdgeDraftState
    proposal_prefix: str = "specedge-official-proactive"
    default_max_depth: int = 2
    default_max_branch_width: int = 8
    default_max_budget: int = 20
    default_leaf_beams: int = 8
    default_root_top_k: int = 1024
    decay_factor: float = -0.05129329438755058

    def propose_proactive(
        self,
        session: Any,
        proposal: CandidateProposal,
        draft_runner: Any,
        context: RuntimeContext,
    ) -> CandidateProposal | None:
        if not bool(proposal.metadata.get("official_specedge_state", False)):
            return None
        slot = self.state.slot(proposal.request_id)
        if not slot.nodes:
            return None
        max_leaf_beams = _config_int(context, "proactive_leaf_beams", self.default_leaf_beams)
        root_top_k = _config_int(context, "proactive_root_top_k", self.default_root_top_k)
        proactive_decay_factor = float(context.method_config.get("proactive_decay_factor", self.decay_factor))
        max_branches = _config_int(context, "proactive_branch_width", self.default_max_branch_width)
        max_nodes = _config_int(context, "proactive_max_budget", self.default_max_budget)
        graph_result = _official_proactive_graph_result(
            slot=slot,
            session=session,
            proposal=proposal,
            draft_runner=draft_runner,
            context=context,
            max_depth=_config_int(context, "proactive_max_depth", self.default_max_depth),
            max_branches=max_branches,
            max_nodes=max_nodes,
            max_leaf_beams=max_leaf_beams,
            root_top_k=root_top_k,
            decay_factor=proactive_decay_factor,
        )
        if graph_result is not None:
            candidate = _official_proactive_candidate_from_graph_result(slot, graph_result)
            generation_metadata = dict(graph_result.get("metadata") or {})
            subtree = graph_result["subtree"]
            subtree_statuses = {
                int(node_id): OfficialTreeStatus(int(status))
                for node_id, status in dict(graph_result.get("subtree_statuses") or {}).items()
            }
            root_status = OfficialTreeStatus(
                int(graph_result.get("root_status", int(OfficialTreeStatus.POST_CANDIDATE)))
            )
        else:
            candidate = _best_official_proactive_root(
                slot,
                draft_runner,
                max_leaf_beams=max_leaf_beams,
                root_top_k=root_top_k,
                decay_factor=proactive_decay_factor,
            )
            generation_metadata = None
            subtree = None
            subtree_statuses = None
            root_status = OfficialTreeStatus.POST_CANDIDATE
        if candidate is None:
            return None
        parent_node, root_token_id, root_logprob, leaf_path = candidate
        prospective_new_tokens = (len(slot.prefix_ids) + len(leaf_path) + 1) - len(session.prompt_ids)
        subtree_depth = min(
            _config_int(context, "proactive_max_depth", self.default_max_depth),
            max(0, int(session.max_new_tokens) - int(prospective_new_tokens)),
        )
        proactive_prefix = [*slot.prefix_ids, *leaf_path, int(root_token_id)]
        if subtree is not None and generation_metadata is not None:
            pass
        elif subtree_depth > 0 and max_nodes > 0:
            generation: TreeDraftGeneration = draft_runner.generate_tree(
                prefix_ids=proactive_prefix,
                max_depth=subtree_depth,
                max_branches=max_branches,
                max_nodes=max_nodes,
                request_id=session.request_id,
                metadata={
                    "official_proactive": True,
                    "parent_proposal_id": proposal.proposal_id,
                    "runner_id": proposal.worker_id,
                },
            )
            subtree = generation.tree
            subtree_statuses = _tree_statuses_from_metadata(dict(generation.metadata))
            generation_metadata = dict(generation.metadata)
        else:
            subtree = CandidateTree(root_prefix_len=len(proactive_prefix), nodes=[])
            subtree_statuses = {}
            generation_metadata = {}

        record = slot.add_proactive_subtree(
            parent_node_id=int(parent_node.node_id),
            root_token_id=int(root_token_id),
            root_logprob=float(root_logprob),
            subtree=subtree,
            subtree_statuses=subtree_statuses,
            root_status=root_status,
        )
        proposal_id = f"{self.proposal_prefix}:{session.request_id}:{session.step_idx}:{proposal.proposal_id}"
        metadata = {
            **generation_metadata,
            "method": "specedge_official",
            "official_specedge_state": True,
            "official_proactive": True,
            "official_proactive_parent_node_id": int(record.parent_node_id),
            "official_proactive_root_node_id": int(record.root_node_id),
            "official_proactive_root_token_id": int(record.root_token_id),
            "official_proactive_node_ids": list(record.node_ids),
            "official_proactive_leaf_path": list(leaf_path),
            "official_proactive_prefix": list(proactive_prefix),
            "tree_node_count": len(slot.nodes),
            "tree_snapshot": slot.to_candidate_tree().to_dict(),
        }
        return slot.to_candidate_proposal(
            proposal_id=proposal_id,
            allow_bonus=False,
            metadata=metadata,
        )


@dataclass
class SpecEdgeOfficialReconcilePolicy(ReconcilePolicy):
    """Official path keeps reuse in method state, not runtime proposal cache."""

    state: OfficialSpecEdgeDraftState

    def reconcile(
        self,
        session: Any,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        accept_result: AcceptResult,
        proactive_proposal: CandidateProposal | None,
        context: RuntimeContext,
    ) -> ReconcileResult:
        del proposal, verification_result, context
        slot = self.state.slot(session.request_id)
        aligned = list(slot.prefix_ids) == list(session.prefix_ids)
        return ReconcileResult(
            reused_proposal=None,
            reused_token_count=len(slot.nodes),
            discarded_token_count=0 if slot.nodes else (0 if proactive_proposal is None else proactive_proposal.draft_length),
            aligned=aligned,
            metadata={
                "reason": "official_state_reuse" if slot.nodes else "official_state_no_reuse",
                "official_state_aligned": aligned,
                "official_state_prefix_len": len(slot.prefix_ids),
                "official_state_tree_node_count": len(slot.nodes),
                "official_state_statuses": {str(node_id): int(status) for node_id, status in slot.statuses.items()},
                "accepted_output_tokens": list(accept_result.output_token_ids),
                "proactive_proposal_id": None if proactive_proposal is None else proactive_proposal.proposal_id,
            },
        )


@dataclass
class SpecEdgeReconcilePolicy(ReconcilePolicy):
    """根据写回后的 committed prefix 判断 proactive tree 能否复用。"""

    def reconcile(
        self,
        session: Any,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        accept_result: AcceptResult,
        proactive_proposal: CandidateProposal | None,
        context: RuntimeContext,
    ) -> ReconcileResult:
        del proposal, verification_result, context
        if proactive_proposal is None:
            return ReconcileResult(metadata={"reason": "no_proactive_proposal"})
        proactive_prefix = [int(token_id) for token_id in proactive_proposal.metadata.get("prefix_ids", [])]
        committed_prefix = list(session.prefix_ids)
        if proactive_prefix == committed_prefix:
            return ReconcileResult(
                reused_proposal=proactive_proposal,
                reused_token_count=int(proactive_proposal.draft_length),
                aligned=True,
                metadata={
                    "reason": "prefix_aligned",
                    "accepted_output_tokens": list(accept_result.output_token_ids),
                    "proactive_prefix_len": len(proactive_prefix),
                },
            )
        subtree = _reuse_subtree_after_committed_token(proactive_proposal, proactive_prefix, committed_prefix)
        if subtree is not None:
            reused_proposal, reused_count, discarded_count = subtree
            return ReconcileResult(
                reused_proposal=reused_proposal,
                reused_token_count=reused_count,
                discarded_token_count=discarded_count,
                aligned=True,
                metadata={
                    "reason": "subtree_aligned",
                    "accepted_output_tokens": list(accept_result.output_token_ids),
                    "proactive_prefix_len": len(proactive_prefix),
                    "committed_prefix_len": len(committed_prefix),
                },
            )
        return ReconcileResult(
            discarded_token_count=int(proactive_proposal.draft_length),
            aligned=False,
            metadata={
                "reason": "prefix_mismatch",
                "accepted_output_tokens": list(accept_result.output_token_ids),
                "proactive_prefix_len": len(proactive_prefix),
                "committed_prefix_len": len(committed_prefix),
            },
        )


@dataclass
class SpecEdgePipelinePlanningPolicy(PlanningPolicy):
    """基于 EMA 的 draft depth 提示策略。"""

    min_depth: int = 1
    max_depth: int = 8
    initial_depth: int = 2
    ema_alpha: float = 0.3
    official_state: OfficialSpecEdgeDraftState | None = None
    _draft_ms_per_token: float | None = None
    _server_verify_ms: float | None = None
    _network_residual_ms: float | None = None

    def observe(
        self,
        *,
        draft_ms_per_token: float | None = None,
        server_verify_ms: float | None = None,
        network_residual_ms: float | None = None,
    ) -> None:
        """更新 EMA 观测值，供下一轮 scheduler hint 使用。"""
        self._draft_ms_per_token = _ema(self._draft_ms_per_token, draft_ms_per_token, self.ema_alpha)
        self._server_verify_ms = _ema(self._server_verify_ms, server_verify_ms, self.ema_alpha)
        self._network_residual_ms = _ema(self._network_residual_ms, network_residual_ms, self.ema_alpha)

    def plan(
        self,
        active_sessions: list[Any],
        resources: Any,
        history: Any,
        context: RuntimeContext,
    ) -> PlanHints:
        del history, context
        depth = self._target_depth()
        worker_preferences = self._official_worker_preferences(active_sessions, resources)
        metadata: dict[str, Any] = {}
        if worker_preferences:
            metadata["official_state_worker_affinity_count"] = len(worker_preferences)

        return PlanHints(
            draft_lengths={session.request_id: depth for session in active_sessions},
            worker_preferences=worker_preferences,
            metadata=metadata,
        )

    def _target_depth(self) -> int:
        if not self._draft_ms_per_token or self._draft_ms_per_token <= 0:
            return int(self.initial_depth)
        target_ms = float(self._server_verify_ms or 0.0) + max(0.0, float(self._network_residual_ms or 0.0))
        if target_ms <= 0:
            return int(self.initial_depth)
        return max(self.min_depth, min(self.max_depth, int(round(target_ms / self._draft_ms_per_token))))

    def _official_worker_preferences(self, active_sessions: list[Any], resources: Any) -> dict[str, str]:
        if self.official_state is None:
            return {}
        available_workers = set(_worker_ids_from_resources(resources))
        preferences: dict[str, str] = {}
        for session in active_sessions:
            request_id = str(session.request_id)
            slot = self.official_state.slots.get(request_id)
            if slot is None:
                continue
            if list(slot.prefix_ids) != list(session.prefix_ids):
                continue
            worker_id = str(slot.draft_worker_id)
            if worker_id in available_workers:
                preferences[request_id] = worker_id
        return preferences


@dataclass
class SpecEdgeTreeAcceptancePolicy(AcceptancePolicy):
    """根据 tree verifier 的 accepted path 产出写回 token。"""

    def accept(
        self,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        context: RuntimeContext,
    ) -> AcceptResult:
        """消费 verifier result，不调用 verifier，不写 session。"""
        if proposal.proposal_id != verification_result.proposal_id:
            raise ValueError("VerificationResult does not belong to the given proposal.")
        if proposal.request_id != verification_result.request_id:
            raise ValueError("VerificationResult request_id does not match proposal.")
        if proposal.shape != "tree" or proposal.tree is None:
            raise ValueError("SpecEdgeTreeAcceptancePolicy requires a tree proposal.")

        nodes_by_id = proposal.tree.nodes_by_id()
        accepted_node_ids = [
            int(node_id)
            for node_id in verification_result.payload.get("accepted_node_ids", [])
        ]
        rejected_node_ids = {
            int(node_id)
            for node_id in verification_result.payload.get("rejected_node_ids", [])
        }
        accepted_tokens = [int(nodes_by_id[node_id].token_id) for node_id in accepted_node_ids]
        rejected_tokens = [
            int(node.token_id)
            for node in proposal.tree.nodes
            if node.node_id in rejected_node_ids
        ]
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
                "accepted_node_ids": accepted_node_ids,
                "rejected_node_ids": sorted(rejected_node_ids),
                "accepted_count": len(accepted_tokens),
                "rejected_count": len(rejected_tokens),
                "has_bonus": bonus_token is not None,
                "target_choices": list(verification_result.payload.get("target_choices", [])),
            },
        )


@dataclass
class SpecEdgeOfficialAcceptancePolicy(AcceptancePolicy):
    """Official SpecEdge acceptance with deferred BatchTree gather/reorder commit."""

    state: OfficialSpecEdgeDraftState
    base_policy: SpecEdgeTreeAcceptancePolicy = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.base_policy is None:
            self.base_policy = SpecEdgeTreeAcceptancePolicy()

    def accept(
        self,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        context: RuntimeContext,
    ) -> AcceptResult:
        result = self.base_policy.accept(proposal, verification_result, context)
        if not bool(proposal.metadata.get("official_specedge_state", False)):
            return result
        return replace(
            result,
            metadata={
                **dict(result.metadata),
                "official_specedge_state": True,
                "official_acceptance_pending": True,
            },
        )

    def commit_acceptance(
        self,
        proposal: CandidateProposal,
        verification_result: VerificationResult,
        accept_result: AcceptResult,
        context: RuntimeContext,
        *,
        draft_runners: dict[str, Any] | None = None,
    ) -> AcceptResult:
        del verification_result, context
        if not bool(proposal.metadata.get("official_specedge_state", False)):
            return accept_result
        slot = self.state.slot(proposal.request_id)
        reorder = slot.apply_acceptance(
            list(accept_result.metadata.get("accepted_node_ids", [])),
            bonus_token=accept_result.bonus_token,
        )
        model_commit = _commit_official_draft_model_state(
            proposal=proposal,
            reorder=reorder,
            slot=slot,
            draft_runners=draft_runners,
        )
        return replace(
            accept_result,
            metadata={
                **dict(accept_result.metadata),
                "official_acceptance_pending": False,
                "official_reorder": {
                    "request_id": reorder.request_id,
                    "emitted_tokens": list(reorder.emitted_tokens),
                    "source_seq_indices": list(reorder.source_seq_indices),
                    "dest_seq_indices": list(reorder.dest_seq_indices),
                    "bonus_token": reorder.bonus_token,
                    "reused_proactive_tree": reorder.reused_proactive_tree,
                    "retained_tree_node_count": reorder.retained_tree_node_count,
                },
                "official_model_commit": model_commit,
                "official_prefix_ids": list(slot.prefix_ids),
                "official_tree_cleared": len(slot.nodes) == 0,
            },
        )


def _config_int(context: RuntimeContext, key: str, default: int) -> int:
    """按 method_config -> run_config 的优先级读取整数配置。"""
    raw = context.method_config.get(key, context.run_config.get(key, default))
    return int(raw)


def _config_bool(context: RuntimeContext, key: str, default: bool) -> bool:
    raw = context.method_config.get(key, context.run_config.get(key, default))
    return bool(raw)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _request_draft_batch_index(request_spec: dict[str, Any], metadata: dict[str, Any]) -> int | None:
    request_index = request_spec.get("draft_batch_index")
    if request_index is not None:
        return int(request_index)
    return _optional_int(metadata.get("official_draft_batch_index", metadata.get("batch_index")))


def _worker_ids_from_resources(resources: Any) -> list[str]:
    if isinstance(resources, dict):
        return [str(worker_id) for worker_id in resources.get("draft_worker_ids", [])]
    return [str(worker_id) for worker_id in getattr(resources, "draft_worker_ids", [])]


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


def _official_draft_job_groups(job_items: list[Any], draft_runners: dict[str, Any]) -> list[list[tuple[int, Any]]]:
    """Group jobs by backing model so one BatchGraphEngine owns each draft batch."""
    groups: dict[int, list[tuple[int, Any]]] = {}
    order: list[int] = []
    for fallback_index, item in enumerate(job_items):
        if isinstance(item, tuple) and len(item) == 2:
            index, job = item
        else:
            index, job = fallback_index, item
        runner = draft_runners[job.worker_id]
        model = getattr(runner, "model", runner)
        key = id(model)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((index, job))
    return [groups[key] for key in order]


def _official_generation_metadata(
    generation: TreeDraftGeneration,
    *,
    session: Any,
    context: RuntimeContext,
    request_spec: dict[str, Any],
    group_size: int,
) -> dict[str, Any]:
    metadata = dict(generation.metadata)
    metadata["prefix_ids"] = list(session.prefix_ids)
    metadata["remaining_tokens"] = int(session.remaining_tokens)
    metadata["allow_bonus"] = (
        False
        if _config_bool(context, "disable_bonus", False)
        else int(request_spec["max_depth"]) < int(session.remaining_tokens)
    )
    metadata["method"] = "specedge_official"
    metadata["tree_node_count"] = len(generation.tree.nodes)
    metadata["tree_max_depth"] = max((node.depth for node in generation.tree.nodes), default=0)
    metadata["tree_snapshot"] = generation.tree.to_dict()
    metadata["force_root_guard"] = _config_bool(context, "force_root_guard", False)
    metadata["official_specedge_state"] = True
    metadata["official_batch_tree"] = True
    metadata["official_draft_group_size"] = int(group_size)
    return metadata


def _official_reused_state_metadata(
    *,
    session: Any,
    job: Any,
    slot: Any,
    context: RuntimeContext,
) -> dict[str, Any]:
    return {
        "prefix_ids": list(session.prefix_ids),
        "remaining_tokens": int(session.remaining_tokens),
        "allow_bonus": False
        if _config_bool(context, "disable_bonus", False)
        else int(job.budget.max_tokens) < int(session.remaining_tokens),
        "method": "specedge_official",
        "tree_node_count": len(slot.nodes),
        "tree_max_depth": max((node.depth for node in slot.nodes), default=0),
        "tree_snapshot": slot.to_candidate_tree().to_dict(),
        "force_root_guard": _config_bool(context, "force_root_guard", False),
        "official_specedge_state": True,
        "official_batch_tree": True,
        "official_state_reused_tree": True,
        "official_needs_prefix_tail_forward": bool(slot.needs_prefix_tail_forward),
    }


def _tree_statuses_from_metadata(metadata: dict[str, Any]) -> dict[int, OfficialTreeStatus]:
    raw = dict(metadata.get("tree_node_statuses") or {})
    return {
        int(node_id): OfficialTreeStatus(int(status))
        for node_id, status in raw.items()
    }


def _commit_official_draft_model_state(
    *,
    proposal: CandidateProposal,
    reorder: Any,
    slot: Any,
    draft_runners: dict[str, Any] | None,
) -> dict[str, Any]:
    if not draft_runners:
        return {"committed": False, "reason": "no_draft_runners"}
    runner_id = str(proposal.worker_id or "")
    runner = draft_runners.get(runner_id) if runner_id else None
    if runner is None and draft_runners:
        runner = next(iter(draft_runners.values()))
    model = getattr(runner, "model", runner)
    commit = getattr(model, "official_specedge_commit_acceptance", None)
    if not callable(commit):
        return {"committed": False, "reason": "model_has_no_official_commit"}
    metadata = commit(
        request_id=proposal.request_id,
        batch_index=slot.draft_batch_index,
        source_seq_indices=list(reorder.source_seq_indices),
        dest_seq_indices=list(reorder.dest_seq_indices),
        prefix_ids=list(slot.prefix_ids),
        retained_tree=slot.to_candidate_tree().to_dict() if slot.nodes else None,
        reused_proactive_tree=bool(reorder.reused_proactive_tree),
    )
    return {"committed": True, **dict(metadata or {})}


def _official_proactive_graph_result(
    *,
    slot: Any,
    session: Any,
    proposal: CandidateProposal,
    draft_runner: Any,
    context: RuntimeContext,
    max_depth: int,
    max_branches: int,
    max_nodes: int,
    max_leaf_beams: int,
    root_top_k: int,
    decay_factor: float,
) -> dict[str, Any] | None:
    if not _config_bool(context, "official_proactive_graph", True):
        return None
    generator = getattr(draft_runner, "generate_official_proactive", None)
    if not callable(generator):
        return None
    result = generator(
        {
            "prefix_ids": list(slot.prefix_ids),
            "tree": slot.to_candidate_tree().to_dict(),
            "tree_node_statuses": {str(node_id): int(status) for node_id, status in slot.statuses.items()},
            "draft_batch_index": slot.draft_batch_index,
            "max_depth": int(max_depth),
            "max_branches": int(max_branches),
            "max_nodes": int(max_nodes),
            "max_leaf_beams": int(max_leaf_beams),
            "root_top_k": int(root_top_k),
            "decay_factor": float(decay_factor),
            "prompt_len": len(session.prompt_ids),
            "max_new_tokens": int(session.max_new_tokens),
            "request_id": session.request_id,
            "runner_id": proposal.worker_id,
            "metadata": {
                "official_specedge_state": True,
                "official_proactive": True,
                "parent_proposal_id": proposal.proposal_id,
            },
        }
    )
    if not result:
        return None
    if result.get("parent_node_id") is None or result.get("root_token_id") is None:
        return None
    subtree = result.get("subtree")
    if isinstance(subtree, CandidateTree):
        parsed_subtree = subtree
    elif subtree:
        parsed_subtree = CandidateTree.from_dict(dict(subtree))
    else:
        proactive_prefix_len = len(slot.prefix_ids) + len(result.get("leaf_path") or []) + 1
        parsed_subtree = CandidateTree(root_prefix_len=proactive_prefix_len, nodes=[])
    parsed_subtree.validate()
    return {
        **dict(result),
        "subtree": parsed_subtree,
    }


def _official_proactive_candidate_from_graph_result(
    slot: Any,
    result: dict[str, Any],
) -> tuple[CandidateNode, int, float, list[int]] | None:
    try:
        parent_node = slot.node_by_id(int(result["parent_node_id"]))
    except KeyError:
        return None
    return (
        parent_node,
        int(result["root_token_id"]),
        float(result.get("root_logprob", 0.0)),
        [int(token_id) for token_id in result.get("leaf_path", [])],
    )


def _best_official_proactive_root(
    slot: Any,
    draft_runner: Any,
    *,
    max_leaf_beams: int,
    root_top_k: int,
    decay_factor: float,
) -> tuple[CandidateNode, int, float, list[int]] | None:
    """Select the official proactive bonus-token root from tree leaves."""
    if max_leaf_beams <= 0 or root_top_k <= 0:
        return None
    tree = slot.to_candidate_tree()
    children_by_parent = tree.children_by_parent()
    nodes_by_id = tree.nodes_by_id()
    leaves = [node for node in tree.nodes if int(node.node_id) not in children_by_parent]
    if not leaves:
        return None
    leaves.sort(
        key=lambda node: (
            float("-inf") if node.draft_logprob is None else float(node.draft_logprob),
            -int(node.node_id),
        ),
        reverse=True,
    )
    leaves = leaves[: int(max_leaf_beams)]
    leaf_paths = [_path_tokens(leaf, nodes_by_id) for leaf in leaves]
    prefixes = [[*slot.prefix_ids, *path] for path in leaf_paths]
    model = getattr(draft_runner, "model", draft_runner)
    topk_batch = model.next_token_topk_batch(prefixes, int(root_top_k))
    if len(topk_batch) != len(leaves):
        raise ValueError("Official proactive top-k batch returned a different batch size.")
    best: tuple[float, int, int, CandidateNode, int, list[int]] | None = None
    for leaf_index, (leaf, path, topk) in enumerate(zip(leaves, leaf_paths, topk_batch)):
        leaf_score = float(leaf.draft_logprob or 0.0)
        for candidate in topk:
            score = leaf_score + float(decay_factor) + float(candidate.logprob)
            key = (score, -leaf_index, -int(candidate.rank), leaf, int(candidate.token_id), path)
            if best is None or key[:3] > best[:3]:
                best = key
    if best is None:
        return None
    score, _leaf_order, _rank_order, leaf, token_id, path = best
    return leaf, int(token_id), float(score), list(path)


def _best_leaf_path_tokens(proposal: CandidateProposal) -> list[int]:
    """选择累计 logprob 最高的 leaf path。"""
    if proposal.tree is None or not proposal.tree.nodes:
        return []
    children_by_parent = proposal.tree.children_by_parent()
    leaves = [node for node in proposal.tree.nodes if node.node_id not in children_by_parent]
    if not leaves:
        leaves = list(proposal.tree.nodes)
    best = sorted(
        leaves,
        key=lambda node: (
            -(node.draft_logprob if node.draft_logprob is not None else float("-inf")),
            node.depth,
            node.node_id,
        ),
    )[0]
    return _path_tokens(best, proposal.tree.nodes_by_id())


def _path_tokens(node: CandidateNode, nodes_by_id: dict[int, CandidateNode]) -> list[int]:
    path: list[int] = []
    current: CandidateNode | None = node
    while current is not None:
        path.append(int(current.token_id))
        current = nodes_by_id.get(current.parent_id) if current.parent_id is not None else None
    path.reverse()
    return path


def _reuse_subtree_after_committed_token(
    proactive_proposal: CandidateProposal,
    proactive_prefix: list[int],
    committed_prefix: list[int],
) -> tuple[CandidateProposal, int, int] | None:
    """当 committed prefix 多接受了 proactive root token 时，复用其子树。"""
    if proactive_proposal.tree is None:
        return None
    if len(committed_prefix) != len(proactive_prefix) + 1:
        return None
    if committed_prefix[: len(proactive_prefix)] != proactive_prefix:
        return None
    committed_token = int(committed_prefix[-1])
    children_by_parent = proactive_proposal.tree.children_by_parent()
    root_matches = [
        node
        for node in children_by_parent.get(None, [])
        if int(node.token_id) == committed_token
    ]
    if not root_matches:
        return None
    root = sorted(
        root_matches,
        key=lambda node: (
            -(node.draft_logprob if node.draft_logprob is not None else float("-inf")),
            node.node_id,
        ),
    )[0]
    descendants = _descendants(root.node_id, children_by_parent)
    if not descendants:
        return None
    id_map = {node.node_id: index for index, node in enumerate(descendants)}
    new_nodes = [
        CandidateNode(
            node_id=id_map[node.node_id],
            parent_id=None if node.parent_id == root.node_id else id_map[int(node.parent_id)],
            token_id=int(node.token_id),
            depth=max(1, int(node.depth) - int(root.depth)),
            draft_logprob=node.draft_logprob,
            draft_worker_id=node.draft_worker_id,
        )
        for node in descendants
    ]
    from specplatform.core import CandidateTree

    tree = CandidateTree(root_prefix_len=len(committed_prefix), nodes=new_nodes)
    tree.validate()
    metadata = dict(proactive_proposal.metadata)
    metadata.update(
        {
            "prefix_ids": list(committed_prefix),
            "tree_node_count": len(new_nodes),
            "tree_snapshot": tree.to_dict(),
            "reused_from_proactive_root_node_id": root.node_id,
        }
    )
    reused = CandidateProposal(
        proposal_id=f"{proactive_proposal.proposal_id}:subtree:{root.node_id}",
        request_id=proactive_proposal.request_id,
        worker_id=proactive_proposal.worker_id,
        shape="tree",
        tokens=[node.token_id for node in tree.nodes],
        tree=tree,
        draft_length=len(tree.nodes),
        timing=dict(proactive_proposal.timing),
        metadata=metadata,
    )
    return reused, len(new_nodes), max(0, proactive_proposal.draft_length - len(new_nodes))


def _descendants(
    root_node_id: int,
    children_by_parent: dict[int | None, list[CandidateNode]],
) -> list[CandidateNode]:
    result: list[CandidateNode] = []
    frontier = list(children_by_parent.get(root_node_id, []))
    while frontier:
        node = frontier.pop(0)
        result.append(node)
        frontier.extend(children_by_parent.get(node.node_id, []))
    return result


def _ema(previous: float | None, value: float | None, alpha: float) -> float | None:
    if value is None:
        return previous
    if previous is None:
        return float(value)
    return (float(alpha) * float(value)) + ((1.0 - float(alpha)) * float(previous))
