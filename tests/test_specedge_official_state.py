"""Official SpecEdge state-machine primitives."""

from math import log
from types import SimpleNamespace
import unittest

from specplatform.core import CandidateNode, CandidateTree, DraftBudget, DraftJob, RuntimeContext, VerificationResult
from specplatform.draft import TreeDraftGeneration
from specplatform.methods import (
    OfficialSpecEdgeDraftState,
    OfficialTreeStatus,
    SpecEdgeOfficialAcceptancePolicy,
    SpecEdgeOfficialCandidateStrategy,
    SpecEdgeOfficialProactiveDraftPolicy,
    SpecEdgePipelinePlanningPolicy,
)
from specplatform.model import TopKToken


def _tok(token_id: int, logprob: float, rank: int = 0) -> TopKToken:
    return TopKToken(token_id=token_id, logprob=logprob, rank=rank)


class OfficialSpecEdgeStateTest(unittest.TestCase):
    def test_state_allocates_unique_batch_rows_per_worker(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=3, draft_worker_id="draft-graph")

        first = state.add_request("req-1", [10], draft_worker_id="draft-a")
        second = state.add_request("req-2", [20], draft_worker_id="draft-a")
        other_worker = state.add_request("req-3", [30], draft_worker_id="draft-b")

        self.assertEqual(first.draft_batch_index, 0)
        self.assertEqual(second.draft_batch_index, 1)
        self.assertEqual(other_worker.draft_batch_index, 0)

    def test_state_rejects_duplicate_explicit_batch_row_for_same_worker(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=2, draft_worker_id="draft-graph")
        state.add_request("req-1", [10], draft_worker_id="draft-a", draft_batch_index=1)

        with self.assertRaises(ValueError):
            state.add_request("req-2", [20], draft_worker_id="draft-a", draft_batch_index=1)

    def test_state_reassigns_batch_row_when_worker_changes(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=2, draft_worker_id="draft-graph")
        state.add_request("req-1", [10], draft_worker_id="draft-a", draft_batch_index=0)
        state.add_request("req-2", [20], draft_worker_id="draft-b", draft_batch_index=0)

        slot = state.ensure_request("req-1", [10], draft_worker_id="draft-b")

        self.assertEqual(slot.draft_worker_id, "draft-b")
        self.assertEqual(slot.draft_batch_index, 1)

    def test_specedge_planning_keeps_official_state_worker_affinity(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=3, draft_worker_id="draft-graph")
        state.add_request("req-1", [10], draft_worker_id="draft-a", draft_batch_index=0)
        state.add_request("req-2", [20], draft_worker_id="draft-b", draft_batch_index=0)
        state.add_request("req-3", [30], draft_worker_id="draft-c", draft_batch_index=0)
        policy = SpecEdgePipelinePlanningPolicy(initial_depth=4, official_state=state)

        hints = policy.plan(
            active_sessions=[
                SimpleNamespace(request_id="req-1", prefix_ids=[10]),
                SimpleNamespace(request_id="req-2", prefix_ids=[20, 21]),
                SimpleNamespace(request_id="req-3", prefix_ids=[30]),
            ],
            resources={"draft_worker_ids": ["draft-a", "draft-b"]},
            history={},
            context=RuntimeContext(),
        )

        self.assertEqual(hints.draft_lengths, {"req-1": 4, "req-2": 4, "req-3": 4})
        self.assertEqual(hints.worker_preferences, {"req-1": "draft-a"})
        self.assertEqual(hints.metadata["official_state_worker_affinity_count"], 1)

    def test_state_selects_candidates_and_prunes_children_with_official_budget(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=2, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10, 11])

        root_ids = slot.add_root_candidates([_tok(2, -0.1), _tok(3, -0.2, rank=1)])
        beams = state.select_candidate_beams(max_beams_per_request=1)

        self.assertEqual(root_ids, [0, 1])
        self.assertEqual([beam.node_id for beam in beams], [0])
        self.assertEqual(slot.statuses[0], OfficialTreeStatus.PROCESSED)
        self.assertEqual(slot.statuses[1], OfficialTreeStatus.CANDIDATE)

        added = slot.add_budgeted_children(
            children_by_parent={
                0: [
                    _tok(4, -0.01),
                    _tok(5, -20.0, rank=1),
                ]
            },
            max_budget=3,
            decay_factor=log(0.9),
        )

        tree = slot.to_candidate_tree()
        self.assertEqual(added, [2])
        self.assertEqual([node.token_id for node in tree.nodes], [2, 3, 4])
        self.assertEqual([node.parent_id for node in tree.nodes], [None, None, 0])
        self.assertEqual(slot.statuses[2], OfficialTreeStatus.CANDIDATE)

    def test_apply_acceptance_compacts_prefix_and_returns_gather_indices(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=1, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10, 11])
        slot.add_root_candidates([_tok(2, -0.1)])
        slot.statuses[0] = OfficialTreeStatus.PROCESSED
        slot.add_budgeted_children(
            children_by_parent={0: [_tok(4, -0.1)]},
            max_budget=4,
            decay_factor=0.0,
        )

        reorder = slot.apply_acceptance([0, 1], bonus_token=9)

        self.assertEqual(reorder.emitted_tokens, [2, 4, 9])
        self.assertEqual(reorder.source_seq_indices, [0, 1, 2, 3])
        self.assertEqual(reorder.dest_seq_indices, [0, 1, 2, 3])
        self.assertEqual(slot.prefix_ids, [10, 11, 2, 4, 9])
        self.assertEqual(slot.nodes, [])
        self.assertEqual(slot.statuses, {})

    def test_apply_acceptance_requires_contiguous_root_to_leaf_path(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=1, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10])
        slot.add_root_candidates([_tok(2, -0.1), _tok(3, -0.2, rank=1)])
        slot.statuses[0] = OfficialTreeStatus.PROCESSED
        slot.add_budgeted_children(
            children_by_parent={0: [_tok(4, -0.1)]},
            max_budget=4,
            decay_factor=0.0,
        )

        with self.assertRaises(ValueError):
            slot.apply_acceptance([1, 2], bonus_token=None)

    def test_to_candidate_proposal_marks_official_state_without_affecting_other_methods(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=1, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10])
        slot.add_root_candidates([_tok(2, -0.1)])

        proposal = slot.to_candidate_proposal(proposal_id="p1", allow_bonus=True)

        self.assertEqual(proposal.shape, "tree")
        self.assertEqual(proposal.request_id, "req-1")
        self.assertEqual(proposal.worker_id, "draft-graph")
        self.assertTrue(proposal.metadata["official_specedge_state"])
        self.assertEqual(proposal.metadata["tree_node_statuses"], {"0": int(OfficialTreeStatus.CANDIDATE)})

    def test_official_candidate_strategy_batches_jobs_and_populates_state(self) -> None:
        class FakeBatchDraftRunner:
            runner_id = "draft-graph"

            def __init__(self) -> None:
                self.requests = []

            def generate_tree_batch(self, requests):
                self.requests = list(requests)
                generations = []
                for request in requests:
                    token_id = int(request["prefix_ids"][-1]) + 1
                    tree = CandidateTree(
                        root_prefix_len=len(request["prefix_ids"]),
                        nodes=[CandidateNode(0, None, token_id, 1, -0.1, request["runner_id"])],
                    )
                    generations.append(
                        TreeDraftGeneration(
                            tree=tree,
                            metadata={
                                "runner_id": request["runner_id"],
                                "official_draft_batch_index": 0,
                                "batch_index": 0,
                                "tree_node_statuses": {"0": int(OfficialTreeStatus.PROCESSED)},
                            },
                        )
                    )
                return generations

        state = OfficialSpecEdgeDraftState(max_batch_size=2, draft_worker_id="draft-graph")
        strategy = SpecEdgeOfficialCandidateStrategy(state=state, default_max_budget=4, default_max_branch_width=2)
        runner = FakeBatchDraftRunner()
        sessions = {
            "r0": SimpleNamespace(request_id="r0", prefix_ids=[10], remaining_tokens=3, step_idx=0),
            "r1": SimpleNamespace(request_id="r1", prefix_ids=[20], remaining_tokens=3, step_idx=0),
        }
        jobs = [
            DraftJob("r0", "draft-graph", DraftBudget(max_tokens=2, max_branches=2)),
            DraftJob("r1", "draft-graph", DraftBudget(max_tokens=2, max_branches=2)),
        ]

        proposals = strategy.propose_batch(
            jobs=jobs,
            sessions_by_id=sessions,
            draft_runners={"draft-graph": runner},
            context=RuntimeContext(method_config={"max_budget": 4, "max_branch_width": 2}),
        )

        self.assertEqual(len(runner.requests), 2)
        self.assertEqual([request["draft_batch_index"] for request in runner.requests], [0, 1])
        self.assertEqual([proposal.tokens for proposal in proposals], [[11], [21]])
        self.assertTrue(all(proposal.metadata["official_specedge_state"] for proposal in proposals))
        self.assertEqual(state.slot("r0").statuses[0], OfficialTreeStatus.PROCESSED)
        self.assertEqual(state.slot("r1").prefix_ids, [20])
        self.assertEqual(state.slot("r0").draft_batch_index, 0)
        self.assertEqual(state.slot("r1").draft_batch_index, 1)

    def test_official_candidate_strategy_grows_reused_state_when_runner_supports_it(self) -> None:
        class FakeBatchDraftRunner:
            runner_id = "draft-graph"

            def __init__(self) -> None:
                self.grow_requests = []

            def grow_official_tree_batch(self, requests):
                self.grow_requests = list(requests)
                generations = []
                for request in requests:
                    tree = CandidateTree(
                        root_prefix_len=len(request["prefix_ids"]),
                        nodes=[CandidateNode(0, None, 11, 1, -0.1, request["runner_id"])],
                    )
                    generations.append(
                        TreeDraftGeneration(
                            tree=tree,
                            metadata={
                                "runner_id": request["runner_id"],
                                "official_draft_batch_index": request["draft_batch_index"],
                                "official_persistent_kv_reused": True,
                                "official_needs_prefix_tail_forward": False,
                                "tree_node_statuses": {"0": int(OfficialTreeStatus.CANDIDATE)},
                            },
                        )
                    )
                return generations

            def generate_tree_batch(self, requests):
                raise AssertionError("reused official state should grow persistent KV instead of full prefill")

        state = OfficialSpecEdgeDraftState(max_batch_size=2, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10, 2, 9], draft_batch_index=1)
        slot.needs_prefix_tail_forward = True
        strategy = SpecEdgeOfficialCandidateStrategy(state=state, default_max_budget=4, default_max_branch_width=2)
        runner = FakeBatchDraftRunner()
        session = SimpleNamespace(request_id="req-1", prefix_ids=[10, 2, 9], remaining_tokens=3, step_idx=1)

        proposal = strategy.propose_batch(
            jobs=[DraftJob("req-1", "draft-graph", DraftBudget(max_tokens=2, max_branches=1))],
            sessions_by_id={"req-1": session},
            draft_runners={"draft-graph": runner},
            context=RuntimeContext(method_config={"max_budget": 4, "max_branch_width": 1}),
        )[0]

        self.assertEqual(len(runner.grow_requests), 1)
        self.assertEqual(runner.grow_requests[0]["prefix_ids"], [10, 2, 9])
        self.assertEqual(runner.grow_requests[0]["draft_batch_index"], 1)
        self.assertTrue(runner.grow_requests[0]["needs_prefix_tail_forward"])
        self.assertEqual(proposal.tokens, [11])
        self.assertTrue(proposal.metadata["official_state_grown_tree"])
        self.assertTrue(proposal.metadata["official_persistent_kv_reused"])
        self.assertFalse(state.slot("req-1").needs_prefix_tail_forward)

    def test_official_acceptance_defers_state_commit_until_winner(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=1, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10])
        slot.add_root_candidates([_tok(2, -0.1)])
        proposal = slot.to_candidate_proposal(proposal_id="p1", allow_bonus=True)
        verification = VerificationResult(
            request_id="req-1",
            proposal_id="p1",
            shape="tree",
            bonus_token=9,
            payload={
                "accepted_node_ids": [0],
                "rejected_node_ids": [],
                "target_choices": [],
            },
        )

        policy = SpecEdgeOfficialAcceptancePolicy(state=state)
        accepted = policy.accept(proposal, verification, RuntimeContext())

        self.assertEqual(accepted.output_token_ids, [2, 9])
        self.assertTrue(accepted.metadata["official_acceptance_pending"])
        self.assertEqual(slot.prefix_ids, [10])

        committed = policy.commit_acceptance(proposal, verification, accepted, RuntimeContext())

        self.assertEqual(slot.prefix_ids, [10, 2, 9])
        self.assertEqual(slot.nodes, [])
        self.assertTrue(slot.needs_prefix_tail_forward)
        self.assertFalse(committed.metadata["official_acceptance_pending"])
        self.assertEqual(committed.metadata["official_reorder"]["emitted_tokens"], [2, 9])

    def test_official_acceptance_forwards_tail_after_no_bonus_acceptance(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=1, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10])
        slot.add_root_candidates([_tok(2, -0.1)])
        proposal = slot.to_candidate_proposal(proposal_id="p1", allow_bonus=True)
        verification = VerificationResult(
            request_id="req-1",
            proposal_id="p1",
            shape="tree",
            bonus_token=None,
            payload={
                "accepted_node_ids": [0],
                "rejected_node_ids": [],
                "target_choices": [],
            },
        )

        policy = SpecEdgeOfficialAcceptancePolicy(state=state)
        accepted = policy.accept(proposal, verification, RuntimeContext())
        policy.commit_acceptance(proposal, verification, accepted, RuntimeContext())

        self.assertEqual(slot.prefix_ids, [10, 2])
        self.assertEqual(slot.nodes, [])
        self.assertTrue(slot.needs_prefix_tail_forward)

    def test_proactive_subtree_is_kept_when_bonus_matches_official_root(self) -> None:
        state = OfficialSpecEdgeDraftState(max_batch_size=1, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10])
        slot.add_root_candidates([_tok(2, -0.1)])
        slot.statuses[0] = OfficialTreeStatus.PROCESSED
        slot.add_budgeted_children(
            children_by_parent={0: [_tok(4, -0.1)]},
            max_budget=8,
            decay_factor=0.0,
        )
        proactive_subtree = CandidateTree(
            root_prefix_len=4,
            nodes=[CandidateNode(0, None, 11, 1, -0.2, "draft-graph")],
        )
        record = slot.add_proactive_subtree(
            parent_node_id=1,
            root_token_id=9,
            root_logprob=-0.3,
            subtree=proactive_subtree,
            subtree_statuses={0: OfficialTreeStatus.PROCESSED},
        )

        reorder = slot.apply_acceptance([0, 1], bonus_token=9)

        self.assertEqual(record.root_token_id, 9)
        self.assertTrue(reorder.reused_proactive_tree)
        self.assertEqual(reorder.emitted_tokens, [2, 4, 9])
        self.assertEqual(reorder.source_seq_indices, [0, 1, 2, 3, 4])
        self.assertEqual(reorder.dest_seq_indices, [0, 1, 2, 3, 4])
        self.assertEqual(reorder.retained_tree_node_count, 1)
        self.assertEqual(slot.prefix_ids, [10, 2, 4, 9])
        self.assertEqual([node.token_id for node in slot.nodes], [11])
        self.assertEqual(slot.nodes[0].parent_id, None)
        self.assertEqual(slot.nodes[0].depth, 1)
        self.assertEqual(slot.statuses[0], OfficialTreeStatus.PROCESSED)

    def test_official_proactive_policy_uses_post_statuses_and_next_round_reuses_state(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.prefix_batches = []

            def next_token_topk_batch(self, prefix_ids_batch, k):
                self.prefix_batches.append((prefix_ids_batch, k))
                return [[_tok(9, -0.01)] for _prefix in prefix_ids_batch]

        class FakeRunner:
            runner_id = "draft-graph"

            def __init__(self) -> None:
                self.model = FakeModel()
                self.generate_tree_calls = 0

            def generate_tree(self, *, prefix_ids, max_depth, max_branches, max_nodes, request_id=None, metadata=None):
                self.generate_tree_calls += 1
                tree = CandidateTree(
                    root_prefix_len=len(prefix_ids),
                    nodes=[CandidateNode(0, None, 11, 1, -0.2, "draft-graph")],
                )
                return TreeDraftGeneration(
                    tree=tree,
                    metadata={"runner_id": "draft-graph", "tree_node_statuses": {"0": int(OfficialTreeStatus.PROCESSED)}},
                )

            def generate_tree_batch(self, requests):
                raise AssertionError("next official round should reuse preserved proactive state")

        state = OfficialSpecEdgeDraftState(max_batch_size=1, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10])
        slot.add_root_candidates([_tok(2, -0.1)])
        proposal = slot.to_candidate_proposal(proposal_id="p1", allow_bonus=True)
        runner = FakeRunner()
        session = SimpleNamespace(request_id="req-1", prefix_ids=[10], prompt_ids=[10], remaining_tokens=4, max_new_tokens=4, step_idx=0)
        proactive_policy = SpecEdgeOfficialProactiveDraftPolicy(
            state=state,
            default_max_depth=1,
            default_max_branch_width=1,
            default_max_budget=2,
            default_leaf_beams=1,
            default_root_top_k=1,
        )

        proactive = proactive_policy.propose_proactive(session, proposal, runner, RuntimeContext())

        self.assertIsNotNone(proactive)
        self.assertEqual(runner.model.prefix_batches[0][0], [[10, 2]])
        self.assertEqual(runner.generate_tree_calls, 1)
        self.assertIsNotNone(slot.proactive_record)
        assert slot.proactive_record is not None
        self.assertEqual(slot.proactive_record.root_token_id, 9)
        self.assertEqual(slot.statuses[slot.proactive_record.root_node_id], OfficialTreeStatus.POST_CANDIDATE)
        subtree_node_id = [node_id for node_id in slot.proactive_record.node_ids if node_id != slot.proactive_record.root_node_id][0]
        self.assertEqual(slot.statuses[subtree_node_id], OfficialTreeStatus.POST_PROCESSED)

        slot.apply_acceptance([0], bonus_token=9)
        session.prefix_ids = list(slot.prefix_ids)
        session.remaining_tokens = 2
        session.step_idx = 1
        candidate_strategy = SpecEdgeOfficialCandidateStrategy(state=state, default_max_budget=2, default_max_branch_width=1)
        reused = candidate_strategy.propose_batch(
            jobs=[DraftJob("req-1", "draft-graph", DraftBudget(max_tokens=1, max_branches=1))],
            sessions_by_id={"req-1": session},
            draft_runners={"draft-graph": runner},
            context=RuntimeContext(),
        )[0]

        self.assertTrue(reused.metadata["official_state_reused_tree"])
        self.assertEqual(reused.tokens, [11])

    def test_official_proactive_policy_prefers_graph_backend_boundary(self) -> None:
        class FakeRunner:
            runner_id = "draft-graph"

            def __init__(self) -> None:
                self.proactive_requests = []
                self.model = SimpleNamespace(next_token_topk_batch=self._unexpected_topk)

            def _unexpected_topk(self, prefix_ids_batch, k):
                raise AssertionError("official graph proactive should not use prefix-list topk fallback")

            def generate_official_proactive(self, request):
                self.proactive_requests.append(dict(request))
                subtree = CandidateTree(
                    root_prefix_len=3,
                    nodes=[CandidateNode(0, None, 11, 1, -0.2, "draft-graph")],
                )
                return {
                    "parent_node_id": 0,
                    "root_token_id": 9,
                    "root_logprob": -0.05,
                    "root_status": int(OfficialTreeStatus.POST_PROCESSED),
                    "leaf_path": [2],
                    "subtree": subtree,
                    "subtree_statuses": {"0": int(OfficialTreeStatus.CANDIDATE)},
                    "metadata": {
                        "official_proactive_graph": True,
                        "official_persistent_kv_reused": True,
                        "tree_draft_backend": "qwen3_batch_graph_official_proactive",
                    },
                }

            def generate_tree(self, **kwargs):
                raise AssertionError("official graph proactive should not full-prefill proactive_prefix")

        state = OfficialSpecEdgeDraftState(max_batch_size=1, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10], draft_batch_index=0)
        slot.add_root_candidates([_tok(2, -0.1)])
        proposal = slot.to_candidate_proposal(proposal_id="p1", allow_bonus=True)
        runner = FakeRunner()
        session = SimpleNamespace(request_id="req-1", prefix_ids=[10], prompt_ids=[10], remaining_tokens=4, max_new_tokens=4, step_idx=0)
        proactive_policy = SpecEdgeOfficialProactiveDraftPolicy(
            state=state,
            default_max_depth=1,
            default_max_branch_width=1,
            default_max_budget=2,
            default_leaf_beams=1,
            default_root_top_k=1,
        )

        proactive = proactive_policy.propose_proactive(session, proposal, runner, RuntimeContext())

        self.assertIsNotNone(proactive)
        self.assertEqual(len(runner.proactive_requests), 1)
        self.assertEqual(runner.proactive_requests[0]["draft_batch_index"], 0)
        self.assertEqual(runner.proactive_requests[0]["tree_node_statuses"], {"0": int(OfficialTreeStatus.CANDIDATE)})
        self.assertTrue(proactive.metadata["official_proactive_graph"])
        self.assertTrue(proactive.metadata["official_persistent_kv_reused"])
        assert slot.proactive_record is not None
        self.assertEqual(slot.proactive_record.root_token_id, 9)
        self.assertEqual(slot.statuses[slot.proactive_record.root_node_id], OfficialTreeStatus.POST_PROCESSED)
        child_id = [node_id for node_id in slot.proactive_record.node_ids if node_id != slot.proactive_record.root_node_id][0]
        self.assertEqual(slot.statuses[child_id], OfficialTreeStatus.POST_CANDIDATE)

    def test_official_acceptance_commits_reorder_to_draft_model_when_available(self) -> None:
        class FakeModel:
            def __init__(self) -> None:
                self.commits = []

            def official_specedge_commit_acceptance(self, **kwargs):
                self.commits.append(kwargs)
                return {"fake_model_committed": True, "batch_index": kwargs["batch_index"]}

        state = OfficialSpecEdgeDraftState(max_batch_size=4, draft_worker_id="draft-graph")
        slot = state.add_request("req-1", [10], draft_worker_id="draft-graph", draft_batch_index=3)
        slot.add_root_candidates([_tok(2, -0.1)])
        proposal = slot.to_candidate_proposal(proposal_id="p1", allow_bonus=True)
        verification = VerificationResult(
            request_id="req-1",
            proposal_id="p1",
            shape="tree",
            bonus_token=9,
            payload={"accepted_node_ids": [0], "rejected_node_ids": [], "target_choices": []},
        )
        policy = SpecEdgeOfficialAcceptancePolicy(state=state)
        accepted = policy.accept(proposal, verification, RuntimeContext())
        model = FakeModel()

        committed = policy.commit_acceptance(
            proposal,
            verification,
            accepted,
            RuntimeContext(),
            draft_runners={"draft-graph": SimpleNamespace(model=model)},
        )

        self.assertEqual(model.commits[0]["request_id"], "req-1")
        self.assertEqual(model.commits[0]["batch_index"], 3)
        self.assertEqual(model.commits[0]["source_seq_indices"], [0, 1])
        self.assertEqual(model.commits[0]["dest_seq_indices"], [0, 1])
        self.assertTrue(committed.metadata["official_model_commit"]["committed"])
        self.assertTrue(committed.metadata["official_model_commit"]["fake_model_committed"])


if __name__ == "__main__":
    unittest.main()
