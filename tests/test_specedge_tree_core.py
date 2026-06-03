"""SpecEdge tree speculative decoding 核心单元测试。"""

import json
import threading
import time
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
from tempfile import TemporaryDirectory
from typing import Any

from specplatform.core import (
    AcceptResult,
    CandidateNode,
    CandidateProposal,
    CandidateTree,
    DraftBudget,
    PhaseEvent,
    RuntimeContext,
    VerificationResult,
)
from specplatform.draft import TopKTreeDraftRunner
from specplatform.methods import (
    SpecEdgePipelinePlanningPolicy,
    SpecEdgeProactiveDraftPolicy,
    SpecEdgeReconcilePolicy,
    SpecEdgeTreeAcceptancePolicy,
    SpecEdgeTreeCandidateStrategy,
)
from specplatform.model import (
    CausalLMRunner,
    ModelBackendCapabilities,
    ModelForwardInput,
    ModelForwardOutput,
    TreeForwardChoice,
    TreeForwardInput,
    TreeForwardOutput,
)
from specplatform.metrics import write_tree_snapshots_jsonl
from specplatform.runtime import AsyncPipelineRuntimeEngine, GenerationSession, RuntimeEngine
from specplatform.schedulers import RoundRobinRequestScheduler
from specplatform.verification import BatchVerifyItem, BatchVerifyRequest, BatchVerifyResponse, BatchVerifyResultItem, HttpTreeVerifierClient, TreeVerifier
from specplatform.verification.schema import TreeVerifyRequest, TreeVerifyResponse


class ScriptedCausalLMRunner(CausalLMRunner):
    """测试内部脚本化 causal LM。"""

    runner_id = "scripted"
    max_len = 64

    def __init__(self, next_tokens_by_prefix: dict[tuple[int, ...], int]) -> None:
        self.next_tokens_by_prefix = dict(next_tokens_by_prefix)
        self.seen_prefixes: list[list[int]] = []

    def encode(self, text: str) -> list[int]:
        return [int(part) for part in text.split()] if text.strip() else []

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        return ModelForwardOutput(logits=[self.next_token_logits(request.input_ids)])

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        self.seen_prefixes.append(list(prefix_ids))
        token_id = self.next_tokens_by_prefix[tuple(prefix_ids)]
        logits = [-10.0] * 16
        logits[token_id] = 10.0
        return logits


class BatchScriptedCausalLMRunner(CausalLMRunner):
    """只支持 batch next-token 的测试 runner，用来验证 tree verifier 调用形态。"""

    runner_id = "batch-scripted"
    max_len = 64

    def __init__(self, next_tokens_by_prefix: dict[tuple[int, ...], int]) -> None:
        self.next_tokens_by_prefix = dict(next_tokens_by_prefix)
        self.batch_calls: list[list[list[int]]] = []
        self.single_calls: list[list[int]] = []

    def encode(self, text: str) -> list[int]:
        return [int(part) for part in text.split()] if text.strip() else []

    def decode(self, token_ids: list[int]) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def forward(self, request: ModelForwardInput) -> ModelForwardOutput:
        return ModelForwardOutput(logits=[self.next_token_logits(request.input_ids)])

    def next_token_logits(self, prefix_ids: list[int]) -> list[float]:
        self.single_calls.append(list(prefix_ids))
        token_id = self.next_tokens_by_prefix[tuple(prefix_ids)]
        logits = [-10.0] * 16
        logits[token_id] = 10.0
        return logits

    def next_token_logits_batch(self, prefix_ids_batch: list[list[int]]) -> list[list[float]]:
        self.batch_calls.append([list(prefix_ids) for prefix_ids in prefix_ids_batch])
        logits_batch: list[list[float]] = []
        for prefix_ids in prefix_ids_batch:
            token_id = self.next_tokens_by_prefix[tuple(prefix_ids)]
            logits = [-10.0] * 16
            logits[token_id] = 10.0
            logits_batch.append(logits)
        return logits_batch

    def backend_capabilities(self) -> ModelBackendCapabilities:
        return ModelBackendCapabilities(
            backend_name="batch-scripted",
            supports_batched_next_token=True,
        )


class TreeForwardScriptedCausalLMRunner(BatchScriptedCausalLMRunner):
    """支持解耦 tree_forward 的测试 runner。"""

    runner_id = "tree-forward-scripted"

    def __init__(self, next_tokens_by_parent: dict[int | None, int]) -> None:
        super().__init__({})
        self.next_tokens_by_parent = dict(next_tokens_by_parent)
        self.tree_forward_calls: list[TreeForwardInput] = []

    def tree_forward(self, request: TreeForwardInput) -> TreeForwardOutput:
        self.tree_forward_calls.append(request)
        parent_ids: list[int | None] = []
        for node in request.nodes:
            if node.parent_id not in parent_ids:
                parent_ids.append(node.parent_id)
        return TreeForwardOutput(
            choices=[
                TreeForwardChoice(
                    parent_node_id=parent_id,
                    target_token_id=self.next_tokens_by_parent[parent_id],
                    prefix_len=len(request.prefix_ids) + (0 if parent_id is None else 1),
                )
                for parent_id in parent_ids
            ],
            metadata={
                "tree_forward_kind": "tree_attention",
                "packed_token_count": len(request.prefix_ids) + len(request.nodes),
                "choice_count": len(parent_ids),
            },
        )

    def backend_capabilities(self) -> ModelBackendCapabilities:
        return ModelBackendCapabilities(
            backend_name="tree-forward-scripted",
            supports_batched_next_token=True,
            supports_tree_attention=True,
        )


class BatchTreeForwardScriptedCausalLMRunner(TreeForwardScriptedCausalLMRunner):
    """支持模型层 fused tree_forward_batch 的测试 runner。"""

    runner_id = "batch-tree-forward-scripted"

    def __init__(self, next_tokens_by_request_parent: dict[tuple[tuple[int, ...], int | None], int]) -> None:
        super().__init__({})
        self.next_tokens_by_request_parent = dict(next_tokens_by_request_parent)
        self.tree_forward_batch_calls: list[list[TreeForwardInput]] = []

    def tree_forward(self, request: TreeForwardInput) -> TreeForwardOutput:
        self.tree_forward_calls.append(request)
        return self._output_for_request(request, batch_index=0, batch_size=1)

    def tree_forward_batch(self, requests: list[TreeForwardInput]) -> list[TreeForwardOutput]:
        self.tree_forward_batch_calls.append(list(requests))
        return [
            self._output_for_request(request, batch_index=index, batch_size=len(requests))
            for index, request in enumerate(requests)
        ]

    def backend_capabilities(self) -> ModelBackendCapabilities:
        return ModelBackendCapabilities(
            backend_name="batch-tree-forward-scripted",
            supports_batched_next_token=True,
            supports_tree_attention=True,
            supports_tree_forward_batch=True,
        )

    def _output_for_request(
        self,
        request: TreeForwardInput,
        *,
        batch_index: int,
        batch_size: int,
    ) -> TreeForwardOutput:
        parent_ids: list[int | None] = []
        for node in request.nodes:
            if node.parent_id not in parent_ids:
                parent_ids.append(node.parent_id)
        return TreeForwardOutput(
            choices=[
                TreeForwardChoice(
                    parent_node_id=parent_id,
                    target_token_id=self.next_tokens_by_request_parent[(tuple(request.prefix_ids), parent_id)],
                    prefix_len=len(request.prefix_ids) + (0 if parent_id is None else 1),
                )
                for parent_id in parent_ids
            ],
            metadata={
                "tree_forward_kind": "tree_attention_batch",
                "tree_forward_batch_kind": "tree_attention_batch",
                "batch_index": batch_index,
                "batch_size": batch_size,
                "packed_token_count": len(request.prefix_ids) + len(request.nodes),
                "choice_count": len(parent_ids),
            },
        )


class TreeSchemaTest(unittest.TestCase):
    """覆盖 tree schema 和 topology validation。"""

    def test_tree_verify_schema_round_trip(self) -> None:
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, 0, 3, 2, -0.2, "draft-0"),
            ],
        )
        request = TreeVerifyRequest(
            request_id="request-1",
            proposal_id="proposal-1",
            prefix_ids=[1],
            tree=tree,
            eos_token_ids=[9],
            allow_bonus=False,
            metadata={"batch_id": "batch-1"},
        )
        response = TreeVerifyResponse(
            request_id="request-1",
            proposal_id="proposal-1",
            accepted_node_ids=[0, 1],
            target_choices=[{"parent_node_id": None, "target_token_id": 2}],
            bonus_token=None,
            rejected_node_ids=[],
            timing={"server_total_ms": 1.0},
        )

        restored_request = TreeVerifyRequest.from_dict(request.to_dict())
        restored_response = TreeVerifyResponse.from_dict(response.to_dict())

        self.assertEqual(restored_request.tree.nodes[1].parent_id, 0)
        self.assertEqual(restored_request.eos_token_ids, [9])
        self.assertFalse(restored_request.allow_bonus)
        self.assertEqual(restored_response.accepted_node_ids, [0, 1])
        self.assertEqual(restored_response.timing["server_total_ms"], 1.0)

    def test_batch_verify_schema_round_trip(self) -> None:
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[CandidateNode(0, None, 2, 1, -0.1, "draft-0")],
        )
        request = BatchVerifyRequest(
            batch_id="batch-1",
            items=[
                BatchVerifyItem(
                    kind="tree",
                    request=TreeVerifyRequest(
                        request_id="request-1",
                        proposal_id="proposal-1",
                        prefix_ids=[1],
                        tree=tree,
                    ),
                )
            ],
            metadata={"source": "test"},
        )
        response = BatchVerifyResponse(
            batch_id="batch-1",
            results=[
                BatchVerifyResultItem(
                    kind="tree",
                    response=TreeVerifyResponse(
                        request_id="request-1",
                        proposal_id="proposal-1",
                        accepted_node_ids=[0],
                        target_choices=[],
                        timing={"server_total_ms": 1.0},
                    ),
                )
            ],
            timing={"server_batch_total_ms": 2.0, "batch_size": 1},
        )

        restored_request = BatchVerifyRequest.from_dict(request.to_dict())
        restored_response = BatchVerifyResponse.from_dict(response.to_dict())

        self.assertEqual(restored_request.batch_id, "batch-1")
        self.assertIsInstance(restored_request.items[0].request, TreeVerifyRequest)
        self.assertEqual(restored_response.timing["server_batch_total_ms"], 2.0)
        self.assertIsInstance(restored_response.results[0].response, TreeVerifyResponse)

    def test_tree_topology_rejects_duplicate_missing_parent_and_bad_depth(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            CandidateTree(
                root_prefix_len=1,
                nodes=[
                    CandidateNode(0, None, 2, 1, None, "draft-0"),
                    CandidateNode(0, None, 3, 1, None, "draft-0"),
                ],
            ).validate()

        with self.assertRaisesRegex(ValueError, "Parent"):
            CandidateTree(
                root_prefix_len=1,
                nodes=[CandidateNode(1, 0, 3, 2, None, "draft-0")],
            ).validate()

        with self.assertRaisesRegex(ValueError, "depth"):
            CandidateTree(
                root_prefix_len=1,
                nodes=[CandidateNode(0, None, 2, 2, None, "draft-0")],
            ).validate()

    def test_tree_snapshot_artifact_merges_accept_path(self) -> None:
        events = [
            PhaseEvent(
                run_id="run-1",
                request_id="request-1",
                method="specedge_tree",
                phase="draft.generate",
                duration_ms=1.0,
                span_kind="leaf",
                proposal_id="proposal-1",
                metadata={
                    "tree_snapshot": {
                        "root_prefix_len": 1,
                        "nodes": [CandidateNode(0, None, 2, 1, -0.1, "draft-0").to_dict()],
                    }
                },
            ),
            PhaseEvent(
                run_id="run-1",
                request_id="request-1",
                method="specedge_tree",
                phase="accept.apply",
                duration_ms=0.1,
                span_kind="leaf",
                proposal_id="proposal-1",
                metadata={
                    "accepted_node_ids": [0],
                    "rejected_node_ids": [],
                    "accepted_count": 1,
                    "rejected_count": 0,
                    "has_bonus": True,
                },
            ),
        ]
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "tree_snapshots.jsonl"

            count = write_tree_snapshots_jsonl(events, path)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(count, 1)
        self.assertEqual(rows[0]["accepted_node_ids"], [0])
        self.assertEqual(rows[0]["tree"]["nodes"][0]["token_id"], 2)


class TreeVerifierTest(unittest.TestCase):
    """覆盖 tree verifier 的 greedy path 和 bonus 语义。"""

    def test_tree_verifier_accepts_longest_target_path_and_bonus(self) -> None:
        target = ScriptedCausalLMRunner({(1,): 2, (1, 2): 3, (1, 2, 3): 4, (1, 5): 6})
        verifier = TreeVerifier(model=target)
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, None, 5, 1, -0.2, "draft-0"),
                CandidateNode(2, 0, 3, 2, -0.3, "draft-0"),
            ],
        )
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-0",
            shape="tree",
            tree=tree,
            metadata={"prefix_ids": [1]},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.accepted_prefix_len, 2)
        self.assertEqual(result.payload["accepted_node_ids"], [0, 2])
        self.assertEqual(result.bonus_token, 4)
        self.assertEqual(result.payload["rejected_node_ids"], [1])
        self.assertIn("target_tree_forward_events", result.timing)

    def test_tree_verifier_returns_mismatch_correction_even_when_bonus_disabled(self) -> None:
        target = ScriptedCausalLMRunner({(1,): 7})
        verifier = TreeVerifier(model=target)
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[CandidateNode(0, None, 2, 1, -0.1, "draft-0")],
        )
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-0",
            shape="tree",
            tree=tree,
            metadata={"prefix_ids": [1], "allow_bonus": False},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.accepted_prefix_len, 0)
        self.assertEqual(result.bonus_token, 7)
        self.assertEqual(result.payload["accepted_node_ids"], [])

    def test_tree_verifier_skips_full_match_bonus_when_not_allowed(self) -> None:
        target = ScriptedCausalLMRunner({(1,): 2})
        verifier = TreeVerifier(model=target)
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[CandidateNode(0, None, 2, 1, -0.1, "draft-0")],
        )
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-0",
            shape="tree",
            tree=tree,
            metadata={"prefix_ids": [1], "allow_bonus": False},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.accepted_prefix_len, 1)
        self.assertIsNone(result.bonus_token)
        self.assertEqual(target.seen_prefixes, [[1]])

    def test_tree_verifier_batches_parent_choice_for_capable_backend(self) -> None:
        target = BatchScriptedCausalLMRunner({(1,): 2, (1, 2): 3, (1, 5): 6})
        verifier = TreeVerifier(model=target)
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, None, 5, 1, -0.2, "draft-0"),
                CandidateNode(2, 0, 3, 2, -0.3, "draft-0"),
                CandidateNode(3, 1, 6, 2, -0.4, "draft-0"),
            ],
        )
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-0",
            shape="tree",
            tree=tree,
            metadata={"prefix_ids": [1], "allow_bonus": False},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.payload["accepted_node_ids"], [0, 2])
        self.assertEqual(target.batch_calls, [[[1], [1, 2], [1, 5]]])
        self.assertEqual(target.single_calls, [])
        events = result.timing["target_tree_forward_events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "tree_choice_batch")
        self.assertEqual(events[0]["batch_size"], 3)
        self.assertEqual(events[0]["parent_node_ids"], [None, 0, 1])
        self.assertEqual(result.timing["target_tree_forward_event_count"], 1)

    def test_tree_verifier_prefers_decoupled_tree_forward_backend(self) -> None:
        target = TreeForwardScriptedCausalLMRunner({None: 2, 0: 3, 1: 6})
        verifier = TreeVerifier(model=target)
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, None, 5, 1, -0.2, "draft-0"),
                CandidateNode(2, 0, 3, 2, -0.3, "draft-0"),
                CandidateNode(3, 1, 6, 2, -0.4, "draft-0"),
            ],
        )
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-0",
            shape="tree",
            tree=tree,
            metadata={"prefix_ids": [1], "allow_bonus": False},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.payload["accepted_node_ids"], [0, 2])
        self.assertEqual(len(target.tree_forward_calls), 1)
        self.assertEqual(target.batch_calls, [])
        events = result.timing["target_tree_forward_events"]
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["kind"], "tree_attention")
        self.assertEqual(events[0]["choice_count"], 3)
        self.assertEqual(events[0]["packed_token_count"], 5)

    def test_tree_forward_batch_fallback_matches_individual_tree_forward(self) -> None:
        target = TreeForwardScriptedCausalLMRunner({None: 2, 0: 3})
        request = TreeForwardInput(
            prefix_ids=[1],
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, 0, 3, 2, -0.2, "draft-0"),
            ],
        )

        single = target.tree_forward(request)
        batched = target.tree_forward_batch([request])[0]

        self.assertEqual(
            [choice.target_token_id for choice in batched.choices],
            [choice.target_token_id for choice in single.choices],
        )
        self.assertEqual(batched.metadata["tree_forward_batch_kind"], "fallback_sequential")
        self.assertEqual(batched.metadata["batch_size"], 1)

    def test_tree_verifier_uses_fused_tree_forward_batch_for_multiple_requests(self) -> None:
        target = BatchTreeForwardScriptedCausalLMRunner(
            {
                ((1,), None): 2,
                ((1,), 0): 3,
                ((4,), None): 5,
                ((4,), 0): 6,
            }
        )
        verifier = TreeVerifier(model=target)
        tree_a = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, 0, 3, 2, -0.2, "draft-0"),
            ],
        )
        tree_b = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 5, 1, -0.1, "draft-1"),
                CandidateNode(1, 0, 6, 2, -0.2, "draft-1"),
            ],
        )
        requests = [
            TreeVerifyRequest(
                request_id="request-a",
                proposal_id="proposal-a",
                prefix_ids=[1],
                tree=tree_a,
                allow_bonus=False,
            ),
            TreeVerifyRequest(
                request_id="request-b",
                proposal_id="proposal-b",
                prefix_ids=[4],
                tree=tree_b,
                allow_bonus=False,
            ),
        ]

        responses = verifier.verify_requests_batch(requests, batch_id="batch-test")

        self.assertEqual([response.accepted_node_ids for response in responses], [[0, 1], [0, 1]])
        self.assertEqual(len(target.tree_forward_batch_calls), 1)
        self.assertEqual(target.tree_forward_calls, [])
        for index, response in enumerate(responses):
            self.assertEqual(response.timing["batch_id"], "batch-test")
            self.assertEqual(response.timing["batch_size"], 2)
            self.assertEqual(response.timing["batch_index"], index)
            self.assertEqual(response.timing["tree_forward_batch_kind"], "tree_attention_batch")
            event = response.timing["target_tree_forward_events"][0]
            self.assertEqual(event["kind"], "tree_attention_batch")
            self.assertEqual(event["metadata"]["shared_batch_event_id"], response.timing["target_tree_forward_events"][0]["shared_batch_event_id"])

    def test_tree_forward_root_guard_falls_back_when_root_choice_is_inconsistent(self) -> None:
        target = TreeForwardScriptedCausalLMRunner({None: 7, 0: 3, 1: 6})
        target.next_tokens_by_prefix = {(1,): 2, (1, 2): 3, (1, 5): 6}
        verifier = TreeVerifier(model=target)
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, None, 5, 1, -0.2, "draft-0"),
                CandidateNode(2, 0, 3, 2, -0.3, "draft-0"),
                CandidateNode(3, 1, 6, 2, -0.4, "draft-0"),
            ],
        )
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-0",
            shape="tree",
            tree=tree,
            metadata={"prefix_ids": [1], "allow_bonus": False},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.payload["accepted_node_ids"], [0, 2])
        self.assertEqual(len(target.tree_forward_calls), 1)
        self.assertEqual(target.batch_calls, [[[1], [1, 2], [1, 5]]])
        kinds = [event["kind"] for event in result.timing["target_tree_forward_events"]]
        self.assertEqual(kinds, ["tree_attention", "tree_root_guard", "tree_choice_batch"])
        self.assertEqual(
            result.timing["target_tree_forward_events"][-1]["metadata"]["fallback_reason"],
            "tree_root_guard_mismatch",
        )

    def test_tree_forward_root_guard_can_be_forced_for_candidate_root(self) -> None:
        target = TreeForwardScriptedCausalLMRunner({None: 5, 0: 3, 1: 6})
        target.next_tokens_by_prefix = {(1,): 2, (1, 2): 3, (1, 5): 6}
        verifier = TreeVerifier(model=target)
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, None, 5, 1, -0.2, "draft-0"),
                CandidateNode(2, 0, 3, 2, -0.3, "draft-0"),
                CandidateNode(3, 1, 6, 2, -0.4, "draft-0"),
            ],
        )
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-0",
            shape="tree",
            tree=tree,
            metadata={"prefix_ids": [1], "allow_bonus": False, "force_root_guard": True},
        )

        result = verifier.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(result.payload["accepted_node_ids"], [0, 2])
        kinds = [event["kind"] for event in result.timing["target_tree_forward_events"]]
        self.assertEqual(kinds, ["tree_attention", "tree_root_guard", "tree_choice_batch"])

    def test_precomputed_root_guard_is_confirmed_before_overriding_tree_forward(self) -> None:
        target = TreeForwardScriptedCausalLMRunner({None: 2})
        target.next_tokens_by_prefix = {(1,): 2}
        verifier = TreeVerifier(model=target)
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[CandidateNode(0, None, 2, 1, -0.1, "draft-0")],
        )
        request = TreeVerifyRequest(
            request_id="request-1",
            proposal_id="proposal-1",
            prefix_ids=[1],
            tree=tree,
            allow_bonus=False,
            metadata={
                "force_root_guard": True,
                "precomputed_root_guard_event": {
                    "kind": "tree_root_guard",
                    "prefix_len": 1,
                    "token_id": 9,
                    "start_ns": 10,
                    "end_ns": 20,
                    "duration_ms": 0.01,
                    "batch_size": 2,
                    "tree_forward_batch_kind": "root_guard_batch",
                },
            },
        )

        response = verifier.verify_request(request)

        self.assertEqual(response.accepted_node_ids, [0])
        self.assertIsNone(response.bonus_token)
        self.assertEqual(response.target_choices[0]["target_token_id"], 2)
        kinds = [event["kind"] for event in response.timing["target_tree_forward_events"]]
        self.assertEqual(kinds, ["tree_attention", "tree_root_guard", "tree_root_guard_confirm"])
        self.assertEqual(
            response.timing["target_tree_forward_events"][-1]["precomputed_root_guard_token_id"],
            9,
        )

    def test_batch_tree_verifier_confirms_stale_precomputed_root_guard_per_request(self) -> None:
        target = BatchTreeForwardScriptedCausalLMRunner(
            {
                ((1,), None): 2,
                ((4,), None): 5,
            }
        )
        target.next_tokens_by_prefix = {(1,): 2}
        verifier = TreeVerifier(model=target)
        request_a = TreeVerifyRequest(
            request_id="request-a",
            proposal_id="proposal-a",
            prefix_ids=[1],
            tree=CandidateTree(
                root_prefix_len=1,
                nodes=[CandidateNode(0, None, 2, 1, -0.1, "draft-0")],
            ),
            allow_bonus=False,
            metadata={
                "force_root_guard": True,
                "precomputed_root_guard_event": {
                    "kind": "tree_root_guard",
                    "prefix_len": 1,
                    "token_id": 9,
                    "start_ns": 10,
                    "end_ns": 20,
                    "duration_ms": 0.01,
                    "batch_size": 2,
                    "tree_forward_batch_kind": "root_guard_batch",
                },
            },
        )
        request_b = TreeVerifyRequest(
            request_id="request-b",
            proposal_id="proposal-b",
            prefix_ids=[4],
            tree=CandidateTree(
                root_prefix_len=1,
                nodes=[CandidateNode(0, None, 5, 1, -0.1, "draft-1")],
            ),
            allow_bonus=False,
        )

        responses = verifier.verify_requests_batch([request_a, request_b], batch_id="batch-guard")

        self.assertEqual([response.accepted_node_ids for response in responses], [[0], [0]])
        self.assertEqual(len(target.tree_forward_batch_calls), 1)
        self.assertEqual(target.single_calls, [[1]])
        self.assertEqual(responses[0].target_choices[0]["target_token_id"], 2)
        self.assertEqual(responses[1].target_choices[0]["target_token_id"], 5)
        kinds_a = [event["kind"] for event in responses[0].timing["target_tree_forward_events"]]
        self.assertEqual(kinds_a, ["tree_attention_batch", "tree_root_guard", "tree_root_guard_confirm"])
        kinds_b = [event["kind"] for event in responses[1].timing["target_tree_forward_events"]]
        self.assertEqual(kinds_b, ["tree_attention_batch"])


class SpecEdgeTreeRuntimeTest(unittest.TestCase):
    """覆盖 tree method 在统一 runtime 里的最小闭环。"""

    def test_tree_runtime_matches_target_greedy_output(self) -> None:
        draft_model = ScriptedCausalLMRunner(
            {
                (1,): 2,
                (1, 2): 4,
                (1, 0): 0,
                (1, 2, 3): 6,
                (1, 2, 3, 6): 8,
                (1, 2, 3, 0): 0,
            }
        )
        target_model = ScriptedCausalLMRunner(
            {
                (1,): 2,
                (1, 2): 3,
                (1, 2, 3): 6,
                (1, 2, 3, 6): 9,
                (1, 0): 0,
                (1, 2, 4): 4,
                (1, 2, 3, 0): 0,
            }
        )
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=4,
            max_len=16,
            eos_token_ids=[9],
        )
        engine = RuntimeEngine(
            candidate_strategy=SpecEdgeTreeCandidateStrategy(default_max_budget=4, default_max_branch_width=2),
            acceptance_policy=SpecEdgeTreeAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=2, max_branches=2)),
            verifier=TreeVerifier(model=target_model),
        )

        result = engine.run(
            run_id="run-tree",
            sessions=[session],
            draft_runners={"draft-worker-0": TopKTreeDraftRunner(model=draft_model, runner_id="draft-worker-0")},
            context=RuntimeContext(
                run_config={"eos_token_ids": [9], "method": "specedge_tree"},
                method_config={"max_budget": 4, "max_branch_width": 2},
            ),
        )

        self.assertEqual(session.generated_ids, [2, 3, 6, 9])
        self.assertEqual(result.request_results[0].output_token_ids, [2, 3, 6, 9])
        self.assertEqual(result.request_results[0].stop_reason, "eos")
        self.assertTrue(
            any(
                event.phase == "draft.topk" and event.span_kind == "detail"
                for event in result.events.events
            )
        )

    def test_async_pipeline_overlaps_proactive_draft_with_verify(self) -> None:
        class DelayedTreeVerifier(TreeVerifier):
            def verify_batch(self, proposals: list[CandidateProposal], context: RuntimeContext | None = None):
                time.sleep(0.05)
                return super().verify_batch(proposals, context)

        draft_model = ScriptedCausalLMRunner(
            {
                (1,): 2,
                (1, 2): 3,
                (1, 0): 0,
                (1, 2, 0): 0,
            }
        )
        target_model = ScriptedCausalLMRunner({(1,): 2, (1, 2): 3})
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=2,
            max_len=16,
            eos_token_ids=[],
        )
        engine = AsyncPipelineRuntimeEngine(
            candidate_strategy=SpecEdgeTreeCandidateStrategy(default_max_budget=2, default_max_branch_width=2),
            acceptance_policy=SpecEdgeTreeAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1, max_branches=2)),
            verifier=DelayedTreeVerifier(model=target_model),
            proactive_policy=SpecEdgeProactiveDraftPolicy(default_max_depth=1, default_max_branch_width=2, default_max_budget=2),
            reconcile_policy=SpecEdgeReconcilePolicy(),
            planning_policy=SpecEdgePipelinePlanningPolicy(initial_depth=1),
        )

        result = engine.run(
            run_id="run-pipeline",
            sessions=[session],
            draft_runners={"draft-worker-0": TopKTreeDraftRunner(model=draft_model, runner_id="draft-worker-0")},
            context=RuntimeContext(
                run_config={"method": "specedge_pipeline"},
                method_config={"max_budget": 2, "max_branch_width": 2, "disable_bonus": True},
            ),
        )

        self.assertEqual(session.generated_ids, [2, 3])
        verify_event = next(event for event in result.events.events if event.phase == "verify.batch_total")
        proactive_event = next(event for event in result.events.events if event.phase == "draft.proactive")
        overlap_ns = min(int(verify_event.end_ns), int(proactive_event.end_ns)) - max(
            int(verify_event.start_ns),
            int(proactive_event.start_ns),
        )
        self.assertGreater(overlap_ns, 0)
        self.assertTrue(
            any(
                event.phase == "pipeline.reconcile" and event.metadata.get("aligned")
                for event in result.events.events
            )
        )

    def test_async_pipeline_executes_proactive_drafts_in_parallel(self) -> None:
        class ImmediateCandidateStrategy:
            def propose(
                self,
                session: GenerationSession,
                draft_runner: Any,
                budget: DraftBudget,
                context: RuntimeContext,
            ) -> CandidateProposal:
                del budget, context
                runner_id = str(draft_runner.runner_id)
                return CandidateProposal(
                    proposal_id=f"proposal:{session.request_id}:{runner_id}",
                    request_id=session.request_id,
                    worker_id=runner_id,
                    shape="linear",
                    tokens=[2],
                    draft_length=1,
                    metadata={"prefix_ids": list(session.prefix_ids), "runner_id": runner_id},
                )

        class SleepyProactivePolicy:
            def __init__(self, sleep_seconds: float) -> None:
                self.sleep_seconds = sleep_seconds

            def propose_proactive(
                self,
                session: GenerationSession,
                proposal: CandidateProposal,
                draft_runner: Any,
                context: RuntimeContext,
            ) -> CandidateProposal:
                del context
                time.sleep(self.sleep_seconds)
                runner_id = str(draft_runner.runner_id)
                return CandidateProposal(
                    proposal_id=f"proactive:{proposal.proposal_id}",
                    request_id=session.request_id,
                    worker_id=runner_id,
                    shape="linear",
                    tokens=[3],
                    draft_length=1,
                    metadata={
                        "prefix_ids": [*session.prefix_ids, 2],
                        "runner_id": runner_id,
                        "tree_node_count": 1,
                    },
                )

        class DelayedVerifier:
            backend_name = "delayed"

            def verify_batch(
                self,
                proposals: list[CandidateProposal],
                context: RuntimeContext | None = None,
            ) -> list[VerificationResult]:
                del context
                time.sleep(0.12)
                return [
                    VerificationResult(
                        request_id=proposal.request_id,
                        proposal_id=proposal.proposal_id,
                        shape=proposal.shape,
                        accepted_prefix_len=1,
                        verified_tokens=[2],
                    )
                    for proposal in proposals
                ]

        class AcceptAllPolicy:
            def accept(
                self,
                proposal: CandidateProposal,
                verification_result: VerificationResult,
                context: RuntimeContext,
            ) -> AcceptResult:
                del verification_result, context
                return AcceptResult(
                    request_id=proposal.request_id,
                    proposal_id=proposal.proposal_id,
                    accepted_tokens=[2],
                    stop_reason="accepted",
                    metadata={"accepted_count": 1},
                )

        class DummyRunner:
            def __init__(self, runner_id: str) -> None:
                self.runner_id = runner_id
                self.metadata = {"runner_id": runner_id}

        sessions = [
            GenerationSession(request_id="request-1", prompt_ids=[1], max_new_tokens=1, max_len=8),
            GenerationSession(request_id="request-2", prompt_ids=[1], max_new_tokens=1, max_len=8),
        ]
        engine = AsyncPipelineRuntimeEngine(
            candidate_strategy=ImmediateCandidateStrategy(),
            acceptance_policy=AcceptAllPolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1), batch_size=2),
            verifier=DelayedVerifier(),
            proactive_policy=SleepyProactivePolicy(sleep_seconds=0.08),
            reconcile_policy=SpecEdgeReconcilePolicy(),
        )

        result = engine.run(
            run_id="run-parallel-proactive",
            sessions=sessions,
            draft_runners={
                "draft-worker-0": DummyRunner("draft-worker-0"),
                "draft-worker-1": DummyRunner("draft-worker-1"),
            },
            context=RuntimeContext(run_config={"method": "specedge_pipeline"}),
        )

        proactive_events = [
            event
            for event in result.events.events
            if event.phase == "draft.proactive" and event.span_kind == "leaf"
        ]
        self.assertEqual(len(proactive_events), 2)
        union_ms = (
            max(int(event.end_ns) for event in proactive_events)
            - min(int(event.start_ns) for event in proactive_events)
        ) / 1_000_000
        summed_ms = sum(event.measured_duration_ms for event in proactive_events)
        self.assertLess(union_ms, summed_ms * 0.8)
        self.assertTrue(all(event.metadata["parallel_proactive"] for event in proactive_events))
        self.assertEqual({event.metadata["proactive_parallelism"] for event in proactive_events}, {2})
        verify_event = next(event for event in result.events.events if event.phase == "verify.batch_total")
        for event in proactive_events:
            overlap_ns = min(int(verify_event.end_ns), int(event.end_ns)) - max(
                int(verify_event.start_ns),
                int(event.start_ns),
            )
            self.assertGreater(overlap_ns, 0)
        self.assertEqual([session.generated_ids for session in sessions], [[2], [2]])

    def test_reconcile_reuses_subtree_when_bonus_matches_proactive_root(self) -> None:
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=4,
            max_len=16,
            generated_ids=[2],
        )
        proactive_tree = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, 0, 3, 2, -0.2, "draft-0"),
                CandidateNode(2, 1, 4, 3, -0.3, "draft-0"),
            ],
        )
        proactive = CandidateProposal(
            proposal_id="proactive-1",
            request_id="request-1",
            worker_id="draft-0",
            shape="tree",
            tree=proactive_tree,
            draft_length=3,
            metadata={"prefix_ids": [1], "allow_bonus": True},
        )

        result = SpecEdgeReconcilePolicy().reconcile(
            session,
            proactive,
            verification_result=None,  # type: ignore[arg-type]
            accept_result=AcceptResult(request_id="request-1", proposal_id="proposal-1", bonus_token=2),
            proactive_proposal=proactive,
            context=RuntimeContext(),
        )

        self.assertTrue(result.aligned)
        self.assertIsNotNone(result.reused_proposal)
        self.assertEqual(result.reused_proposal.metadata["prefix_ids"], [1, 2])
        self.assertEqual([node.token_id for node in result.reused_proposal.tree.nodes], [3, 4])
        self.assertEqual(result.reused_proposal.tree.nodes[0].parent_id, None)

    def test_async_pipeline_reused_proactive_subtree_keeps_committed_prefix(self) -> None:
        draft_model = ScriptedCausalLMRunner(
            {
                (1,): 2,
                (1, 2): 3,
                (1, 2, 3): 4,
                (1, 2, 3, 4): 5,
            }
        )
        target_model = ScriptedCausalLMRunner(
            {
                (1,): 2,
                (1, 2): 3,
                (1, 2, 3): 4,
                (1, 2, 3, 4): 5,
            }
        )
        session = GenerationSession(
            request_id="request-1",
            prompt_ids=[1],
            max_new_tokens=4,
            max_len=16,
            eos_token_ids=[],
        )
        engine = AsyncPipelineRuntimeEngine(
            candidate_strategy=SpecEdgeTreeCandidateStrategy(default_max_budget=1, default_max_branch_width=1),
            acceptance_policy=SpecEdgeTreeAcceptancePolicy(),
            scheduler=RoundRobinRequestScheduler(default_budget=DraftBudget(max_tokens=1, max_branches=1)),
            verifier=TreeVerifier(model=target_model),
            proactive_policy=SpecEdgeProactiveDraftPolicy(default_max_depth=2, default_max_branch_width=1, default_max_budget=2),
            reconcile_policy=SpecEdgeReconcilePolicy(),
        )

        result = engine.run(
            run_id="run-proactive-subtree-prefix",
            sessions=[session],
            draft_runners={"draft-worker-0": TopKTreeDraftRunner(model=draft_model, runner_id="draft-worker-0")},
            context=RuntimeContext(
                run_config={"method": "specedge_pipeline"},
                method_config={
                    "max_budget": 1,
                    "max_branch_width": 1,
                    "proactive_max_depth": 2,
                    "proactive_branch_width": 1,
                    "proactive_max_budget": 2,
                },
            ),
        )

        self.assertEqual(session.generated_ids, [2, 3, 4, 5])
        self.assertEqual(result.request_results[0].output_token_ids, [2, 3, 4, 5])
        self.assertTrue(
            any(
                event.phase == "pipeline.reconcile"
                and event.metadata.get("reason") == "subtree_aligned"
                and event.metadata.get("reused_proposal_id")
                for event in result.events.events
            )
        )
        reused_draft_event = next(
            event
            for event in result.events.events
            if event.phase == "draft.reuse_proactive"
        )
        reused_proposal_id = str(reused_draft_event.proposal_id)
        self.assertTrue(reused_proposal_id.endswith(":subtree:0"))
        reused_verify_event = next(
            event
            for event in result.events.events
            if event.phase == "accept.apply" and event.proposal_id == reused_proposal_id
        )
        target_choices = reused_verify_event.metadata["target_choices"]
        self.assertEqual(target_choices[0]["prefix_len"], 3)
        self.assertEqual(target_choices[0]["target_token_id"], 4)


class HttpTreeVerifierClientTest(unittest.TestCase):
    """验证 3090 HTTP tree client 和 /verify_tree JSON 契约一致。"""

    def test_http_tree_client_posts_tree_verify_request(self) -> None:
        captured: dict[str, Any] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers["Content-Length"])
                captured["path"] = self.path
                captured["payload"] = json.loads(self.rfile.read(content_length).decode("utf-8"))
                body = json.dumps(
                    TreeVerifyResponse(
                        request_id="request-1",
                        proposal_id="proposal-1",
                        accepted_node_ids=[0],
                        target_choices=[{"parent_node_id": None, "target_token_id": 2, "matched_node_id": 0}],
                        bonus_token=3,
                        rejected_node_ids=[1],
                        timing={"server_total_ms": 0.2, "target_tree_forward_total_ms": 0.1},
                    ).to_dict()
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(_shutdown_server, server, thread)

        client = HttpTreeVerifierClient(base_url=f"http://127.0.0.1:{server.server_port}")
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[
                CandidateNode(0, None, 2, 1, -0.1, "draft-0"),
                CandidateNode(1, None, 5, 1, -0.2, "draft-0"),
            ],
        )
        proposal = CandidateProposal(
            proposal_id="proposal-1",
            request_id="request-1",
            worker_id="draft-0",
            shape="tree",
            tree=tree,
            metadata={"prefix_ids": [1], "eos_token_ids": [9]},
        )

        result = client.verify_proposal(proposal, RuntimeContext())

        self.assertEqual(captured["path"], "/verify_tree")
        self.assertEqual(captured["payload"]["prefix_ids"], [1])
        self.assertEqual(captured["payload"]["tree"]["nodes"][0]["token_id"], 2)
        self.assertEqual(result.accepted_prefix_len, 1)
        self.assertEqual(result.bonus_token, 3)
        self.assertEqual(result.timing["response_timing"]["server_total_ms"], 0.2)
        self.assertIn("verify.http_total", [event["phase"] for event in result.timing["client_events"]])

    def test_http_tree_client_posts_batch_verify_request(self) -> None:
        captured: dict[str, Any] = {}

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers["Content-Length"])
                captured["path"] = self.path
                captured["payload"] = json.loads(self.rfile.read(content_length).decode("utf-8"))
                results = []
                for item in captured["payload"]["items"]:
                    request = item["request"]
                    results.append(
                        BatchVerifyResultItem(
                            kind="tree",
                            response=TreeVerifyResponse(
                                request_id=request["request_id"],
                                proposal_id=request["proposal_id"],
                                accepted_node_ids=[0],
                                target_choices=[],
                                rejected_node_ids=[],
                                timing={
                                    "server_total_ms": 0.2,
                                    "target_tree_forward_total_ms": 0.1,
                                    "batch_size": 2,
                                    "tree_forward_batch_kind": "fallback_sequential",
                                },
                            ),
                        )
                    )
                body = json.dumps(
                    BatchVerifyResponse(
                        batch_id=captured["payload"]["batch_id"],
                        results=results,
                        timing={"server_batch_total_ms": 0.4, "batch_size": 2},
                    ).to_dict()
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(_shutdown_server, server, thread)

        client = HttpTreeVerifierClient(base_url=f"http://127.0.0.1:{server.server_port}")
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[CandidateNode(0, None, 2, 1, -0.1, "draft-0")],
        )
        proposals = [
            CandidateProposal(
                proposal_id=f"proposal-{index}",
                request_id=f"request-{index}",
                worker_id="draft-0",
                shape="tree",
                tree=tree,
                metadata={"prefix_ids": [1]},
            )
            for index in range(2)
        ]

        results = client.verify_batch(proposals, RuntimeContext())

        self.assertEqual(captured["path"], "/verify_tree_batch")
        self.assertEqual(len(captured["payload"]["items"]), 2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].timing["response_timing"]["server_batch_total_ms"], 0.4)
        self.assertEqual(results[0].timing["response_timing"]["tree_forward_batch_kind"], "fallback_sequential")

    def test_http_tree_client_chunks_large_batch_verify_requests(self) -> None:
        captured: list[dict[str, Any]] = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers["Content-Length"])
                payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
                captured.append({"path": self.path, "payload": payload})
                if self.path == "/verify_tree":
                    body = json.dumps(
                        TreeVerifyResponse(
                            request_id=payload["request_id"],
                            proposal_id=payload["proposal_id"],
                            accepted_node_ids=[0],
                            target_choices=[],
                            rejected_node_ids=[],
                            timing={"server_total_ms": 0.1},
                        ).to_dict()
                    ).encode("utf-8")
                else:
                    results = []
                    for item in payload["items"]:
                        request = item["request"]
                        results.append(
                            BatchVerifyResultItem(
                                kind="tree",
                                response=TreeVerifyResponse(
                                    request_id=request["request_id"],
                                    proposal_id=request["proposal_id"],
                                    accepted_node_ids=[0],
                                    target_choices=[],
                                    rejected_node_ids=[],
                                    timing={
                                        "server_total_ms": 0.1,
                                        "batch_size": len(payload["items"]),
                                        "tree_forward_batch_kind": "tree_attention_batch",
                                    },
                                ),
                            )
                        )
                    body = json.dumps(
                        BatchVerifyResponse(
                            batch_id=payload["batch_id"],
                            results=results,
                            timing={"server_batch_total_ms": 0.2, "batch_size": len(payload["items"])},
                        ).to_dict()
                    ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: Any) -> None:
                return None

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(_shutdown_server, server, thread)

        client = HttpTreeVerifierClient(
            base_url=f"http://127.0.0.1:{server.server_port}",
            max_batch_items=2,
        )
        tree = CandidateTree(
            root_prefix_len=1,
            nodes=[CandidateNode(0, None, 2, 1, -0.1, "draft-0")],
        )
        proposals = [
            CandidateProposal(
                proposal_id=f"proposal-{index}",
                request_id=f"request-{index}",
                worker_id="draft-0",
                shape="tree",
                tree=tree,
                metadata={"prefix_ids": [1]},
            )
            for index in range(5)
        ]

        results = client.verify_batch(proposals, RuntimeContext())

        self.assertEqual(len(results), 5)
        self.assertEqual([call["path"] for call in captured], ["/verify_tree_batch", "/verify_tree_batch", "/verify_tree"])
        self.assertEqual([len(call["payload"].get("items", [])) for call in captured[:2]], [2, 2])
        self.assertEqual(results[0].timing["client_batch_original_size"], 5)
        self.assertEqual(results[0].timing["client_batch_max_items"], 2)


def _shutdown_server(server: HTTPServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(timeout=2.0)


if __name__ == "__main__":
    unittest.main()
