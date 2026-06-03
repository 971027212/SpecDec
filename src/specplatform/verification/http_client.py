from __future__ import annotations

"""3090 侧 HTTP verifier client。

runtime 仍然只依赖 VerifierBackend；这个 client 把 verify_proposal 转成
POST /verify_linear 请求，并把 A100 响应还原成 VerificationResult。
"""

import json
from dataclasses import dataclass, field
from threading import Lock
from time import perf_counter_ns
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from specplatform.core import CandidateProposal, RuntimeContext, VerificationResult
from specplatform.verification.base import VerifierBackend
from specplatform.verification.linear import _eos_token_ids, _proposal_prefix_ids, _validate_response_for_request
from specplatform.verification.schema import (
    BatchVerifyItem,
    BatchVerifyRequest,
    BatchVerifyResponse,
    LinearVerifyRequest,
    LinearVerifyResponse,
    TreeVerifyRequest,
    TreeVerifyResponse,
)
from specplatform.verification.tree import (
    _eos_token_ids as _tree_eos_token_ids,
    _proposal_prefix_ids as _tree_proposal_prefix_ids,
    _validate_response_for_request as _validate_tree_response_for_request,
)


@dataclass(frozen=True)
class TransportProfile:
    """HTTP transport 的观测/建模配置，不对真实网络限速。"""

    mode: str = "observe"
    uplink_mbps: float | None = None
    downlink_mbps: float | None = None
    rtt_ms: float | None = None

    def upload_ms(self, byte_count: int) -> float | None:
        return _modeled_transfer_ms(byte_count, self.uplink_mbps)

    def downlink_ms(self, byte_count: int) -> float | None:
        return _modeled_transfer_ms(byte_count, self.downlink_mbps)


@dataclass
class HttpLinearVerifierClient(VerifierBackend):
    """调用远端 /verify_linear 的最小 HTTP verifier backend。"""

    base_url: str
    endpoint: str = "/verify_linear"
    batch_endpoint: str = "/verify_linear_batch"
    timeout_s: float = 60.0
    backend_name: str = "linear_http"
    transport_profile: TransportProfile = field(default_factory=TransportProfile)

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        """发送单个 proposal 到远端 target verifier。"""
        if proposal.shape != "linear":
            raise ValueError("HttpLinearVerifierClient only supports linear proposals.")

        verify_request = LinearVerifyRequest(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            prefix_ids=_proposal_prefix_ids(proposal),
            draft_tokens=list(proposal.tokens),
            eos_token_ids=_eos_token_ids(proposal, context),
            allow_bonus=bool(proposal.metadata.get("allow_bonus", True)),
            metadata=dict(proposal.metadata),
        )
        response, timing = self._post_json(verify_request)
        validate_start_ns = perf_counter_ns()
        _validate_response_for_request(response, verify_request)
        validate_end_ns = perf_counter_ns()
        timing["client_events"].append(
            _client_event(
                "verify.response_validate",
                validate_start_ns,
                validate_end_ns,
            )
        )
        timing["response_timing"] = dict(response.timing)
        timing["network_or_queue_residual_ms"] = _network_or_queue_residual_ms(timing)
        return VerificationResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            shape=proposal.shape,
            accepted_prefix_len=response.accepted_prefix_len,
            verified_tokens=list(response.verified_tokens),
            bonus_token=response.bonus_token,
            timing=timing,
            payload=response.to_dict(),
            metadata={
                "backend_name": self.backend_name,
                **dict(response.metadata),
                "timing": timing,
            },
        )

    def verify_batch(
        self,
        proposals: list[CandidateProposal],
        context: RuntimeContext | None = None,
    ) -> list[VerificationResult]:
        """用 batch endpoint 验证一组 linear proposals。"""
        if len(proposals) <= 1:
            return [self.verify_proposal(proposal, context) for proposal in proposals]
        requests = [
            LinearVerifyRequest(
                request_id=proposal.request_id,
                proposal_id=proposal.proposal_id,
                prefix_ids=_proposal_prefix_ids(proposal),
                draft_tokens=list(proposal.tokens),
                eos_token_ids=_eos_token_ids(proposal, context),
                allow_bonus=bool(proposal.metadata.get("allow_bonus", True)),
                metadata=dict(proposal.metadata),
            )
            for proposal in proposals
        ]
        batch_request = BatchVerifyRequest(
            batch_id=_batch_id_for_proposals(proposals),
            items=[BatchVerifyItem(kind="linear", request=request) for request in requests],
            metadata={"proposal_ids": [proposal.proposal_id for proposal in proposals]},
        )
        batch_response, timing = self._post_batch_json(batch_request)
        responses = [result.response for result in batch_response.results]
        if len(responses) != len(requests):
            raise ValueError("Batch linear verifier returned a different number of responses.")
        results: list[VerificationResult] = []
        for proposal, request, response in zip(proposals, requests, responses):
            if not isinstance(response, LinearVerifyResponse):
                raise ValueError("Batch linear verifier returned a non-linear response.")
            validate_start_ns = perf_counter_ns()
            _validate_response_for_request(response, request)
            validate_end_ns = perf_counter_ns()
            result_timing = _timing_for_batch_item(
                timing,
                batch_response=batch_response,
                response_timing=dict(response.timing),
                validate_start_ns=validate_start_ns,
                validate_end_ns=validate_end_ns,
            )
            results.append(
                VerificationResult(
                    request_id=proposal.request_id,
                    proposal_id=proposal.proposal_id,
                    shape=proposal.shape,
                    accepted_prefix_len=response.accepted_prefix_len,
                    verified_tokens=list(response.verified_tokens),
                    bonus_token=response.bonus_token,
                    timing=result_timing,
                    payload=response.to_dict(),
                    metadata={
                        "backend_name": self.backend_name,
                        **dict(response.metadata),
                        "timing": result_timing,
                    },
                )
            )
        return results

    def _post_json(self, verify_request: LinearVerifyRequest) -> tuple[LinearVerifyResponse, dict[str, Any]]:
        """用标准库发 JSON 请求，减少服务端/客户端额外依赖。"""
        url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"
        serialize_start_ns = perf_counter_ns()
        body = json.dumps(verify_request.to_dict()).encode("utf-8")
        serialize_end_ns = perf_counter_ns()
        http_request = urllib_request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        http_start_ns = perf_counter_ns()
        try:
            with urllib_request.urlopen(http_request, timeout=self.timeout_s) as response:
                raw = response.read()
            http_end_ns = perf_counter_ns()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP verifier returned {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to reach HTTP verifier at {url}: {exc}") from exc
        deserialize_start_ns = perf_counter_ns()
        payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
        response = LinearVerifyResponse.from_dict(payload)
        deserialize_end_ns = perf_counter_ns()
        timing = {
            **_transport_timing(len(body), len(raw), self.transport_profile),
            "client_events": [
                _client_event("verify.client_serialize", serialize_start_ns, serialize_end_ns),
                _client_event(
                    "verify.http_total",
                    http_start_ns,
                    http_end_ns,
                    metadata={
                        "url": url,
                        "request_bytes": len(body),
                        "response_bytes": len(raw),
                        "modeled_upload_ms": self.transport_profile.upload_ms(len(body)),
                        "modeled_downlink_ms": self.transport_profile.downlink_ms(len(raw)),
                    },
                ),
                _client_event("verify.client_deserialize", deserialize_start_ns, deserialize_end_ns),
            ],
        }
        _attach_client_duration_fields(timing, "verify")
        return response, timing

    def _post_batch_json(self, batch_request: BatchVerifyRequest) -> tuple[BatchVerifyResponse, dict[str, Any]]:
        """发送 JSON batch 请求。"""
        url = f"{self.base_url.rstrip('/')}/{self.batch_endpoint.lstrip('/')}"
        return _post_batch_json(url, batch_request, self.timeout_s, self.transport_profile, "verify")


@dataclass
class HttpLinearVerifierPoolClient(VerifierBackend):
    """Dispatch whole linear verify batches across multiple target replicas."""

    base_urls: list[str]
    endpoint: str = "/verify_linear"
    batch_endpoint: str = "/verify_linear_batch"
    timeout_s: float = 60.0
    backend_name: str = "linear_http_pool"
    transport_profile: TransportProfile = field(default_factory=TransportProfile)
    _next_index: int = field(default=0, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)
    _clients: list[HttpLinearVerifierClient] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        urls = [str(url).rstrip("/") for url in self.base_urls if str(url).strip()]
        if not urls:
            raise ValueError("HttpLinearVerifierPoolClient requires at least one base URL.")
        self.base_urls = urls
        self._clients = [
            HttpLinearVerifierClient(
                base_url=url,
                endpoint=self.endpoint,
                batch_endpoint=self.batch_endpoint,
                timeout_s=self.timeout_s,
                backend_name="linear_http_pool_member",
                transport_profile=self.transport_profile,
            )
            for url in urls
        ]

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        client_index, client = self._next_client()
        result = client.verify_proposal(proposal, context)
        self._annotate_results([result], client_index=client_index, batch_size=1)
        return result

    def verify_batch(
        self,
        proposals: list[CandidateProposal],
        context: RuntimeContext | None = None,
    ) -> list[VerificationResult]:
        client_index, client = self._next_client()
        results = client.verify_batch(proposals, context)
        self._annotate_results(results, client_index=client_index, batch_size=len(proposals))
        return results

    def _next_client(self) -> tuple[int, HttpLinearVerifierClient]:
        with self._lock:
            client_index = self._next_index % len(self._clients)
            self._next_index += 1
        return client_index, self._clients[client_index]

    def _annotate_results(
        self,
        results: list[VerificationResult],
        *,
        client_index: int,
        batch_size: int,
    ) -> None:
        url = self.base_urls[client_index]
        for result in results:
            result.metadata["backend_name"] = self.backend_name
            result.metadata["target_pool_url"] = url
            result.metadata["target_pool_index"] = client_index
            result.metadata["target_pool_size"] = len(self.base_urls)
            result.metadata["target_pool_batch_size"] = int(batch_size)
            result.timing["target_pool_url"] = url
            result.timing["target_pool_index"] = client_index
            result.timing["target_pool_size"] = len(self.base_urls)
            result.timing["target_pool_batch_size"] = int(batch_size)
            for event in result.timing.get("client_events", []):
                metadata = event.setdefault("metadata", {})
                metadata["target_pool_url"] = url
                metadata["target_pool_index"] = client_index
                metadata["target_pool_size"] = len(self.base_urls)


@dataclass
class HttpTreeVerifierClient(VerifierBackend):
    """调用远端 /verify_tree 的 HTTP verifier backend。"""

    base_url: str
    endpoint: str = "/verify_tree"
    batch_endpoint: str = "/verify_tree_batch"
    timeout_s: float = 60.0
    backend_name: str = "tree_http"
    transport_profile: TransportProfile = field(default_factory=TransportProfile)
    max_batch_items: int | None = 8

    def verify_proposal(
        self,
        proposal: CandidateProposal,
        context: RuntimeContext | None = None,
    ) -> VerificationResult:
        """发送单个 tree proposal 到远端 target verifier。"""
        if proposal.shape != "tree" or proposal.tree is None:
            raise ValueError("HttpTreeVerifierClient only supports tree proposals.")

        verify_request = TreeVerifyRequest(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            prefix_ids=_tree_proposal_prefix_ids(proposal),
            tree=proposal.tree,
            eos_token_ids=_tree_eos_token_ids(proposal, context),
            allow_bonus=bool(proposal.metadata.get("allow_bonus", True)),
            metadata=dict(proposal.metadata),
        )
        response, timing = self._post_json(verify_request)
        validate_start_ns = perf_counter_ns()
        _validate_tree_response_for_request(response, verify_request)
        validate_end_ns = perf_counter_ns()
        timing["client_events"].append(
            _client_event(
                "verify.response_validate",
                validate_start_ns,
                validate_end_ns,
            )
        )
        timing["response_timing"] = dict(response.timing)
        timing["network_or_queue_residual_ms"] = _network_or_queue_residual_ms(timing)
        nodes_by_id = proposal.tree.nodes_by_id()
        accepted_tokens = [nodes_by_id[int(node_id)].token_id for node_id in response.accepted_node_ids]
        return VerificationResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            shape=proposal.shape,
            accepted_prefix_len=len(response.accepted_node_ids),
            verified_tokens=list(accepted_tokens),
            bonus_token=response.bonus_token,
            timing=timing,
            payload=response.to_dict(),
            metadata={
                "backend_name": self.backend_name,
                **dict(response.metadata),
                "timing": timing,
            },
        )

    def verify_batch(
        self,
        proposals: list[CandidateProposal],
        context: RuntimeContext | None = None,
    ) -> list[VerificationResult]:
        """用 batch endpoint 验证一组 tree proposals。"""
        max_batch_items = None if self.max_batch_items is None else int(self.max_batch_items)
        if max_batch_items is not None and max_batch_items > 0 and len(proposals) > max_batch_items:
            chunks = [
                proposals[index : index + max_batch_items]
                for index in range(0, len(proposals), max_batch_items)
            ]
            results: list[VerificationResult] = []
            for chunk_index, chunk in enumerate(chunks):
                for result in self.verify_batch(chunk, context):
                    result.timing["client_batch_chunk_index"] = chunk_index
                    result.timing["client_batch_chunk_count"] = len(chunks)
                    result.timing["client_batch_original_size"] = len(proposals)
                    result.timing["client_batch_max_items"] = max_batch_items
                    results.append(result)
            return results
        if len(proposals) <= 1:
            return [self.verify_proposal(proposal, context) for proposal in proposals]
        requests = [
            TreeVerifyRequest(
                request_id=proposal.request_id,
                proposal_id=proposal.proposal_id,
                prefix_ids=_tree_proposal_prefix_ids(proposal),
                tree=proposal.tree,
                eos_token_ids=_tree_eos_token_ids(proposal, context),
                allow_bonus=bool(proposal.metadata.get("allow_bonus", True)),
                metadata=dict(proposal.metadata),
            )
            for proposal in proposals
        ]
        batch_request = BatchVerifyRequest(
            batch_id=_batch_id_for_proposals(proposals),
            items=[BatchVerifyItem(kind="tree", request=request) for request in requests],
            metadata={"proposal_ids": [proposal.proposal_id for proposal in proposals]},
        )
        batch_response, timing = self._post_batch_json(batch_request)
        responses = [result.response for result in batch_response.results]
        if len(responses) != len(requests):
            raise ValueError("Batch tree verifier returned a different number of responses.")
        results: list[VerificationResult] = []
        for proposal, request, response in zip(proposals, requests, responses):
            if not isinstance(response, TreeVerifyResponse):
                raise ValueError("Batch tree verifier returned a non-tree response.")
            validate_start_ns = perf_counter_ns()
            _validate_tree_response_for_request(response, request)
            validate_end_ns = perf_counter_ns()
            result_timing = _timing_for_batch_item(
                timing,
                batch_response=batch_response,
                response_timing=dict(response.timing),
                validate_start_ns=validate_start_ns,
                validate_end_ns=validate_end_ns,
            )
            nodes_by_id = proposal.tree.nodes_by_id()
            accepted_tokens = [nodes_by_id[int(node_id)].token_id for node_id in response.accepted_node_ids]
            results.append(
                VerificationResult(
                    request_id=proposal.request_id,
                    proposal_id=proposal.proposal_id,
                    shape=proposal.shape,
                    accepted_prefix_len=len(response.accepted_node_ids),
                    verified_tokens=list(accepted_tokens),
                    bonus_token=response.bonus_token,
                    timing=result_timing,
                    payload=response.to_dict(),
                    metadata={
                        "backend_name": self.backend_name,
                        **dict(response.metadata),
                        "timing": result_timing,
                    },
                )
            )
        return results

    def _post_json(self, verify_request: TreeVerifyRequest) -> tuple[TreeVerifyResponse, dict[str, Any]]:
        """用标准库发 JSON tree verify 请求。"""
        url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"
        serialize_start_ns = perf_counter_ns()
        body = json.dumps(verify_request.to_dict()).encode("utf-8")
        serialize_end_ns = perf_counter_ns()
        http_request = urllib_request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        http_start_ns = perf_counter_ns()
        try:
            with urllib_request.urlopen(http_request, timeout=self.timeout_s) as response:
                raw = response.read()
            http_end_ns = perf_counter_ns()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP tree verifier returned {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to reach HTTP tree verifier at {url}: {exc}") from exc
        deserialize_start_ns = perf_counter_ns()
        payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
        response = TreeVerifyResponse.from_dict(payload)
        deserialize_end_ns = perf_counter_ns()
        timing = {
            **_transport_timing(len(body), len(raw), self.transport_profile),
            "client_events": [
                _client_event("verify.client_serialize", serialize_start_ns, serialize_end_ns),
                _client_event(
                    "verify.http_total",
                    http_start_ns,
                    http_end_ns,
                    metadata={
                        "url": url,
                        "request_bytes": len(body),
                        "response_bytes": len(raw),
                        "modeled_upload_ms": self.transport_profile.upload_ms(len(body)),
                        "modeled_downlink_ms": self.transport_profile.downlink_ms(len(raw)),
                    },
                ),
                _client_event("verify.client_deserialize", deserialize_start_ns, deserialize_end_ns),
            ],
        }
        _attach_client_duration_fields(timing, "verify")
        return response, timing

    def _post_batch_json(self, batch_request: BatchVerifyRequest) -> tuple[BatchVerifyResponse, dict[str, Any]]:
        """发送 JSON tree batch 请求。"""
        url = f"{self.base_url.rstrip('/')}/{self.batch_endpoint.lstrip('/')}"
        return _post_batch_json(url, batch_request, self.timeout_s, self.transport_profile, "verify")


def _client_event(
    phase: str,
    start_ns: int,
    end_ns: int,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """构造 3090 侧 HTTP client 细粒度事件。"""
    return {
        "phase": phase,
        "start_ns": int(start_ns),
        "end_ns": int(end_ns),
        "duration_ms": (int(end_ns) - int(start_ns)) / 1_000_000,
        "metadata": dict(metadata or {}),
    }


def _network_or_queue_residual_ms(timing: dict[str, Any]) -> float | None:
    """估算 HTTP 总耗时中无法由 A100 server_total 解释的部分。"""
    response_timing = dict(timing.get("response_timing") or {})
    server_total = response_timing.get("server_batch_total_ms", response_timing.get("server_total_ms"))
    if server_total is None:
        return None
    http_total = None
    for event in timing.get("client_events", []):
        if event.get("phase") == "verify.http_total":
            http_total = float(event.get("duration_ms", 0.0))
            break
    if http_total is None:
        return None
    return http_total - float(server_total)


def _batch_id_for_proposals(proposals: list[CandidateProposal]) -> str:
    return "batch:" + ",".join(proposal.proposal_id for proposal in proposals)


def _modeled_transfer_ms(byte_count: int, mbps: float | None) -> float | None:
    if mbps is None or mbps <= 0:
        return None
    return (int(byte_count) * 8.0) / (float(mbps) * 1_000_000.0) * 1000.0


def _transport_timing(
    request_bytes: int,
    response_bytes: int,
    profile: TransportProfile,
) -> dict[str, Any]:
    upload_ms = profile.upload_ms(request_bytes)
    downlink_ms = profile.downlink_ms(response_bytes)
    return {
        "transport_profile": {
            "mode": profile.mode,
            "uplink_mbps": profile.uplink_mbps,
            "downlink_mbps": profile.downlink_mbps,
            "rtt_ms": profile.rtt_ms,
        },
        "request_bytes": int(request_bytes),
        "response_bytes": int(response_bytes),
        "modeled_upload_ms": upload_ms,
        "modeled_downlink_ms": downlink_ms,
    }


def _timing_for_batch_item(
    batch_timing: dict[str, Any],
    *,
    batch_response: BatchVerifyResponse,
    response_timing: dict[str, Any],
    validate_start_ns: int,
    validate_end_ns: int,
) -> dict[str, Any]:
    timing = dict(batch_timing)
    merged_response_timing = {
        **dict(batch_response.timing),
        **dict(response_timing),
        "batch_id": batch_response.batch_id,
    }
    timing["client_events"] = [
        *list(batch_timing.get("client_events", [])),
        _client_event("verify.response_validate", validate_start_ns, validate_end_ns),
    ]
    timing["response_timing"] = merged_response_timing
    timing["network_or_queue_residual_ms"] = _network_or_queue_residual_ms(timing)
    return timing


def _post_batch_json(
    url: str,
    batch_request: BatchVerifyRequest,
    timeout_s: float,
    transport_profile: TransportProfile,
    phase_prefix: str,
) -> tuple[BatchVerifyResponse, dict[str, Any]]:
    serialize_start_ns = perf_counter_ns()
    body = json.dumps(batch_request.to_dict()).encode("utf-8")
    serialize_end_ns = perf_counter_ns()
    http_request = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    http_start_ns = perf_counter_ns()
    try:
        with urllib_request.urlopen(http_request, timeout=timeout_s) as response:
            raw = response.read()
        http_end_ns = perf_counter_ns()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP batch verifier returned {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to reach HTTP batch verifier at {url}: {exc}") from exc
    deserialize_start_ns = perf_counter_ns()
    payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
    response = BatchVerifyResponse.from_dict(payload)
    deserialize_end_ns = perf_counter_ns()
    timing = {
        **_transport_timing(len(body), len(raw), transport_profile),
        "client_events": [
            _client_event(f"{phase_prefix}.client_serialize", serialize_start_ns, serialize_end_ns),
            _client_event(
                f"{phase_prefix}.http_total",
                http_start_ns,
                http_end_ns,
                metadata={
                    "url": url,
                    "request_bytes": len(body),
                    "response_bytes": len(raw),
                    "batch_id": batch_request.batch_id,
                    "batch_size": len(batch_request.items),
                    "modeled_upload_ms": transport_profile.upload_ms(len(body)),
                    "modeled_downlink_ms": transport_profile.downlink_ms(len(raw)),
                },
            ),
            _client_event(f"{phase_prefix}.client_deserialize", deserialize_start_ns, deserialize_end_ns),
        ],
    }
    _attach_client_duration_fields(timing, phase_prefix)
    return response, timing


def _attach_client_duration_fields(timing: dict[str, Any], phase_prefix: str) -> None:
    for event in timing.get("client_events", []):
        phase = event.get("phase")
        if phase == f"{phase_prefix}.client_serialize":
            timing["client_serialize_ms"] = float(event.get("duration_ms") or 0.0)
        elif phase == f"{phase_prefix}.http_total":
            timing["client_http_total_ms"] = float(event.get("duration_ms") or 0.0)
        elif phase == f"{phase_prefix}.client_deserialize":
            timing["client_deserialize_ms"] = float(event.get("duration_ms") or 0.0)
