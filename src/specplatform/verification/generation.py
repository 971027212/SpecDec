from __future__ import annotations

"""HTTP target-only greedy generation client."""

import json
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from specplatform.verification.http_client import _client_event, _network_or_queue_residual_ms
from specplatform.verification.schema import GreedyGenerateRequest, GreedyGenerateResponse


@dataclass
class HttpGreedyGeneratorClient:
    """调用远端 /generate_greedy 的 target-only baseline client。"""

    base_url: str
    endpoint: str = "/generate_greedy"
    timeout_s: float = 60.0

    def generate(
        self,
        *,
        request_id: str,
        prefix_ids: list[int],
        max_new_tokens: int,
        eos_token_ids: list[int] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[GreedyGenerateResponse, dict[str, Any]]:
        """发送 target-only greedy generation 请求，并返回 response 与 client timing。"""
        generate_request = GreedyGenerateRequest(
            request_id=request_id,
            prefix_ids=list(prefix_ids),
            max_new_tokens=int(max_new_tokens),
            eos_token_ids=list(eos_token_ids or []),
            metadata=dict(metadata or {}),
        )
        response, timing = self._post_json(generate_request)
        if response.request_id != generate_request.request_id:
            raise ValueError("GreedyGenerateResponse request_id does not match request.")
        timing["response_timing"] = dict(response.timing)
        timing["network_or_queue_residual_ms"] = _network_or_queue_residual_ms(timing)
        return response, timing

    def _post_json(self, generate_request: GreedyGenerateRequest) -> tuple[GreedyGenerateResponse, dict[str, Any]]:
        """用标准库发 JSON target-only 请求。"""
        url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"
        serialize_start_ns = perf_counter_ns()
        body = json.dumps(generate_request.to_dict()).encode("utf-8")
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
            raise RuntimeError(f"HTTP greedy generator returned {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to reach HTTP greedy generator at {url}: {exc}") from exc
        deserialize_start_ns = perf_counter_ns()
        payload: dict[str, Any] = json.loads(raw.decode("utf-8"))
        response = GreedyGenerateResponse.from_dict(payload)
        deserialize_end_ns = perf_counter_ns()
        timing = {
            "client_events": [
                _client_event("target.client_serialize", serialize_start_ns, serialize_end_ns),
                _client_event(
                    "target.http_total",
                    http_start_ns,
                    http_end_ns,
                    metadata={
                        "url": url,
                        "request_bytes": len(body),
                        "response_bytes": len(raw),
                    },
                ),
                _client_event("target.client_deserialize", deserialize_start_ns, deserialize_end_ns),
            ],
        }
        return response, timing
