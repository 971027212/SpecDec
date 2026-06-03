from __future__ import annotations

"""A100 侧最小 target verifier HTTP 服务。

默认加载 /data/chajiahao/hf_models/Qwen3-14B，提供 POST /verify_linear、
POST /verify_tree 和 POST /generate_greedy。
服务只做 greedy target verification/generation，不决定 acceptance。
"""

import argparse
import json
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from time import perf_counter_ns
from typing import Any

from specplatform.model import load_causal_lm_runner
from specplatform.verification import (
    BatchVerifyItem,
    BatchVerifyRequest,
    BatchVerifyResponse,
    BatchVerifyResultItem,
    GreedyGenerateRequest,
    GreedyGenerateResponse,
    LinearVerifier,
    LinearVerifyRequest,
    TreeVerifier,
    TreeVerifyRequest,
)


def build_parser() -> argparse.ArgumentParser:
    """解析服务启动参数。"""
    parser = argparse.ArgumentParser(description="Run minimal A100 linear verifier service.")
    parser.add_argument("--model-path", default="/data/chajiahao/hf_models/Qwen3-14B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument(
        "--backend",
        default="hf_cached",
        choices=["hf_eager", "eager", "hf_cached", "cached", "qwen3_graph", "graph"],
        help="Target model backend. Use hf_cached for SLED KV-cache single-pass verification.",
    )
    parser.add_argument(
        "--no-backend-fallback",
        action="store_true",
        help="Fail instead of falling back when a requested graph backend is unavailable.",
    )
    return parser


def make_handler(linear_verifier: LinearVerifier, tree_verifier: TreeVerifier) -> type[BaseHTTPRequestHandler]:
    """把 verifier 注入 HTTP handler；handler 自身不保存算法状态。"""

    class VerifyHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server 固定方法名
            if self.path == "/verify_linear":
                self._handle_verify_linear()
                return
            if self.path == "/verify_linear_batch":
                self._handle_verify_linear_batch()
                return
            if self.path == "/verify_tree":
                self._handle_verify_tree()
                return
            if self.path == "/verify_tree_batch":
                self._handle_verify_tree_batch()
                return
            if self.path == "/generate_greedy":
                self._handle_generate_greedy()
                return
            self._send_json({"error": "not found"}, status=404)

        def _handle_verify_linear(self) -> None:
            server_start_ns = perf_counter_ns()
            try:
                read_start_ns = perf_counter_ns()
                raw = self._read_body()
                read_end_ns = perf_counter_ns()
                parse_start_ns = perf_counter_ns()
                payload = json.loads(raw)
                parse_end_ns = perf_counter_ns()
                verify_request = LinearVerifyRequest.from_dict(payload)
                verify_start_ns = perf_counter_ns()
                response = linear_verifier.verify_request(verify_request)
                verify_end_ns = perf_counter_ns()
                timing = {
                    **dict(response.timing),
                    "request_read_ms": _duration_ms(read_start_ns, read_end_ns),
                    "request_parse_ms": _duration_ms(parse_start_ns, parse_end_ns),
                    "queue_wait_ms": 0.0,
                    "batch_wait_ms": 0.0,
                    "batch_id": str(verify_request.metadata.get("batch_id", verify_request.proposal_id)),
                    "batch_size": 1,
                    "verify_total_ms": _duration_ms(verify_start_ns, verify_end_ns),
                }
                response = replace(response, timing=timing)
                serialize_start_ns = perf_counter_ns()
                _ = json.dumps(response.to_dict(), ensure_ascii=False).encode("utf-8")
                serialize_end_ns = perf_counter_ns()
                timing["response_serialize_ms"] = _duration_ms(serialize_start_ns, serialize_end_ns)
                timing["server_total_ms"] = _duration_ms(server_start_ns, perf_counter_ns())
                self._send_json(replace(response, timing=timing).to_dict(), status=200)
            except Exception as exc:  # pragma: no cover - 服务脚本兜底返回错误
                self._send_json({"error": str(exc)}, status=500)

        def _handle_verify_linear_batch(self) -> None:
            server_start_ns = perf_counter_ns()
            try:
                read_start_ns = perf_counter_ns()
                raw = self._read_body()
                read_end_ns = perf_counter_ns()
                parse_start_ns = perf_counter_ns()
                payload = json.loads(raw)
                parse_end_ns = perf_counter_ns()
                batch_request = BatchVerifyRequest.from_dict(payload)
                verify_start_ns = perf_counter_ns()
                linear_requests: list[LinearVerifyRequest] = []
                for item in batch_request.items:
                    if item.kind != "linear" or not isinstance(item.request, LinearVerifyRequest):
                        raise ValueError("POST /verify_linear_batch only accepts linear items.")
                    linear_requests.append(item.request)
                responses = linear_verifier.verify_requests_batch(
                    linear_requests,
                    batch_id=batch_request.batch_id,
                )
                results: list[BatchVerifyResultItem] = []
                for index, response in enumerate(responses):
                    timing = {
                        **dict(response.timing),
                        "queue_arrival_ns": server_start_ns,
                        "queue_wait_ms": 0.0,
                        "batch_wait_ms": 0.0,
                        "batch_id": batch_request.batch_id,
                        "batch_size": len(batch_request.items),
                        "batch_index": index,
                        "verify_total_ms": float(response.timing.get("target_forward_total_ms") or 0.0),
                    }
                    results.append(
                        BatchVerifyResultItem(
                            kind="linear",
                            response=replace(response, timing=timing),
                        )
                    )
                verify_end_ns = perf_counter_ns()
                timing = {
                    "request_read_ms": _duration_ms(read_start_ns, read_end_ns),
                    "request_parse_ms": _duration_ms(parse_start_ns, parse_end_ns),
                    "queue_wait_ms": 0.0,
                    "batch_wait_ms": 0.0,
                    "batch_id": batch_request.batch_id,
                    "batch_size": len(batch_request.items),
                    "server_batch_verify_ms": _duration_ms(verify_start_ns, verify_end_ns),
                    "server_batch_total_ms": _duration_ms(server_start_ns, perf_counter_ns()),
                    "tree_forward_batch_kind": "not_applicable",
                    "linear_forward_batch_kind": _linear_forward_batch_kind(responses),
                }
                response = BatchVerifyResponse(
                    batch_id=batch_request.batch_id,
                    results=results,
                    metadata={"kind": "linear"},
                    timing=timing,
                )
                serialize_start_ns = perf_counter_ns()
                _ = json.dumps(response.to_dict(), ensure_ascii=False).encode("utf-8")
                serialize_end_ns = perf_counter_ns()
                timing["response_serialize_ms"] = _duration_ms(serialize_start_ns, serialize_end_ns)
                timing["server_batch_total_ms"] = _duration_ms(server_start_ns, perf_counter_ns())
                self._send_json(replace(response, timing=timing).to_dict(), status=200)
            except Exception as exc:  # pragma: no cover - 服务脚本兜底返回错误
                self._send_json({"error": str(exc)}, status=500)

        def _handle_verify_tree(self) -> None:
            server_start_ns = perf_counter_ns()
            try:
                read_start_ns = perf_counter_ns()
                raw = self._read_body()
                read_end_ns = perf_counter_ns()
                parse_start_ns = perf_counter_ns()
                payload = json.loads(raw)
                parse_end_ns = perf_counter_ns()
                verify_request = TreeVerifyRequest.from_dict(payload)
                verify_start_ns = perf_counter_ns()
                response = tree_verifier.verify_request(verify_request)
                verify_end_ns = perf_counter_ns()
                timing = {
                    **dict(response.timing),
                    "request_read_ms": _duration_ms(read_start_ns, read_end_ns),
                    "request_parse_ms": _duration_ms(parse_start_ns, parse_end_ns),
                    "queue_wait_ms": 0.0,
                    "batch_wait_ms": 0.0,
                    "batch_id": str(verify_request.metadata.get("batch_id", verify_request.proposal_id)),
                    "batch_size": 1,
                    "verify_total_ms": _duration_ms(verify_start_ns, verify_end_ns),
                }
                response = replace(response, timing=timing)
                serialize_start_ns = perf_counter_ns()
                _ = json.dumps(response.to_dict(), ensure_ascii=False).encode("utf-8")
                serialize_end_ns = perf_counter_ns()
                timing["response_serialize_ms"] = _duration_ms(serialize_start_ns, serialize_end_ns)
                timing["server_total_ms"] = _duration_ms(server_start_ns, perf_counter_ns())
                self._send_json(replace(response, timing=timing).to_dict(), status=200)
            except Exception as exc:  # pragma: no cover - 服务脚本兜底返回错误
                self._send_json({"error": str(exc)}, status=500)

        def _handle_verify_tree_batch(self) -> None:
            server_start_ns = perf_counter_ns()
            try:
                read_start_ns = perf_counter_ns()
                raw = self._read_body()
                read_end_ns = perf_counter_ns()
                parse_start_ns = perf_counter_ns()
                payload = json.loads(raw)
                parse_end_ns = perf_counter_ns()
                batch_request = BatchVerifyRequest.from_dict(payload)
                batch_items = _with_batched_root_guard(tree_verifier, batch_request.items)
                verify_start_ns = perf_counter_ns()
                requests: list[TreeVerifyRequest] = []
                for item in batch_items:
                    if item.kind != "tree" or not isinstance(item.request, TreeVerifyRequest):
                        raise ValueError("POST /verify_tree_batch only accepts tree items.")
                    requests.append(item.request)
                responses = tree_verifier.verify_requests_batch(
                    requests,
                    batch_id=batch_request.batch_id,
                )
                verify_end_ns = perf_counter_ns()
                batch_verify_ms = _duration_ms(verify_start_ns, verify_end_ns)
                results: list[BatchVerifyResultItem] = []
                tree_forward_batch_kinds = {
                    str(response.timing.get("tree_forward_batch_kind"))
                    for response in responses
                    if response.timing.get("tree_forward_batch_kind") is not None
                }
                tree_forward_batch_kind = (
                    sorted(tree_forward_batch_kinds)[0]
                    if len(tree_forward_batch_kinds) == 1
                    else ",".join(sorted(tree_forward_batch_kinds))
                ) or "unknown"
                for index, response in enumerate(responses):
                    timing = {
                        **dict(response.timing),
                        "queue_arrival_ns": server_start_ns,
                        "queue_wait_ms": 0.0,
                        "batch_wait_ms": 0.0,
                        "batch_id": batch_request.batch_id,
                        "batch_size": len(batch_request.items),
                        "batch_index": index,
                        "verify_total_ms": float(response.timing.get("verify_total_ms") or batch_verify_ms / max(1, len(responses))),
                        "tree_forward_batch_kind": tree_forward_batch_kind,
                    }
                    results.append(
                        BatchVerifyResultItem(
                            kind="tree",
                            response=replace(response, timing=timing),
                        )
                    )
                timing = {
                    "request_read_ms": _duration_ms(read_start_ns, read_end_ns),
                    "request_parse_ms": _duration_ms(parse_start_ns, parse_end_ns),
                    "queue_wait_ms": 0.0,
                    "batch_wait_ms": 0.0,
                    "batch_id": batch_request.batch_id,
                    "batch_size": len(batch_items),
                    "server_batch_verify_ms": batch_verify_ms,
                    "server_batch_total_ms": _duration_ms(server_start_ns, perf_counter_ns()),
                    "tree_forward_batch_kind": tree_forward_batch_kind,
                }
                response = BatchVerifyResponse(
                    batch_id=batch_request.batch_id,
                    results=results,
                    metadata={"kind": "tree"},
                    timing=timing,
                )
                serialize_start_ns = perf_counter_ns()
                _ = json.dumps(response.to_dict(), ensure_ascii=False).encode("utf-8")
                serialize_end_ns = perf_counter_ns()
                timing["response_serialize_ms"] = _duration_ms(serialize_start_ns, serialize_end_ns)
                timing["server_batch_total_ms"] = _duration_ms(server_start_ns, perf_counter_ns())
                self._send_json(replace(response, timing=timing).to_dict(), status=200)
            except Exception as exc:  # pragma: no cover - 服务脚本兜底返回错误
                self._send_json({"error": str(exc)}, status=500)

        def _handle_generate_greedy(self) -> None:
            server_start_ns = perf_counter_ns()
            try:
                read_start_ns = perf_counter_ns()
                raw = self._read_body()
                read_end_ns = perf_counter_ns()
                parse_start_ns = perf_counter_ns()
                payload = json.loads(raw)
                parse_end_ns = perf_counter_ns()
                generate_request = GreedyGenerateRequest.from_dict(payload)
                generate_start_ns = perf_counter_ns()
                response = _generate_greedy(tree_verifier, generate_request)
                generate_end_ns = perf_counter_ns()
                timing = {
                    **dict(response.timing),
                    "request_read_ms": _duration_ms(read_start_ns, read_end_ns),
                    "request_parse_ms": _duration_ms(parse_start_ns, parse_end_ns),
                    "queue_wait_ms": 0.0,
                    "batch_wait_ms": 0.0,
                    "batch_id": str(generate_request.metadata.get("batch_id", generate_request.request_id)),
                    "batch_size": 1,
                    "generate_total_ms": _duration_ms(generate_start_ns, generate_end_ns),
                }
                response = replace(response, timing=timing)
                serialize_start_ns = perf_counter_ns()
                _ = json.dumps(response.to_dict(), ensure_ascii=False).encode("utf-8")
                serialize_end_ns = perf_counter_ns()
                timing["response_serialize_ms"] = _duration_ms(serialize_start_ns, serialize_end_ns)
                timing["server_total_ms"] = _duration_ms(server_start_ns, perf_counter_ns())
                self._send_json(replace(response, timing=timing).to_dict(), status=200)
            except Exception as exc:  # pragma: no cover - 服务脚本兜底返回错误
                self._send_json({"error": str(exc)}, status=500)

        def do_GET(self) -> None:  # noqa: N802 - http.server 固定方法名
            if self.path == "/health":
                self._send_json(
                    {
                        "status": "ok",
                        "linear_backend": linear_verifier.backend_name,
                        "tree_backend": tree_verifier.backend_name,
                        "model_backend": linear_verifier.metadata.get("model_backend"),
                        "model_backend_capabilities": linear_verifier.metadata.get("model_backend_capabilities"),
                    },
                    status=200,
                )
                return
            self._send_json({"error": "not found"}, status=404)

        def _read_body(self) -> str:
            content_length = int(self.headers.get("Content-Length", "0"))
            return self.rfile.read(content_length).decode("utf-8")

        def _send_json(self, payload: dict[str, Any], *, status: int) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            """保留 http.server 日志格式，但让输出更短。"""
            print(f"{self.address_string()} - {format % args}")

    return VerifyHandler


def _generate_greedy(verifier: TreeVerifier, request: GreedyGenerateRequest) -> GreedyGenerateResponse:
    """用 target model 跑 greedy baseline。"""
    if not request.prefix_ids:
        raise ValueError("Greedy generation requires a non-empty prefix.")
    eos_token_ids = set(request.eos_token_ids)
    working_prefix = list(request.prefix_ids)
    generated: list[int] = []
    forward_events: list[dict[str, Any]] = []
    stop_reason = None
    for index in range(max(0, int(request.max_new_tokens))):
        start_ns = perf_counter_ns()
        token_id = int(verifier.model.greedy_next_token(working_prefix))
        end_ns = perf_counter_ns()
        generated.append(token_id)
        working_prefix.append(token_id)
        forward_events.append(
            {
                "index": index,
                "kind": "target_only",
                "prefix_len": len(working_prefix) - 1,
                "token_id": token_id,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": _duration_ms(start_ns, end_ns),
            }
        )
        if token_id in eos_token_ids:
            stop_reason = "eos"
            break
    if stop_reason is None:
        stop_reason = "length"
    return GreedyGenerateResponse(
        request_id=request.request_id,
        generated_tokens=generated,
        stop_reason=stop_reason,
        timing={
            "target_forward_events": forward_events,
            "target_forward_total_ms": sum(float(event["duration_ms"]) for event in forward_events),
        },
    )


def _with_batched_root_guard(verifier: TreeVerifier, items: list[BatchVerifyItem]) -> list[BatchVerifyItem]:
    """为强 root guard 请求批量预取 safe root token。"""
    guard_entries: list[tuple[int, TreeVerifyRequest]] = []
    for index, item in enumerate(items):
        if item.kind == "tree" and isinstance(item.request, TreeVerifyRequest):
            if bool(item.request.metadata.get("force_root_guard", False)):
                guard_entries.append((index, item.request))
    if not guard_entries:
        return items

    prefixes = [request.prefix_ids for _index, request in guard_entries]
    start_ns = perf_counter_ns()
    token_ids = [int(token_id) for token_id in verifier.model.greedy_next_tokens(prefixes)]
    end_ns = perf_counter_ns()
    duration_ms = _duration_ms(start_ns, end_ns)
    per_item_ms = duration_ms / max(1, len(guard_entries))
    updated = list(items)
    for (index, request), token_id in zip(guard_entries, token_ids):
        metadata = dict(request.metadata)
        metadata["precomputed_root_guard_event"] = {
            "kind": "tree_root_guard",
            "prefix_len": len(request.prefix_ids),
            "token_id": int(token_id),
            "start_ns": start_ns,
            "end_ns": end_ns,
            "duration_ms": per_item_ms,
            "batch_size": len(guard_entries),
            "tree_forward_batch_kind": "root_guard_batch",
        }
        updated[index] = BatchVerifyItem(
            kind="tree",
            request=replace(request, metadata=metadata),
        )
    return updated


def _linear_forward_batch_kind(responses: list[LinearVerifyResponse]) -> str:
    kinds: set[str] = set()
    for response in responses:
        for kind in response.timing.get("linear_forward_batch_kinds", []):
            kinds.add(str(kind))
    return ",".join(sorted(kinds)) if kinds else "unknown"


def _duration_ms(start_ns: int, end_ns: int) -> float:
    """把 perf_counter_ns 差值转换为毫秒。"""
    return (int(end_ns) - int(start_ns)) / 1_000_000


def main() -> None:
    """加载 target 模型并启动 HTTP verifier。"""
    args = build_parser().parse_args()
    runner = load_causal_lm_runner(
        args.model_path,
        runner_id="a100-qwen3-14b",
        backend=args.backend,
        device=args.device,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        allow_fallback=not args.no_backend_fallback,
    )
    capabilities = runner.backend_capabilities().to_dict()
    linear_verifier = LinearVerifier(
        model=runner,
        backend_name="a100_http_linear",
        metadata={
            "model_path": args.model_path,
            "model_backend": capabilities.get("backend_name"),
            "model_backend_capabilities": capabilities,
        },
    )
    tree_verifier = TreeVerifier(
        model=runner,
        backend_name="a100_http_tree",
        metadata={
            "model_path": args.model_path,
            "model_backend": capabilities.get("backend_name"),
            "model_backend_capabilities": capabilities,
        },
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(linear_verifier, tree_verifier))
    print(
        "Serving /verify_linear, /verify_linear_batch, /verify_tree, "
        f"/verify_tree_batch and /generate_greedy on {args.host}:{args.port} "
        f"with {args.model_path} backend={capabilities.get('backend_name')}"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
