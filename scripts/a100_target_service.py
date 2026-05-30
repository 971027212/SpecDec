from __future__ import annotations

"""A100 侧最小 target verifier HTTP 服务。

默认加载 /data/chajiahao/hf_models/Qwen3-14B，提供 POST /verify_linear。
服务只做 greedy target verification，不做 batch/tree，也不决定 acceptance。
"""

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from specplatform.model import TransformersCausalLMRunner
from specplatform.verification import LinearVerifier, LinearVerifyRequest


def build_parser() -> argparse.ArgumentParser:
    """解析服务启动参数。"""
    parser = argparse.ArgumentParser(description="Run minimal A100 linear verifier service.")
    parser.add_argument("--model-path", default="/data/chajiahao/hf_models/Qwen3-14B")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--device-map", default=None)
    return parser


def make_handler(verifier: LinearVerifier) -> type[BaseHTTPRequestHandler]:
    """把 verifier 注入 HTTP handler；handler 自身不保存算法状态。"""

    class VerifyHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server 固定方法名
            if self.path != "/verify_linear":
                self._send_json({"error": "not found"}, status=404)
                return
            try:
                payload = self._read_json()
                verify_request = LinearVerifyRequest.from_dict(payload)
                response = verifier.verify_request(verify_request)
                self._send_json(response.to_dict(), status=200)
            except Exception as exc:  # pragma: no cover - 服务脚本兜底返回错误
                self._send_json({"error": str(exc)}, status=500)

        def do_GET(self) -> None:  # noqa: N802 - http.server 固定方法名
            if self.path == "/health":
                self._send_json({"status": "ok", "backend": verifier.backend_name}, status=200)
                return
            self._send_json({"error": "not found"}, status=404)

        def _read_json(self) -> dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(content_length).decode("utf-8")
            return json.loads(raw)

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


def main() -> None:
    """加载 target 模型并启动 HTTP verifier。"""
    args = build_parser().parse_args()
    runner = TransformersCausalLMRunner.from_pretrained(
        args.model_path,
        runner_id="a100-qwen3-14b",
        device=args.device,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
    )
    verifier = LinearVerifier(
        model=runner,
        backend_name="a100_http_linear",
        metadata={"model_path": args.model_path},
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(verifier))
    print(f"Serving /verify_linear on {args.host}:{args.port} with {args.model_path}")
    server.serve_forever()


if __name__ == "__main__":
    main()
