from __future__ import annotations

"""3090 侧 HTTP verifier client。

runtime 仍然只依赖 VerifierBackend；这个 client 把 verify_proposal 转成
POST /verify_linear 请求，并把 A100 响应还原成 VerificationResult。
"""

import json
from dataclasses import dataclass
from typing import Any
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError

from specplatform.core import CandidateProposal, RuntimeContext, VerificationResult
from specplatform.verification.base import VerifierBackend
from specplatform.verification.linear import _eos_token_ids, _proposal_prefix_ids, _validate_response_for_request
from specplatform.verification.schema import LinearVerifyRequest, LinearVerifyResponse


@dataclass
class HttpLinearVerifierClient(VerifierBackend):
    """调用远端 /verify_linear 的最小 HTTP verifier backend。"""

    base_url: str
    endpoint: str = "/verify_linear"
    timeout_s: float = 60.0
    backend_name: str = "linear_http"

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
        response = self._post_json(verify_request)
        _validate_response_for_request(response, verify_request)
        return VerificationResult(
            request_id=proposal.request_id,
            proposal_id=proposal.proposal_id,
            shape=proposal.shape,
            accepted_prefix_len=response.accepted_prefix_len,
            verified_tokens=list(response.verified_tokens),
            bonus_token=response.bonus_token,
            payload=response.to_dict(),
            metadata={"backend_name": self.backend_name, **dict(response.metadata)},
        )

    def _post_json(self, verify_request: LinearVerifyRequest) -> LinearVerifyResponse:
        """用标准库发 JSON 请求，减少服务端/客户端额外依赖。"""
        url = f"{self.base_url.rstrip('/')}/{self.endpoint.lstrip('/')}"
        body = json.dumps(verify_request.to_dict()).encode("utf-8")
        http_request = urllib_request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(http_request, timeout=self.timeout_s) as response:
                payload: dict[str, Any] = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP verifier returned {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Failed to reach HTTP verifier at {url}: {exc}") from exc
        return LinearVerifyResponse.from_dict(payload)
