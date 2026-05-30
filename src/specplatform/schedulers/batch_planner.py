from __future__ import annotations

"""verify batch 的轻量后处理。

scheduler 先按 request_id 计划 batch；runtime 在 proposal 生成后用这里的函数
把 request_id 映射成 proposal_id。
"""

from specplatform.core import VerifyBatch


def attach_proposals_to_batches(
    batches: list[VerifyBatch],
    proposal_ids_by_request: dict[str, str],
) -> list[VerifyBatch]:
    """把已生成的 proposal_id 回填到 verify batches。"""
    for batch in batches:
        batch.proposal_ids = [
            proposal_ids_by_request[request_id]
            for request_id in batch.request_ids
            if request_id in proposal_ids_by_request
        ]
    return batches
