from __future__ import annotations

from specplatform.core import VerifyBatch


def attach_proposals_to_batches(
    batches: list[VerifyBatch],
    proposal_ids_by_request: dict[str, str],
) -> list[VerifyBatch]:
    for batch in batches:
        batch.proposal_ids = [
            proposal_ids_by_request[request_id]
            for request_id in batch.request_ids
            if request_id in proposal_ids_by_request
        ]
    return batches
