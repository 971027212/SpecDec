# verification

Verification exposes method-facing verifier APIs for the Phase 1 unified runtime.

Allowed:
- verify one `CandidateProposal`.
- verify a batch of proposals.
- fake verifier behavior for tests and prototypes.

Forbidden:
- method acceptance decisions.
- request/session execution loops.
- metrics attribution policy.
- real HTTP, Torch, GraphEngine, or A100 backend integration.
