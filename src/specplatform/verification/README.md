# verification

Verification exposes method-facing verifier APIs for the Phase 1 unified runtime.

Allowed:
- verify one `CandidateProposal`.
- verify a batch of proposals.
- return `VerificationResult` facts for acceptance policies.

Forbidden:
- method acceptance decisions.
- request/session execution loops.
- metrics attribution policy.
