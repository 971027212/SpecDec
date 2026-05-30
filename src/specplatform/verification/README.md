# verification

Verification exposes method-facing verifier APIs for the Phase 1 unified runtime.

Current minimal implementation:
- `LinearVerifier` performs local greedy target verification.
- `LinearVerifyRequest` / `LinearVerifyResponse` define `POST /verify_linear`.
- `HttpLinearVerifierClient` lets the 3090 runtime call the A100 verifier service.

Allowed:
- verify one `CandidateProposal`.
- verify a batch of proposals.
- return `VerificationResult` facts for acceptance policies.

Forbidden:
- method acceptance decisions.
- request/session execution loops.
- metrics attribution policy.
