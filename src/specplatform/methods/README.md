# methods

Methods describe algorithm differences only.

Current minimal implementation:
- `LinearCandidateStrategy` wraps `DraftGeneration.tokens` into a linear `CandidateProposal`.
- `GreedyPrefixAcceptancePolicy` consumes `VerificationResult` and emits `AcceptResult`.

Allowed:
- `CandidateStrategy`: create a `CandidateProposal` from a session, draft runner, and budget.
- `AcceptancePolicy`: turn a `VerificationResult` into an `AcceptResult`.
- `PlanningPolicy`: return planning hints for schedulers.

Forbidden:
- complete request loops.
- direct verifier calls from `AcceptancePolicy`.
- batch verify orchestration inside strategies.
