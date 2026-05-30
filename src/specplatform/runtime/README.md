# runtime

Runtime owns session execution.

Allowed:
- execute an `ExecutablePlan`.
- call candidate strategies, verifier backends, acceptance policies, and metrics recorders in order.
- apply `AcceptResult` to sessions.

Forbidden:
- method-name-specific branches.
- scheduler decisions.
- method-owned event recording.
