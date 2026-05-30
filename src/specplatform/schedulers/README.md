# schedulers

Schedulers turn active sessions, resources, and planning hints into an `ExecutablePlan`.

Allowed:
- assign requests to draft workers.
- assign draft budgets.
- build verify batches.

Forbidden:
- calling draft runners.
- calling verifier backends.
- applying accepted tokens to sessions.
