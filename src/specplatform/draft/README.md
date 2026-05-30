# draft

Draft modules expose fake draft runner adapters for the Phase 1 unified runtime.

Allowed:
- fake runner adapters used by tests and prototypes.
- deterministic encode/decode helpers for the fake runtime.

Forbidden:
- method acceptance logic.
- verifier calls.
- batch scheduling decisions.
- real backend integration.
