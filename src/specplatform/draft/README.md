# draft

Draft modules expose draft runner adapters for the minimal speculative decoding
runtime.

Allowed:
- generate draft tokens from the current prefix and a draft budget.
- keep draft-model execution details behind a runner interface.

Forbidden:
- method acceptance logic.
- verifier calls.
- batch scheduling decisions.
