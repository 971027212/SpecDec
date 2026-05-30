# draft

Draft modules expose draft runner adapters for the minimal speculative decoding
runtime.

Current minimal implementation:
- `GreedyDraftRunner.generate_tokens(prefix_ids, max_tokens)` repeatedly calls a
  `CausalLMRunner` and returns raw `DraftGeneration` tokens.

Allowed:
- generate draft tokens from the current prefix and a draft budget.
- keep draft-model execution details behind a runner interface.

Forbidden:
- method acceptance logic.
- verifier calls.
- batch scheduling decisions.
