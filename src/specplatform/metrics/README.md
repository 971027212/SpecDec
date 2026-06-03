# metrics

Metrics records events and writes artifacts.

Allowed:
- `PhaseEvent` recording.
- system-level and request-attributed summaries.
- CSV/JSON artifacts.
- timing chart rendering from existing artifacts.

Forbidden:
- method-specific generation logic.
- verifier calls.
- scheduler decisions.
