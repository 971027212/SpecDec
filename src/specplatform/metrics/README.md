# metrics

Metrics records events and writes artifacts.

Allowed:
- `PhaseEvent` recording.
- system-level and request-attributed summaries.
- CSV/JSON artifacts.

Forbidden:
- method-specific generation logic.
- verifier calls.
- scheduler decisions.
