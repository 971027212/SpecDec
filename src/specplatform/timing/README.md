# timing

Timing owns measured spans and timing summary views.

Allowed:
- Real measured `TimingSpan` objects.
- Conversion from real spans to system-level `PhaseEvent` records.
- Request-level attribution events derived from shared spans.
- Summary views with explicit `event_scope` and `span_kind` filters.

Forbidden:
- Attribution `TimingSpan` objects.
- Method-owned timing writes.
- Mixing system leaf, system aggregate, and request attribution in one summary row.

Phase 1 uses `parent_span_id` only for attribution provenance. Aggregate and leaf nesting is represented by shared `run_id`, `round_id`, and `plan_id`.
