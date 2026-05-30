# core

Shared runtime data models live here.

Allowed:
- candidate, result, executable plan, and read-only runtime context types.
- small validation rules for those data models.

Forbidden:
- request execution loops.
- verifier calls.
- metrics recording side effects.
