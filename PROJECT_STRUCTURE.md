# SpecPlatform Project Structure

Current directory: clean minimal speculative decoding skeleton.

Local active path:

```text
D:\specDec
```

Server active paths:

```text
/home/chajiahao/data/specDec
/data/chajiahao/specDec
```

Repository-external legacy archives:

```text
D:\specDec_archives\legacy_20260530
/home/chajiahao/data/specDec_archives/legacy_20260530
/data/chajiahao/specDec_archives/legacy_20260530
```

## Active Tree

```text
D:\specDec
  src/
    specplatform/
      core/
      draft/
      methods/
      metrics/
      model/
      runtime/
      schedulers/
      timing/
      verification/
      config.py
  tests/
    test_timing_phase1.py
    test_metrics_schema.py
    test_cleanup_step0.py
  README.md
  PROJECT_STRUCTURE.md
```

No active code should live under `archive/`, `result/`, old runner folders, or
old method-specific experiment folders.

## Main Flow

```text
GenerationSession
  -> Scheduler.plan(...)
  -> ExecutablePlan(draft_jobs, verify_batches)
  -> RuntimeEngine executes draft jobs
  -> CandidateStrategy.propose(...)
  -> CandidateProposal
  -> VerifierBackend.verify_batch(...)
  -> VerificationResult
  -> AcceptancePolicy.accept(...)
  -> AcceptResult
  -> GenerationSession.append_tokens(...)
  -> TimingSpan real measurements
  -> PhaseEvent metrics
  -> CSV/JSON artifacts
```

## Package Boundaries

### core/

Shared data models. This package does not run request loops, call verifier
backends, or record metrics side effects.

| File | Purpose |
|---|---|
| `candidate.py` | `CandidateProposal`, with `linear` and `tree` shapes. |
| `result.py` | `VerificationResult` and `AcceptResult`. |
| `plan.py` | `DraftBudget`, `DraftJob`, `VerifyBatch`, `PlanHints`, `ExecutablePlan`. |
| `context.py` | Read-only `RuntimeContext`; no engine/verifier/metrics escape hatch. |
| `target.py` | Target placement config: default `a100`, supported `3090`. |
| `types.py` | `CandidateNode`, `CandidateTree`, `PhaseEvent`. |

### runtime/

Unified runtime execution layer. It executes an `ExecutablePlan` and must not
branch on method names.

### methods/

Algorithm difference layer. It owns strategies, policies, and planning hints;
it must not contain a full request loop.

### schedulers/

Request-to-worker, draft budget, and verify-batch planning.

### draft/

Draft runner boundary. Draft code generates draft tokens only; it does not
accept/reject tokens, call verifiers, or decide batches. `GreedyDraftRunner`
now runs a `CausalLMRunner` greedily and returns raw `DraftGeneration` tokens.

### verification/

Unified verifier API. HTTP, Torch, A100, or 3090 target-service integrations
belong behind `VerifierBackend`.

### metrics/

Event recording and artifact writing.

### timing/

Real measurement spans and summary views. Request attribution is derived from
shared spans and is not a real measured span.

### model/

Model runner abstraction. `CausalLMRunner` defines the minimal real causal LM
interface; Transformers implementations are added in later steps.

## Target Placement Rule

The target/verifier model defaults to the A100 server, but the platform must
also support placing target/verifier on the 3090 server. This is represented as
configuration, not as method-specific runtime logic:

```text
target_placement = a100 | 3090
target_backend
target_host
target_device
```

`RuntimeEngine` continues to depend on `VerifierBackend` only.

## Tests

```text
tests/test_unified_runtime_phase1.py
tests/test_timing_phase1.py
tests/test_metrics_schema.py
tests/test_cleanup_step0.py
```

Coverage includes:

- shared batch verify is recorded once at system level.
- request-level attribution defaults to average split.
- `RuntimeContext` does not expose execution escape hatches.
- `runtime/engine.py` has no method-name-specific branches.
- Phase 1 artifact writers create required files.
- timing spans, request attribution, and summaries do not double-count time.
- `PhaseEvent`, `CandidateTree`, and target placement schema basics.
- active fake/baseline modules have been removed.

## Verify

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_timing_phase1 tests.test_metrics_schema tests.test_cleanup_step0 -v
```
