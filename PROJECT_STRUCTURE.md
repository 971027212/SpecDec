# SpecPlatform Project Structure

Current directory: clean minimal speculative decoding skeleton.

Local active path:

```text
/home/chajiahao/data/specDec
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
/home/chajiahao/data/specDec
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
    test_greedy_draft_runner.py
    test_model_interface.py
    test_minimal_speculative_loop.py
    test_timing_charts.py
    test_specedge_tree_core.py
    test_specedge_smoke_runner.py
    test_worker_registry_scheduler.py
    test_experiment_matrix_runner.py
  scripts/
    a100_target_service.py
    3090_specedge_smoke.py
    3090_speculative_smoke.py
    render_timing_charts.py
    run_experiment_matrix.py
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
branch on method names. The current loop is:
`scheduler -> draft -> candidate -> verifier -> acceptance -> append`.

### methods/

Algorithm difference layer. It owns strategies, policies, and planning hints;
it must not contain a full request loop. Current minimal method components are
`LinearCandidateStrategy`, `GreedyPrefixAcceptancePolicy`, SpecEdge tree
policies, `DiPSDPlanningPolicy`, and `SLEDPlanningPolicy`.

### schedulers/

Request-to-worker, draft budget, and verify-batch planning. `RequestPool`,
`BatchAssignmentPolicy`, and `DraftLengthPolicy` are shared by SpecEdge,
DiP-SD, and SLED. Matrix experiments can now choose either legacy shared-model
workers or explicit registry-backed workers with homogeneous/heterogeneous
speed profiles. SLED uses `worker_preferences` for one stable edge-device draft
worker per request; `PlanHints.candidate_worker_preferences` remains a generic
runtime feature but is not part of the SLED paper reproduction.

### draft/

Draft runner boundary. Draft code generates draft tokens only; it does not
accept/reject tokens, call verifiers, or decide batches. `GreedyDraftRunner`
now runs a `CausalLMRunner` greedily and returns raw `DraftGeneration` tokens.
`DraftWorkerRegistry` owns configurable multi-draft loading and constructs
typed runners from per-worker `model_path/device/backend/draft_type` settings.
Speed profile metadata includes relative speed, latency, quality, and expected
acceptance hints for method planning.

### verification/

Unified verifier API. HTTP, Torch, A100, or 3090 target-service integrations
belong behind `VerifierBackend`. Current minimal implementations are
`LinearVerifier`, `LinearVerifyRequest/Response`, and `HttpLinearVerifierClient`.

### metrics/

Event recording, artifact writing, matplotlib timing charts, and matrix-level
comparison plots.  Matrix summaries include a long per-method table, a wide
per-cell comparison table, aggregate method stats, best-method tables, and
speedup heatmaps for SpecEdge/DiP-SD/SLED.

### timing/

Real measurement spans and summary views. Request attribution is derived from
shared spans and is not a real measured span.

### model/

Model runner abstraction. `CausalLMRunner` defines the minimal real causal LM
interface; `TransformersCausalLMRunner` adapts local Hugging Face causal LM
weights without owning a generation loop.

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
tests/test_timing_phase1.py
tests/test_metrics_schema.py
tests/test_cleanup_step0.py
tests/test_greedy_draft_runner.py
tests/test_model_interface.py
tests/test_minimal_speculative_loop.py
tests/test_timing_charts.py
tests/test_specedge_tree_core.py
tests/test_specedge_smoke_runner.py
tests/test_worker_registry_scheduler.py
tests/test_experiment_matrix_runner.py
```

Coverage includes:

- shared batch verify is recorded once at system level.
- request-level attribution defaults to average split.
- `RuntimeContext` does not expose execution escape hatches.
- `runtime/engine.py` has no method-name-specific branches.
- Phase 1 artifact writers create required files.
- timing spans, request attribution, and summaries do not double-count time.
- timing chart audit and optional PNG/SVG rendering.
- `PhaseEvent`, `CandidateTree`, and target placement schema basics.
- active fake/baseline modules have been removed.
- draft proposal is verified.
- accepted/bonus tokens are written back to `GenerationSession`.
- runtime contains no method-name branch.

## Verify

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
