# SpecPlatform Minimal Speculative Decoding Skeleton

This directory is the active skeleton for rebuilding a minimal real
speculative decoding loop. The active tree keeps only shared data models,
runtime orchestration boundaries, schedulers, timing, metrics, and tests needed
to add the real draft/target flow one module at a time.

```text
src/specplatform/
  core/
  draft/
  methods/
  metrics/
  model/
  runtime/
  schedulers/
  timing/
  verification/
tests/
```

Legacy methods, old runners, old experiment outputs, local SSH traces, and sync
snapshots live outside this skeleton:

```text
D:\specDec_archives\legacy_20260530
/home/chajiahao/data/specDec_archives/legacy_20260530
/data/chajiahao/specDec_archives/legacy_20260530
```

## Current Capabilities

- `CandidateProposal`
- `CausalLMRunner`
- `VerificationResult`
- `AcceptResult`
- `ExecutablePlan`
- `RuntimeContext`
- `TargetPlacementConfig`
- `RuntimeEngine`
- `RoundRobinRequestScheduler`
- `TimingSpan`
- `TimingRecorder`
- `TimingAttributor`
- Phase 1 metrics artifact writers

## Target Minimal Flow

```text
3090 draft model (Qwen3-1.7B)
  -> linear draft tokens
  -> HTTP request to A100 target verifier (Qwen3-14B)
  -> token-by-token verification
  -> prefix acceptance policy
  -> GenerationSession.append_tokens(...)
```

The fake runners and baseline loop have been removed from active code. Real
model, draft, verifier, acceptance, HTTP service, and HTTP client components are
added in later small steps.

## Target Placement

Target/verifier placement is a configuration boundary, not a runtime branch.
The default target placement is `a100`, and `3090` remains a supported placement
for runs where the target model must live on the RTX 3090 server.

`RuntimeContext.backend_info` may carry:

```text
target_placement = a100 | 3090
target_backend
target_host
target_device
```

The unified runtime still sees only a `VerifierBackend`; HTTP, Torch, and
A100/3090 service implementations are added behind that boundary.

## Verify

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_timing_phase1 tests.test_metrics_schema tests.test_cleanup_step0 -v
```

See `PROJECT_STRUCTURE.md` for the active tree and package boundaries.
