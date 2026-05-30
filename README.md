# SpecPlatform Unified Runtime Skeleton

This directory is the active Phase 1 skeleton for the unified speculative
decoding runtime. It keeps only the minimal runtime loop, fake runners, timing,
metrics, and tests needed to evolve the new platform.

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
- `FakeDraftRunner`
- `FakeProposalVerifier`
- `FakeLinearCandidateStrategy`
- `LinearPrefixAcceptancePolicy`
- Phase 1 metrics artifact writers

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

The unified runtime still sees only a `VerifierBackend`; real HTTP, Torch, or
A100/3090 service implementations are not active skeleton code yet.

## Verify

PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m unittest tests.test_timing_phase1 tests.test_unified_runtime_phase1 tests.test_metrics_schema -v
```

See `PROJECT_STRUCTURE.md` for the active tree and package boundaries.
