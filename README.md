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
- `TransformersCausalLMRunner`
- `GreedyDraftRunner`
- `LinearCandidateStrategy`
- `LinearVerifier`
- `HttpLinearVerifierClient`
- `GreedyPrefixAcceptancePolicy`
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
model interface, draft runner, linear candidate strategy, verifier contract,
acceptance policy, unified runtime loop, A100 HTTP service, and 3090 HTTP client
are now implemented for the minimal single-request linear path.

## Smoke Commands

A100 target service:

```bash
cd /data/chajiahao/specDec
PYTHONPATH=src /data/chajiahao/miniconda3/envs/specedge/bin/python \
  scripts/a100_target_service.py \
  --model-path /data/chajiahao/hf_models/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8010
```

3090 draft-side smoke:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src /home/chajiahao/miniconda3/bin/python \
  scripts/3090_speculative_smoke.py \
  --draft-model-path /home/chajiahao/data/hf_models/Qwen3-1.7B \
  --target-url http://a100-specdec:8010 \
  --max-new-tokens 16 \
  --draft-tokens 4
```

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
python -m unittest discover -s tests -v
```

See `PROJECT_STRUCTURE.md` for the active tree and package boundaries.
