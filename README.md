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
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/3090_speculative_smoke.py \
  --draft-model-path /home/chajiahao/data/hf_models/Qwen3-1.7B \
  --target-url http://172.16.11.62:8010 \
  --max-new-tokens 16 \
  --draft-tokens 4 \
  --run-id 3090_a100_smoke \
  --output-dir experiments/3090_a100_smoke/latest \
  --plot-formats png,svg
```

SpecEdge V1 three-baseline smoke:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/3090_specedge_smoke.py \
  --config configs/specedge_v1_smoke.yaml \
  --run-id 3090_a100_specedge_smoke \
  --output-dir experiments/3090_a100_specedge_smoke/latest \
  --plot-formats png,svg
```

Use the built-in 8-prompt mixed sample set:

```bash
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/3090_specedge_smoke.py \
  --config configs/specedge_v1_smoke.yaml \
  --use-sample-prompts \
  --run-id 3090_a100_specedge_8prompt \
  --output-dir experiments/3090_a100_specedge_8prompt/latest
```

This runner writes `target_only/`, `linear/`, and `tree/` subdirectories plus
`combined_summary.json`. The first correctness gate is:

```text
combined_summary.matches_target_only.linear == true
combined_summary.matches_target_only.tree == true
```

On the 3090 server, `a100-specdec` is useful as an SSH alias, but HTTP smoke
traffic should use the A100 data-plane address `172.16.11.62`.

SpecEdge/DiP-SD/SLED shared-method smoke:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/3090_specedge_smoke.py \
  --config configs/specedge_dipsd_sled_multidraft_smoke.yaml
```

`draft.worker_count` keeps the legacy shared-model compatibility path.  For
true independent draft workers, use `draft.workers` entries with per-worker
`model_path`, `device`, `backend`, `draft_type`, and `speed_profile`.  The
runtime still receives only a `{worker_id: runner}` map; the registry owns model
loading and runner construction.  Matrix runs can generate those explicit
workers with `--draft-worker-mode explicit`; add
`--worker-speed-profile heterogeneous` to give DiP-SD/SLED distinct speed,
latency, and quality metadata.

Draft-side warmup is recorded as setup, not inference-loop time.  The smoke
runner writes `setup.load_draft_model`, `setup.warm_draft_workers`, and
`setup.warm_draft_worker` events before any method starts; `runtime.round_total`
and matrix speedups continue to use only method runtime.  Configure warmup under
`draft.warmup` (`enabled`, `tokens`, `tree_depth`, `branch_width`,
`max_budget`, `parallelism`).  For a real 8-card 3090 probe, use
`configs/specedge_dipsd_sled_8gpu_smoke.yaml`, which maps workers across
`cuda:0..7` and warms all 8 workers in parallel.

Systematic matrix runner:

```bash
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/run_experiment_matrix.py \
  --base-config configs/specedge_dipsd_sled_multidraft_smoke.yaml \
  --output-dir experiments/method_matrix/latest \
  --request-counts 1,2,4,8,16 \
  --draft-worker-counts 1,2,4 \
  --depths 1,2,4,8 \
  --network-profiles observe,low_uplink,high_rtt \
  --resume \
  --rerun-mismatches \
  --continue-on-error
```

When `request_count` is larger than the prompt file, the smoke runner cycles the
sample prompts and gives repeated requests unique ids such as
`mixed-001-repeat1`. This keeps the matrix dimension honest for
`request_count=16` even when the prompt fixture has fewer distinct rows.

Explicit-worker matrix variant:

```bash
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/run_experiment_matrix.py \
  --base-config configs/specedge_dipsd_sled_multidraft_smoke.yaml \
  --output-dir experiments/method_matrix_explicit/latest \
  --request-counts 1,2,4,8,16 \
  --draft-worker-counts 1,2,4 \
  --depths 1,2,4,8 \
  --network-profiles observe,low_uplink,high_rtt \
  --draft-worker-mode explicit \
  --worker-speed-profile heterogeneous \
  --resume \
  --rerun-mismatches \
  --continue-on-error
```

Original-method 8-GPU reproduction matrix:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-specdec \
  /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/run_experiment_matrix.py \
  --base-config configs/specedge_dipsd_sled_8gpu_smoke.yaml \
  --output-dir experiments/original_method_matrix_8gpu_small/latest \
  --request-counts 4,8 \
  --draft-worker-counts 4,8 \
  --depths 8 \
  --network-profiles observe \
  --draft-worker-mode explicit \
  --worker-speed-profile model_size \
  --draft-worker-model-paths /home/chajiahao/data/hf_models/Qwen3-0.6B,/home/chajiahao/data/hf_models/Qwen3-1.7B \
  --draft-worker-devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --draft-worker-backends qwen3_graph \
  --draft-worker-torch-dtypes fp16 \
  --draft-worker-draft-types both \
  --methods target_only,tree_stop_wait,specedge_pipeline,dip_sd,sled \
  --plot-formats png,svg \
  --no-plots \
  --resume \
  --rerun-mismatches \
  --continue-on-error
```

This preset keeps DiP-SD and SLED on the stop-wait reproduction path and uses
the async pipeline only for SpecEdge.  It disables per-method plots for speed,
but still writes matrix-level charts including `matrix_phase_distribution`.

Original-method 8-GPU depth sweep:

```bash
PYTHONPATH=src MPLCONFIGDIR=/tmp/matplotlib-specdec \
  /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/run_experiment_matrix.py \
  --base-config configs/specedge_dipsd_sled_8gpu_smoke.yaml \
  --output-dir experiments/original_method_matrix_8gpu_depth_sweep/latest \
  --request-counts 8 \
  --draft-worker-counts 8 \
  --depths 2,4,8 \
  --network-profiles observe \
  --draft-worker-mode explicit \
  --worker-speed-profile model_size \
  --draft-worker-model-paths /home/chajiahao/data/hf_models/Qwen3-0.6B,/home/chajiahao/data/hf_models/Qwen3-1.7B \
  --draft-worker-devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --draft-worker-backends qwen3_graph \
  --draft-worker-torch-dtypes fp16 \
  --draft-worker-draft-types both \
  --methods target_only,tree_stop_wait,specedge_pipeline,dip_sd,sled \
  --plot-formats png,svg \
  --no-plots \
  --resume \
  --rerun-mismatches \
  --continue-on-error
```

The depth sweep report includes a `Depth Effect` table and
`plots/matrix_runtime_by_depth.*`, which are more informative than the default
request-count plot when only `depth` varies.

`sled.confidence_threshold` controls SLED dynamic drafting.  Each request is
owned by one edge-device draft worker; that worker keeps drafting until the
greedy token confidence drops below the threshold or the depth cap is reached.
The server then verifies many requests together with batched linear
verification.  Same-request multi-worker candidate selection is not part of the
SLED reproduction.

It writes:

```text
matrix_summary.csv
matrix_summary.json
matrix_comparison.csv
matrix_comparison.json
matrix_method_aggregate.csv
matrix_method_aggregate.json
matrix_phase_distribution.csv
matrix_phase_distribution.json
matrix_status.json
matrix_best_methods.csv
matrix_best_methods.json
plots/matrix_runtime_by_method.png
plots/matrix_runtime_by_depth.png
plots/matrix_speedup_vs_tree_stop_wait.png
plots/matrix_speedup_heatmap_specedge_pipeline.png
plots/matrix_speedup_heatmap_dip_sd.png
plots/matrix_speedup_heatmap_sled.png
plots/matrix_server_idle_gap.png
plots/matrix_method_aggregate.png
plots/matrix_phase_distribution.png
plots/matrix_best_method_counts.png
```

`matrix_summary.*` is the long table: one row per method per cell.
`matrix_comparison.*` is the wide table: one row per matrix cell, with
SpecEdge/DiP-SD/SLED speedups against `tree_stop_wait` and `target_only`.
`matrix_method_aggregate.*` reports mean speedup, win count, target-match
counts, average verify batch size, idle gap, and target forward counts by
method.
`matrix_phase_distribution.*` reads each method's `phase_summary.csv` and
aggregates only `system_leaf_summary` rows, so it compares the real main-phase
busy time without double-counting detail or attribution rows.  Its `busy/wall`
ratio can exceed 1.0 when draft workers run in parallel or when draft and
verify overlap in a pipelined method.
The matrix tables also carry flattened `method_reproduction` fields such as
`reproduction_execution_mode`, missing original-method mechanism counts, and
future-optimization counts, so batch reports do not mix reproduction evidence
with optimization variants.
`matrix_status.json` reports completed, failed, and incomplete matrix cells.
Each cell's smoke stdout/stderr is written to `logs/<run_id>.log` by default;
use `--stream-cell-output` when debugging a single cell interactively.

Each smoke run, including per-method timing charts when enabled, stays under
`experiments/method_matrix/latest/runs/<run_id>/`.

The smoke writes framework timing/metrics artifacts under `--output-dir`:

```text
phase_events.jsonl
phase_events.csv
phase_summary.csv
request_results.json
smoke_output.json
plots/timeline_gantt.png
plots/timeline_gantt.svg
plots/compact_timeline_distribution.png
plots/compact_timeline_distribution.svg
plots/phase_breakdown.png
plots/round_waterfall.png
plots/overlap_concurrency.png
plots/worker_batch_lanes.png
plots/network_breakdown.png
plots/proactive_reuse_chart.png
plots/http_verify_breakdown.png
plots/timing_audit.json
plots/timing_audit.txt
tree_snapshots.jsonl
```

`smoke_output.json` and `combined_summary.json` include setup metrics:
`setup_load_draft_model_ms`, `setup_warm_draft_workers_ms`,
`setup_warm_draft_worker_event_count`, and `setup_total_ms`.  These values are
for diagnosing cold start and model placement; they are deliberately separate
from default inference-time comparisons.

`combined_summary.json` also includes `method_reproduction`.  This is the
method-fidelity checklist for separating paper reproduction from later platform
optimizations.  In the current runner, `specedge_pipeline` is allowed to use the
async pipeline because proactive draft/verify overlap is part of the SpecEdge
core.  `dip_sd` and `sled` intentionally stay on the stop-wait shared runtime
unless a separate optimization experiment enables otherwise; their report rows
will mark inactive pipeline/proactive behavior as partial or future work instead
of counting it as original-method speedup.

Use `--sync-dest user@host:/path/` to rsync the whole output directory to a
desktop machine. Existing artifacts can be re-rendered without rerunning models:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/render_timing_charts.py \
  --input-dir experiments/3090_a100_smoke/latest \
  --formats png,svg
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
