# specDec 3090/A100 Handoff

Generated: 2026-06-02 UTC

## Latest A100 Connectivity Update, 2026-06-03 UTC

The 3090 host can SSH to the A100 machine via the configured alias:

```bash
ssh a100-specdec
```

SSH alias details from `~/.ssh/config`:

```text
Host a100-specdec
  HostName 172.16.11.62
  User chajiahao
  IdentityFile ~/.ssh/specdec_a100_ed25519
  IdentitiesOnly yes
```

For experiment HTTP traffic, keep using the A100 data-plane IP:

```text
http://172.16.11.62:8011
```

Do not use `http://a100-specdec:<port>` for HTTP verifier traffic. On the 3090
host that name currently resolves through a `198.18.0.60` path that accepts TCP
connections but returns empty HTTP replies. Use the alias for SSH only.

Current working target verifier:

```bash
curl --noproxy '*' -sS --max-time 5 http://172.16.11.62:8011/health
```

Expected health fields:

```text
status = ok
model_backend = qwen3_graph
backend_fallback = false
supports_linear_verify_batch = true
supports_tree_forward_batch = true
supports_cuda_graph = true
```

Current service was started on A100 GPU1 because GPU0 is occupied by an unrelated
root-owned process (`python3 VBSFL-privacy-accuracy-time-revised1.py`, PID
1224777 observed on 2026-06-03) using about 50GB. Starting qwen3_graph on GPU0,
or starting it with the default long-context graph capture settings, can OOM.

Working restart command:

```bash
ssh a100-specdec
cd /data/chajiahao/specDec
PYTHONPATH=src \
SPECPLATFORM_QWEN3_GRAPH_MAX_LEN=2048 \
SPECPLATFORM_QWEN3_GRAPH_MAX_TOKENS=20 \
SPECPLATFORM_QWEN3_GRAPH_MAX_BATCH_SIZE=8 \
nohup /data/chajiahao/miniconda3/envs/specedge/bin/python \
  scripts/a100_target_service.py \
  --model-path /data/chajiahao/hf_models/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8011 \
  --device cuda:1 \
  --attn-implementation eager \
  --backend qwen3_graph \
  --no-backend-fallback \
  > /tmp/specdec_a100_service_8011.log 2>&1 &
```

If `8011/health` is refused, first check:

```bash
ssh a100-specdec 'ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "a100_target_service|8011" | grep -v grep'
ssh a100-specdec 'tail -120 /tmp/specdec_a100_service_8011.log'
ssh a100-specdec 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.free,memory.total --format=csv,noheader'
```

## Latest Handoff Update, 2026-06-02 Three-Method Tuning Stop Point

Updated: 2026-06-02 15:09:03 UTC.

This section is the current handoff state for the active user goal. It
supersedes the older "Four-Method DiP-SD Paper-MILP Reproduction" section
below for current performance conclusions. Keep the older section only as
historical context.

### User Direction At Stop

The user explicitly said:

```text
停，总结更新handoff md
```

So the in-progress DiP-SD solver-mode tuning sweep was stopped. There should be
no active `run_experiment_matrix.py`, `run_method_tuning.py`, or
`3090_specedge_smoke.py` process left from this work.

### Current Active Comparison Scope

Current fair-comparison scope:

```text
target_only baseline
specedge_pipeline
sled_async
dip_sd
```

Fairness envelope used for tuning/evaluation:

```text
same tuning prompts: data/sample_prompts_mixed.jsonl
same held-out prompts: data/sample_prompts_heldout.jsonl
same A100 target verifier: http://172.16.11.62:8011
same 3090 draft-worker pool
same correctness gate: match target_only
locked config for held-out matrix:
  experiments/three_method_tuning_verifyfix/latest/locked_three_method_compare.yaml
```

### Important Current Documents

Detailed method explanation and six inference examples:

```text
THREE_METHOD_INFERENCE_EXAMPLES.md
```

Current partial matrix diagnosis and optimization conclusion:

```text
THREE_METHOD_PARTIAL_MATRIX_DIAGNOSIS.md
```

These are the best human-facing summaries so far.

### Locked Best Tuning Before Held-Out Matrix

Source:

```text
experiments/three_method_tuning_verifyfix/latest/best_tuning.json
```

Locked SpecEdge:

```text
tree.max_depth = 4
tree.branch_width = 8
tree.max_budget = 20
specedge.official = true
pipeline.proactive_depth = 8
```

Locked SLED:

```text
sled.batch_size = 8
sled.confidence_threshold = 0.6
sled.max_speculation_tokens = 4
sled.async.proactive_tokens = 8
```

Locked DiP-SD:

```text
dip_sd.max_draft_length = 4
dip_sd.solver = paper_milp_or_dinkelbach
dip_sd.plan_cache_enabled = true
dip_sd.steady_state_enabled = true
dip_sd.calibration_enabled = false
```

### Partial Held-Out Matrix Status

Matrix directory:

```text
experiments/three_method_locked_matrix_verifyfix/latest
```

The full 240-cell matrix was intentionally stopped because the completed cells
already showed stable correctness but no overall speedup.

Completed clean cells:

```text
completed_cells: 69
correctness_clean_cells: 69
mismatch_count: 0
```

Partial speedup vs `target_only` over the 69 completed held-out cells:

| method | mean speedup | median | best | worst |
| --- | ---: | ---: | ---: | ---: |
| SpecEdge | 0.657x | 0.614x | 0.992x | 0.467x |
| SLED | 0.662x | 0.654x | 0.909x | 0.478x |
| DiP-SD | 0.465x | 0.431x | 0.701x | 0.304x |

Interpretation:

- Greater than `1.0x` means faster than target-only.
- None of the three methods is currently faster than target-only on average.
- Continuing the full 240-cell matrix is not recommended until the methods are
  optimized.

### DiP-SD Calibration On/Off Verification

Calibration experiment directory:

```text
experiments/dip_sd_calibration_tuning_verifyfix/latest
```

Calibration profile used:

```text
experiments/three_method_locked_matrix_verifyfix/latest/runs/rc2_dw2_dlocked_mt16_observe/dip_sd/dip_sd_latency_calibration.json
```

Recommended method config from that profile:

```text
dip_sd_draft_beta = 65.28746926666665
dip_sd_draft_c = 0.0
dip_sd_verify_beta = 3.4492080640198424
dip_sd_verify_c = 1.3940573138835278e-09
```

Calibration results:

| max_draft_length | calibration | match | method ms | target ms | speedup | solver/planner ms | draft-ready wait ms | avg verify batch |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | off | true | 37734.940 | 3786.050 | 0.100x | 28359.703 | 4682.649 | 3.516 |
| 4 | on | true | 34559.991 | 4351.203 | 0.126x | 29905.735 | 1002.930 | 3.780 |
| 8 | off | true | 40729.294 | 4474.060 | 0.110x | 31354.022 | 4899.786 | 3.516 |
| 8 | on | true | 35020.223 | 4406.346 | 0.126x | 30440.602 | 903.894 | 3.780 |

Conclusion:

- Calibration works and reduces draft-ready wait.
- Calibration does not solve the main problem.
- DiP-SD is still dominated by online solver/planner time near 30 seconds.

### DiP-SD Solver-Mode Sweep, Interrupted

Partial solver-mode sweep directory:

```text
experiments/dip_sd_solver_mode_tuning_verifyfix/latest
```

Command was interrupted by the user. Only these completed summaries should be
used:

| solver | max_draft_length | calibration | match | method ms | target ms | speedup | solver/planner ms | backend |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | --- |
| dinkelbach | 4 | on | true | 17911.995 | 3717.468 | 0.208x | 9904.459 | dinkelbach_coordinate |
| enumerate | 4 | on | true | 40743.745 | 3777.836 | 0.093x | 36368.843 | enumerate |

The `heuristic` candidate was running when the user said stop, so it was killed
and should not be treated as a completed result.

Interpretation:

- Switching from paper MILP to `dinkelbach` reduces DiP-SD planner/solver cost
  from about 30s to about 9.9s in this small run.
- That is a real improvement, but still only `0.208x` vs target-only.
- `enumerate` is worse than paper MILP here, with solver time about 36.4s.
- Next DiP-SD work should focus on removing or amortizing online planning, not
  merely expanding the normal hyperparameter grid.

### Code Changes Made During This Update

Changed:

```text
scripts/run_method_tuning.py
tests/test_method_tuning_runner.py
THREE_METHOD_PARTIAL_MATRIX_DIAGNOSIS.md
HANDOFF_3090.md
```

Important fix:

- `scripts/run_method_tuning.py` now writes DiP-SD calibration profiles as
  absolute paths in generated candidate configs.
- Without this, a valid relative profile path was resolved relative to the
  generated config directory and calibration candidates failed with
  `FileNotFoundError`.

Verification run after the code fix:

```text
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  -m unittest tests.test_method_tuning_runner -q

Ran 5 tests in 0.046s
OK
```

Additional verification:

```text
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  -m unittest tests.test_method_tuning_runner tests.test_experiment_matrix_runner -q

Ran 29 tests in 1.668s
OK
```

### Recommended Next Step

Do not resume the full 240-cell held-out matrix yet.

Recommended next work:

1. For DiP-SD, implement or evaluate a no-online-solver / stronger-plan-cache
   path so planner cost can be separated from algorithmic pipeline behavior.
2. For SpecEdge, do a focused sweep around the near-best cells:
   `request_count=2,4`, `max_new_tokens=16,32`,
   `branch_width=4,8`, `proactive_depth=4,8,12`.
3. For SLED, do a queue/threshold sweep:
   `confidence_threshold=0.55,0.6,0.65`,
   `queue_max_wait_ms=0,2,5`, `batch_size=4,8`.
4. Only after a small optimized matrix shows any method exceeding `1.0x` should
   the full 240-cell matrix be resumed.

## Latest Handoff Update, 2026-06-02 Four-Method DiP-SD Paper-MILP Reproduction

Updated: 2026-06-02 09:13:59 UTC.

This section supersedes every experimental conclusion below. Older three-method
and old DiP-SD result directories can remain on disk as historical evidence, but
do not use them as the current result set.

### Current Goal

The current target is a four-method comparison on the same experiment platform:

```text
target_only
specedge_pipeline
sled_async
dip_sd
```

Implementation direction from the user:

- Old DiP-SD behavior/results should not be preserved as the current method.
- DiP-SD should be implemented as a decoupled method, without breaking
  target-only, SpecEdge, or SLED.
- The platform should expose the real eight RTX 3090 draft GPUs to the methods.
  Do not require every method to consume all eight cards every round; the
  important requirement is that the workers are real, explicit, audited, and
  available.
- The target verifier is still the A100 qwen3 graph service at
  `http://172.16.11.62:8011`.

### Current Canonical Four-Method Run

Current result directory:

```text
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602
```

Download bundle:

```text
transfer/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602_results.zip
```

Config family:

```text
base config: configs/four_method_compare_1xa100.yaml
generated cell config:
  experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/configs/rc8_dw8_d8_observe.yaml

methods: target_only,specedge_pipeline,sled_async,dip_sd
request_count: 8
draft_worker_count: 8
depth: 8
max_new_tokens: 8
network_profile: observe
draft workers: explicit qwen3_graph workers on cuda:0..cuda:7
draft models: alternating Qwen3-0.6B and Qwen3-1.7B
target: one A100 qwen3_graph service at http://172.16.11.62:8011
```

Matrix status:

```text
completed_cell_count: 1
failed_cell_count: 0
incomplete_cell_count: 0
method_row_count: 4
mismatches vs target_only: 0 for specedge_pipeline, sled_async, dip_sd
draft_registry_audit_ok: true
draft_worker_backend_fallback_count: 0
draft_worker_device_count: 8
```

Summary from `matrix_summary.csv` / `matrix_comparison.csv`:

| method | total ms | WSTGR tok/s | speedup vs target | target calls | avg verify batch | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| target_only | 4522.666 | 14.151 | 1.000x | 64 | 1.000 | A100 autoregressive baseline |
| specedge_pipeline | 5221.100 | 12.258 | 0.866x | 19 | 6.800 | fastest speculative method in this cell |
| sled_async | 5301.637 | 12.072 | 0.853x | 72 | 7.000 | strong overlap, but still slower than target in this run |
| dip_sd | 41949.361 | 1.526 | 0.108x | 62 | 3.516 | functionally correct, but dominated by online planner/solver and waits |

Important interpretation:

- The current four-method run is correctness-clean: SpecEdge, SLED, and DiP-SD
  match `target_only`.
- The platform did use eight real draft workers on cuda:0..cuda:7 with no
  backend fallback.
- SpecEdge and SLED are both active on the async pipeline path and use real
  worker lanes in the timeline charts.
- DiP-SD is much closer to the paper algorithm than the old reproduction, but
  performance reproduction is not finished. The run includes a real paper-MILP
  solver backend and steady-state draft prefetch, yet the online solver/planner
  cost is too large for this short max-new-token=8 experiment.

### DiP-SD Implementation State

Main code locations:

```text
src/specplatform/methods/dip_sd/solver.py
src/specplatform/runtime/distributed_pipeline.py
scripts/3090_specedge_smoke.py
tests/test_dip_sd_solver.py
tests/test_distributed_pipeline_runtime.py
tests/test_specedge_smoke_runner.py
```

Implemented pieces:

- Decoupled DiP-SD method path under `src/specplatform/methods/dip_sd/`.
- Paper-style solver mode:
  `dip_sd.solver = paper_milp_or_dinkelbach`.
- PySCIPOpt/SCIP backend is active:
  `dip_sd_solver_backend_name = pyscipopt_scip`.
- Paper-MILP completion evidence:
  `dip_sd_paper_solver_complete = true`.
- Assignment subproblem is implemented as a set-partitioning MILP.
- Draft-length subproblem is implemented as a Dinkelbach-style MILP surrogate.
- Solver cache is present and recorded.
- Distributed draft workers plus central A100 batch verification are wired
  through the shared distributed pipeline.
- Phase-level draft/verify pipeline is active for DiP-SD.
- Steady-state cross-round draft prefetch is implemented and recorded:
  current matrix saw `steady_state_prefetch_submit_count = 23`,
  `steady_state_prefetch_reused_draft_count = 13`.
- Real latency observation/fitting telemetry exists:
  `dip_sd_latency_observation_count = 137`,
  `dip_sd_latency_fit_count = 11`.
- Calibration application path exists and has a separate real smoke proof.

Key DiP-SD artifacts from the canonical run:

```text
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/dip_sd/dip_sd_solver_trace.json
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/dip_sd/dip_sd_latency_calibration.json
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/dip_sd/estimated_vs_actual_pipeline_span.csv
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/dip_sd/solver_time.csv
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/dip_sd/phase_events.csv
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/dip_sd/phase_summary.csv
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/dip_sd/pipeline_stage_timeline.csv
```

Separate DiP-SD smoke proofs:

```text
experiments/dip_sd_paper_milp_reuse_smoke_20260602
  paper_milp: true
  solver backend: pyscipopt_scip
  steady_state_prefetch_submit_count: 3
  steady_state_prefetch_reused_draft_count: 1

experiments/dip_sd_calibrated_paper_milp_real_smoke_20260602
  latency_calibration_applied: true
  paper_milp: true
  solver backend: pyscipopt_scip
  steady_state_prefetch_submit_count: 4
  steady_state_prefetch_reused_draft_count: 4
  dip_sd_solver_total_ms: about 208 ms
```

### Why Current DiP-SD Is Slower

The current DiP-SD result is not slow because it failed correctness or because
the platform only used one 3090. It is slow because this reproduction still pays
a very heavy online planner/solver and waiting cost:

```text
dip_sd_effective_total_ms: 41949.361
dip_sd_solver_total_ms: 32026.967
phase_name_pipeline_planner_wait_ms: 32028.891
draft_ready_wait_total_ms: 5476.935
server_idle_gap_ms: 10786.842
```

This is partly an implementation/system gap and partly a reproduction-policy
choice. The DiP-SD paper's speedups rely on the planning overhead being small,
amortized, cached, or outside the hot path relative to the generated workload.
In this platform run, the paper MILP/Dinkelbach solve is included inside the
online timed path for a short `max_new_tokens=8` workload, so the solver cost
dominates everything else. In other words: the paper algorithmic pieces are now
present, but the performance reproduction is not complete until planning is
calibrated, amortized, cached more aggressively, or evaluated separately.

### Key Plots To Show

Matrix-level plots:

```text
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/plots/matrix_runtime_by_method.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/plots/matrix_speedup_vs_target_only.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/plots/matrix_phase_distribution.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/plots/matrix_target_call_efficiency.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/plots/matrix_verify_batch_size.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/plots/matrix_server_idle_gap.png
```

Per-method timeline/lane plots:

```text
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/target_only/plots/worker_batch_lanes.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/specedge_pipeline/plots/worker_batch_lanes.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/sled_async/plots/worker_batch_lanes.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/dip_sd/plots/worker_batch_lanes.png

experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/target_only/plots/timeline_gantt.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/specedge_pipeline/plots/timeline_gantt.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/sled_async/plots/timeline_gantt.png
experiments/four_method_compare_1xa100_paper_milp_explicit8_matrix_20260602/runs/rc8_dw8_d8_observe/dip_sd/plots/timeline_gantt.png
```

### What To Do Next

Priority order for the next agent:

1. Treat the four-method directory above as the only current comparison result.
   Old DiP-SD and old three-method numbers are useful only for archaeology.
2. Improve DiP-SD hot-path planning:
   cache by request shape/worker profile, avoid resolving identical MILPs,
   make planning asynchronous where possible, and add a no-planner-cost
   sensitivity report so algorithmic pipeline behavior can be separated from
   solver overhead.
3. Run a calibration-first DiP-SD workflow:
   collect latency observations, write the calibration profile, then rerun the
   four-method matrix with `latency_calibration_applied = true`.
4. Explain and reduce the first-draft idle region:
   break out setup/warmup, first planner solve, first assignment, worker
   dispatch, and A100 wait in the lane chart metadata.
5. Expand the matrix after the DiP-SD planner fix:
   vary request count, draft-worker count, depth, max_new_tokens, and network
   profile while keeping explicit real 3090 workers available.
6. Keep the four methods decoupled:
   changes to DiP-SD should stay inside `methods/dip_sd`,
   `runtime/distributed_pipeline.py`, and shared telemetry only when needed.
   Do not regress `target_only`, `specedge_pipeline`, or `sled_async`.

### Verification

Latest local verification:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
/home/chajiahao/miniconda3/envs/edge-specdec/bin/python -m unittest discover -s tests -q

Ran 182 tests in 5.606s
OK
```

Use the `edge-specdec` conda environment for real GPU runs. The base Python has
a different Torch/CUDA stack and is not the reliable path for these experiments.

## Previous Handoff Update, 2026-06-02 Clean Three-Method Matrix and Plot Modes

This section is older than the four-method DiP-SD update above, but fresher than
the historical sections below.

### What Changed

- Removed stale matrix data directories:
  - `experiments/target_specedge_sled_matrix_1xa100_official_20260602`
  - `experiments/target_specedge_sled_matrix_1xa100_official_20260602_v2`
  - the pre-fix partial `experiments/target_specedge_sled_three_method_compare_20260602`
- Added fixed chart batches and plot modes:
  - per-result mode writes the same 9 single-run charts every time:
    `timeline_gantt`, `compact_timeline_distribution`, `phase_breakdown`,
    `round_waterfall`, `overlap_concurrency`, `worker_batch_lanes`,
    `network_breakdown`, `proactive_reuse_chart`, `http_verify_breakdown`;
  - matrix mode writes the same 15 comparison charts every time, including
    runtime, speedup-vs-target, verify batch size, idle gap, SLED metrics,
    method aggregate, phase distribution, and SpecEdge/SLED speedup heatmaps;
  - CLI modes:
    `scripts/run_experiment_matrix.py --plot-mode single|matrix|both|none`;
    `scripts/render_timing_charts.py --mode single|matrix`.
- Fixed official SpecEdge state/worker affinity:
  `SpecEdgePipelinePlanningPolicy` now uses `OfficialSpecEdgeDraftState` to
  keep an existing request on the same draft worker while its persistent
  BatchGraphEngine row is still valid. This fixes the `4-4+` duplicate
  `draft_batch_index` failure where a reused tree was grown on a different
  worker than the one that owned the slot.
- Fixed qwen3 graph draft batch row propagation:
  platform requests now carry explicit `draft_batch_index` through
  `TopKTreeDraftRunner.generate_tree_batch(...)` into
  `Qwen3GraphCausalLMRunner.generate_tree_topk_batch_graph(...)`; the backend
  validates duplicate rows and writes normalized row metadata back.
- Fixed matrix reporting for the current method names:
  `sled_async` is aliased into `sled_*` comparison/report columns, and
  `Top Speedup Cells` now supports `target_only` as the baseline when
  `tree_stop_wait` was not run. `matrix_comparison.csv` now uses
  `best_method` for the true overall winner and adds
  `best_speculative_method` for the speculative-only winner, so target-only is
  no longer silently excluded from that column.
- Added default downloadable result bundle:
  `scripts/run_experiment_matrix.py` and `scripts/render_timing_charts.py`
  now write a lightweight zip bundle under `transfer/` unless
  `--no-result-zip` is set. `--local-results-dir` is now explicit-only and is
  documented as a server-side copy, not the Windows PC. The current Windows
  transfer path is an HTTP download from the server because direct write to
  `C:\` requires a Windows share or credentials.

### Matrix Run

Command family:

```text
scripts/run_experiment_matrix.py
base_config: configs/target_specedge_sled_matrix_1xa100.yaml
output_dir: experiments/target_specedge_sled_three_method_compare_20260602
methods: target_only,specedge_pipeline,sled_async
request-draft pairs: 1-1,2-2,4-4,6-6,8-8,8-1
depth: 8
max_new_tokens: 8
draft workers: explicit qwen3_graph on cuda:0..7
target: one A100 at http://172.16.11.62:8011
```

Matrix status:

```text
completed_cell_count: 6
failed_cell_count: 0
matrix_row_count: 18
mismatches vs target_only: 0
```

Summary table from `matrix_comparison.csv`:

| pair | target ms | specedge ms | specedge vs target | sled ms | sled vs target | fastest overall |
|---|---:|---:|---:|---:|---:|---|
| 1-1 | 525.9 | 959.1 | 0.548x | 678.7 | 0.775x | target_only |
| 2-2 | 988.6 | 1362.9 | 0.725x | 1156.1 | 0.855x | target_only |
| 4-4 | 1974.7 | 2381.7 | 0.829x | 1954.6 | 1.010x | sled_async |
| 6-6 | 2992.2 | 4221.2 | 0.709x | 3611.1 | 0.829x | target_only |
| 8-8 | 4087.6 | 5388.3 | 0.759x | 4822.7 | 0.848x | target_only |
| 8-1 | 4055.2 | 4309.9 | 0.941x | 4592.4 | 0.883x | target_only |

Interpretation:

- Correctness is green for all three methods in all six cells.
- SpecEdge is using A100 qwen3 graph tree verification:
  `tree_attention_batch_qwen3_graph` appears in all SpecEdge cells.
- SpecEdge observes real multi-request target batches from `2-2` onward:
  avg verify batch sizes are about `1.8`, `3.73`, `5.1`, `6.79`, `6.92`
  in the multi-request cells.
- SLED is the best non-target method in 5/6 cells and slightly beats
  target-only only in `4-4` (`1.010x`). SpecEdge is closest at `8-1`
  (`0.941x`) but does not beat target-only in this max-new-token=8 setup.
- The current traces still point to draft/proactive overhead and server idle
  gaps dominating the small-token run. The official tree verify path is active,
  but saved target calls are not enough yet to pay for draft-side qwen3 graph
  work and scheduling overhead.
- A diagnosis report was added at `matrix_diagnosis.md`. Key points:
  previous `matrix_best_methods` was a fastest-speculative-method table and
  excluded `target_only`; the current `target_only` baseline is also 7-12%
  faster than the previous handoff in multi-request cells; SLED's batching
  savings only beat its queue/idle/draft overhead in `4-4`; SpecEdge reduces
  target calls but adds more tree draft/proactive wall time than it saves in
  target HTTP for `max_new_tokens=8`.

### Requested Lane Plots

Per-method worker/batch lane charts were generated for every cell. Useful
examples:

```text
experiments/target_specedge_sled_three_method_compare_20260602/runs/rc8_dw8_d8_observe/specedge_pipeline/plots/worker_batch_lanes.png
experiments/target_specedge_sled_three_method_compare_20260602/runs/rc8_dw8_d8_observe/sled_async/plots/worker_batch_lanes.png
experiments/target_specedge_sled_three_method_compare_20260602/runs/rc8_dw1_d8_observe/specedge_pipeline/plots/worker_batch_lanes.png
experiments/target_specedge_sled_three_method_compare_20260602/runs/rc8_dw1_d8_observe/sled_async/plots/worker_batch_lanes.png
```

Matrix-level plots are under:

```text
experiments/target_specedge_sled_three_method_compare_20260602/plots/
```

Key result files:

```text
experiments/target_specedge_sled_three_method_compare_20260602/matrix_summary.csv
experiments/target_specedge_sled_three_method_compare_20260602/matrix_comparison.csv
experiments/target_specedge_sled_three_method_compare_20260602/matrix_report.md
experiments/target_specedge_sled_three_method_compare_20260602/matrix_diagnosis.md
experiments/target_specedge_sled_three_method_compare_20260602/plots/matrix_speedup_vs_target_only.png
experiments/target_specedge_sled_three_method_compare_20260602/plots/matrix_verify_batch_size.png
experiments/target_specedge_sled_three_method_compare_20260602/plots/matrix_speedup_heatmap_specedge_vs_target.png
experiments/target_specedge_sled_three_method_compare_20260602/plots/matrix_speedup_heatmap_sled_vs_target.png
```

Windows-download bundle status:

```text
HTTP download service: stopped at user request
zip on Linux server: /home/chajiahao/data/specDec/transfer/target_specedge_sled_three_method_compare_20260602_results.zip
size: about 69 MB
```

Important: the user's Windows PC is `172.16.4.5`. Do not describe
`/home/chajiahao/...` as "on the user's computer"; that is the Linux experiment
server. Direct copy into `C:\...` needs a Windows share or credentials.

If the user later asks to download the bundle to Windows, restart the temporary
HTTP server from the Linux experiment server:

```text
setsid -f python3 -m http.server 8765 --bind 0.0.0.0 --directory /home/chajiahao/data/specDec/transfer
```

Then the Windows PowerShell download command is:

```powershell
$dest="$env:USERPROFILE\Desktop\specdec_results"
$zip="$dest\target_specedge_sled_three_method_compare_20260602_results.zip"
New-Item -ItemType Directory -Force $dest
Invoke-WebRequest -Uri "http://172.16.11.60:8765/target_specedge_sled_three_method_compare_20260602_results.zip" -OutFile $zip
Expand-Archive -Force $zip $dest
```

Zip contents to expect after extraction:

```text
target_specedge_sled_three_method_compare_20260602/
  matrix_summary.csv
  matrix_comparison.csv
  matrix_report.md
  matrix_diagnosis.md
  plots/                         # 15 matrix-level comparison charts, png+svg
  runs/<cell>/<method>/plots/     # 9 per-result charts per method, png+svg
```

Important plot locations inside the extracted Windows folder:

```text
target_specedge_sled_three_method_compare_20260602/plots/matrix_speedup_vs_target_only.png
target_specedge_sled_three_method_compare_20260602/plots/matrix_verify_batch_size.png
target_specedge_sled_three_method_compare_20260602/runs/rc8_dw8_d8_observe/specedge_pipeline/plots/worker_batch_lanes.png
target_specedge_sled_three_method_compare_20260602/runs/rc8_dw8_d8_observe/sled_async/plots/worker_batch_lanes.png
```

### Verification

Latest local verification:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
/home/chajiahao/miniconda3/envs/edge-specdec/bin/python -m unittest discover -s tests -q

Ran 158 tests in 5.711s OK
```

Focused new coverage:

- fixed chart mode contracts for single-result and matrix comparison plots;
- matrix comparison/report handles `sled_async` as the SLED column family;
- report top-speedup tables work with `target_only` baseline;
- result zip export includes matrix summaries, reports, and plot folders;
- official SpecEdge planning keeps persistent slots on their original worker;
- qwen3 graph tree-batch generation respects explicit draft batch rows.

## Previous Handoff Update, 2026-06-02 Summary Fix and Smoke Proof

This section is older than the matrix proof above.

### What Changed

- Smoke summary extraction now records official qwen3 graph tree-verify backend
  evidence even when the service returns it under
  `response_timing.target_tree_forward_events[*].metadata` instead of only at
  the top `response_timing.tree_forward_batch_kind` field.
- Added `_tree_forward_batch_kinds_from_timing(...)` in
  `scripts/3090_specedge_smoke.py`.
- Added unit coverage for nested qwen3 graph metadata:
  `tree_attention_batch_qwen3_graph` is reported in
  `method_efficiency["specedge_pipeline"]["tree_forward_batch_kinds"]` and the
  theory check `specedge_tree_forward_batch_kind_recorded` becomes true.

### Verified Live Smoke

Target service:

```text
http://172.16.11.62:8011/health
model_backend=qwen3_graph
supports_tree_forward_batch=true
supports_cuda_graph=true
backend_fallback=false
```

Short real 3090-to-A100 run:

```text
run_id: official_specedge_strict_smoke_1req_summaryfix
output_dir: experiments/official_specedge_strict_smoke_1req_summaryfix
device: cuda:1
methods: target_only,specedge_pipeline
sample_count: 1
max_new_tokens: 8
```

Key `combined_summary.json` result:

```text
matches_target_only.specedge_pipeline: true
generated tokens:
  [10956, 22160, 47116, 54851, 101053, 100751, 100627, 101951]

specedge_pipeline:
  tree_forward_batch_kinds: ["tree_attention_batch_qwen3_graph"]
  tree_backend_fallback_event_count: 0
  target_forward_call_count: 3
  main_target_calls_per_output_token: 0.375
  tree_choice_prefix_count: 13
  tree_batch_compression_ratio: 6.5
  proactive_draft_event_count: 2
  proactive_reconcile_count: 2
  overlap_ratio: 0.980884
  target_verify_urls: ["http://172.16.11.62:8011"]
  avg_verify_batch_size: 1.0

target_only:
  target_forward_call_count: 8
```

Important interpretation:

- This proves the 1-request strict path is using A100 qwen3 graph tree verify
  without backend fallback and matches target-only tokens.
- `avg_verify_batch_size` is still `1.0` because this smoke used one request;
  it does not prove multi-request server batching yet.
- `proactive_reused_token_count` was `0` in this tiny prompt/run, so proactive
  overlap is observed but retained-descendant reuse still needs a larger trace.

### Verification

Focused smoke-runner tests:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
/home/chajiahao/miniconda3/envs/edge-specdec/bin/python -m unittest tests.test_specedge_smoke_runner -q

Ran 11 tests in 2.976s OK
```

Full local suite:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
/home/chajiahao/miniconda3/envs/edge-specdec/bin/python -m unittest discover -s tests -q

Ran 147 tests in 4.535s OK
```

### Current Official SpecEdge Gaps

- Run the request/draft matrix on 3090 draft + one A100 target:
  `1-1`, `2-2`, `4-4`, `6-6`, `8-8`, `8-1` with `max_new_tokens=8`.
- Confirm multi-request batches show `avg_verify_batch_size > 1` and preserve
  `tree_attention_batch_qwen3_graph` with no backend fallback.
- Generate the requested pipeline-lane charts for SpecEdge and SLED from the
  new matrix traces.
- Audit a larger trace for retained proactive descendants being reused from the
  same draft `BatchGraphEngine` row after accept/gather.

## Previous Handoff Update, 2026-06-02 Official Budget Gather Pass

This section is older than the summary-fix smoke proof above.

### What Changed

- qwen3 graph official tree budget trimming now compacts draft KV positions
  with `BatchGraphEngine.gather(...)` whenever the platform-native
  `CandidateTree` list is compressed:
  - first-round `generate_tree_topk_batch_graph(...)` final budget trim;
  - next-round `grow_official_tree_batch_graph(...)` in-loop/final budget trim.
- Budget gather events are recorded in draft timing metadata:
  - `draft.batch_graph_budget_gather`;
  - `draft.batch_graph_official_grow_budget_gather`;
  - metadata flag `official_budget_kv_gather`.
- qwen3 graph official proactive grow now uses official-style budget-bucket
  child selection over existing proactive beams plus incoming children, instead
  of simply stopping as soon as `proactive_max_budget` is reached.
- Proactive graph metadata now reports `official_proactive_budget_pruning`.
- Official proactive draft now has a backend boundary instead of always using
  prefix-list top-k and full-prefix subtree generation:
  - `TopKTreeDraftRunner.generate_official_proactive(...)`
  - `Qwen3GraphCausalLMRunner.generate_official_proactive_graph(...)`
- `SpecEdgeOfficialProactiveDraftPolicy` now prefers the graph proactive
  boundary when available, then falls back to the previous platform-compatible
  path only for non-graph runners.
- qwen3 graph proactive generation now runs in-place on the existing draft
  `BatchGraphEngine` row:
  - no `reset()` and no `prefill()`;
  - leaf selection forwards current tree leaves with explicit tree masks;
  - best bonus/root token is selected from graph logits;
  - proactive root is appended with official POST status;
  - proactive descendants are grown by forwarding `POST_CANDIDATE` nodes on
    the same KV row;
  - returned subtree/statuses are written back through
    `OfficialSpecEdgeSlot.add_proactive_subtree(...)`.
- `OfficialSpecEdgeSlot.add_proactive_subtree(...)` now accepts an explicit
  root status, so graph proactive can mark the root `POST_PROCESSED` when it
  has actually been forwarded, matching official behavior more closely.
- Proactive graph generation respects remaining max-new-token budget by using
  `prompt_len` and `max_new_tokens` passed from the method layer.

### Current Official SpecEdge Gaps

- The major local algorithmic gaps are now mostly at the live-system proof
  layer rather than the unit-test boundary:
  - run 3090/A100 timing experiments and verify draft events contain
    `tree_draft_backend=qwen3_batch_graph_official_grow` and
    `qwen3_batch_graph_official_proactive` without unexpected fallback;
  - inspect timing metadata for `official_budget_kv_gather` when budget trim
    actually compacts a tree;
  - verify A100 `/health` reports qwen3_graph target tree verify, not
    `hf_cached`;
  - inspect real timing traces for whether proactive retained descendants are
    consumed from the same BatchGraphEngine row after accept/gather.
- The platform still intentionally uses `OfficialSpecEdgeDraftState` +
  `CandidateTree` as a platform-native adapter rather than vendoring the
  official tensor `BatchTree` class verbatim.

### Verification

Latest local verification:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
/home/chajiahao/miniconda3/envs/edge-specdec/bin/python -m unittest discover -s tests -q

Ran 146 tests in 5.411s OK
```

Focused new coverage:

- qwen3 first-round official batch tree trim gathers KV from old tree positions
  to compacted tree positions;
- qwen3 next-round official persistent grow trim gathers KV after budget
  compaction;
- official proactive policy prefers the graph backend boundary and does not
  call prefix-list top-k / full-prefill subtree fallback when graph proactive is
  available;
- qwen3 official proactive graph does not call `reset()` or `prefill()`;
- qwen3 official proactive graph forwards leaf selection and POST root growth
  on the persistent batch row with expected cache positions and masks;
- qwen3 official proactive graph budget-bucket pruning keeps high-score
  children and filters low-score incoming children.

## Previous Handoff Update, 2026-06-02 Persistent Official Grow Pass

This section is older than the budget gather pass above.

### What Changed

- Official SpecEdge state now records when the target bonus token has been
  appended to the committed prefix but still needs to be forwarded on the draft
  BatchGraphEngine row:
  - `OfficialSpecEdgeSlot.needs_prefix_tail_forward`
  - no-proactive acceptance sets this flag after commit;
  - fresh/grown tree replacement clears it unless the backend reports it still
    pending.
- `TopKTreeDraftRunner` now exposes a separate official state-grow boundary:
  - `grow_official_tree_batch(...)`
  - this delegates to model backends without changing the generic tree runner
    contract.
- `SpecEdgeOfficialCandidateStrategy` now grows aligned official state trees
  instead of immediately returning stale preserved trees:
  - batches grow requests by backing model;
  - passes existing tree, node statuses, `draft_batch_index`, and the pending
    prefix-tail flag;
  - falls back to old reuse only when the runner has no official grow boundary.
- `Qwen3GraphCausalLMRunner.grow_official_tree_batch_graph(...)` now implements
  the official post-commit draft path:
  - does not call `BatchGraphEngine.reset()` or `prefill()` when a persistent
    draft row exists;
  - forwards the pending bonus/prefix-tail token at `prefix_len - 1`;
  - forwards existing `CANDIDATE` tree nodes in fixed-shape batch tensors;
  - uses explicit `cache_batch_indices`, `cache_seq_indices`, and tree masks;
  - marks selected candidates `PROCESSED`;
  - appends children with the same budget-bucket rule and `log(0.9)` decay;
  - emits metadata such as `official_persistent_kv_reused`,
    `official_state_grow`, and `qwen3_batch_graph_official_grow`.
- Missing `draft_batch_index` is now an explicit fallback:
  `official_persistent_kv_reused=False` with reason
  `missing_draft_batch_index`, followed by the previous full-prefill batch tree
  path.

### Current Official SpecEdge Gaps

- The local code path now has the official gather -> persistent draft grow
  bridge. This still needs live 3090/A100 experiment confirmation from timing
  metadata, especially that next-round draft events show
  `tree_draft_backend=qwen3_batch_graph_official_grow` and no unexpected
  fallback.
- Proactive state semantics are official-style (`POST_CANDIDATE` /
  `POST_PROCESSED` and retention after target bonus match), but the proactive
  subtree generation path should still be audited on GPU traces. The important
  check is whether retained proactive descendants have valid draft KV in the
  same BatchGraphEngine row before they are consumed by later grow steps.
- The platform intentionally uses `OfficialSpecEdgeDraftState` +
  `CandidateTree` as a platform-native adapter rather than vendoring the
  official tensor `BatchTree` class verbatim.
- A100 target tree verify still needs live service confirmation via `/health`
  and timing metadata. Local tests prove the backend boundary uses
  `tree_forward_batch`; they do not prove the currently running service is
  qwen3_graph.

### Verification

Latest local verification:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
/home/chajiahao/miniconda3/envs/edge-specdec/bin/python -m unittest discover -s tests -q

Ran 142 tests in 5.430s OK
```

Focused new coverage:

- official acceptance marks no-proactive bonus as a pending prefix-tail forward;
- official candidate strategy calls persistent state grow when the runner
  supports it;
- qwen3 official persistent grow does not call `reset()` or `prefill()`;
- qwen3 official persistent grow forwards the gathered batch row and expands
  prefix-tail -> candidate nodes with expected statuses.

## Latest Handoff Update, 2026-06-01 Official SpecEdge Pass

This section is older than the 2026-06-02 persistent grow pass above.

### What Changed

- `specedge_pipeline` now defaults to the official-style SpecEdge path when
  `specedge.official: true`.
- Draft tree expansion has a batch entry point:
  - `TopKTreeDraftRunner.generate_tree_batch(...)`
  - `Qwen3GraphCausalLMRunner.generate_tree_topk_batch_graph(...)`
- The qwen3 graph draft path now uses `BatchGraphEngine` for multi-request
  tree expansion, with official-style:
  - per-request prefill into explicit KV;
  - candidate beam selection;
  - fixed-shape batch frontier forward;
  - tree attention masks;
  - `cache_batch_indices` / `cache_seq_indices`;
  - budget-bucket pruning with `log(0.9)` decay and score floor.
- Added `OfficialSpecEdgeDraftState` integration:
  - request slots;
  - node statuses;
  - proposal export;
  - acceptance reorder metadata;
  - state commit only after the winning proposal is selected.
- Runtime now has a generic `commit_acceptance` hook. It is method-agnostic and
  only calls the hook when a policy exposes it.
- Official SpecEdge no longer reuses the old cached-tree proactive path.
  Old proactive is deliberately disabled for the official path so it cannot
  bypass official state.
- Official proactive has since been wired back in through official state:
  - proactive draft appends `POST_CANDIDATE` / `POST_PROCESSED` nodes;
  - if target bonus matches the proactive root, commit keeps the proactive
    descendants and converts POST statuses back to normal statuses;
  - the next official candidate round reuses the preserved state tree instead
    of using the old runtime cached proposal path.
- Accept commit now reaches the draft model backend:
  - runtime passes `draft_runners` through the generic `commit_acceptance` hook;
  - `SpecEdgeOfficialAcceptancePolicy` calls
    `model.official_specedge_commit_acceptance(...)` when available;
  - `Qwen3GraphCausalLMRunner.official_specedge_commit_acceptance(...)` applies
    `BatchGraphEngine.gather(batch_idx, source_seq_indices, dest_seq_indices)`;
  - qwen3 graph draft metadata carries `official_draft_batch_index`, and the
    official slot remembers that row for later gather.

### Current Official SpecEdge Gaps

These are the remaining differences from the official repo implementation:

- Persistent draft KV across decode rounds is not yet preserved with
  `BatchGraphEngine.gather(...)`. The current official path rebuilds/prefills
  the committed prefix each proposal round, then uses BatchGraphEngine for tree
  growth inside that round.
- Proactive now follows the official POST status/reorder semantics at the
  platform state level, but the qwen3 draft backend still rebuilds/prefills more
  than the official in-place CUDA graph implementation. The next gap is using
  the gathered draft KV to avoid unnecessary prefix prefill in all next-round
  expansion paths.
- The platform uses `OfficialSpecEdgeDraftState` + `CandidateTree` as a clean
  platform-native BatchTree adapter, not the official tensor `BatchTree` class
  verbatim.
- A100 target tree verify must still be confirmed on the live service via
  `/health` and timing metadata. The local tests prove the model boundary uses
  `tree_forward_batch`; they do not prove the currently running A100 process is
  using qwen3_graph.
- Legacy `tree` / `tree_stop_wait` classes remain as compatibility baselines and
  test fixtures. `specedge_pipeline` is the official path; old proactive is no
  longer active there.

### Verification

Latest local verification:

```text
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
/home/chajiahao/miniconda3/envs/edge-specdec/bin/python -m unittest discover -s tests -q

Ran 140 tests in 5.962s OK
```

Focused new coverage:

- official candidate strategy batches draft jobs and populates official state;
- official acceptance defers state mutation until winner commit;
- qwen3 draft batch graph path uses one `BatchGraphEngine.forward` step.
- official proactive uses POST statuses;
- target bonus match preserves proactive subtree for the next round.
- qwen3 graph official commit gathers `BatchGraphEngine` KV cache rows.

## Previous Handoff Update, 2026-06-01

This section is older than the official SpecEdge pass above. Use it only for
historical context.

### User Goal

The user wants to stop this window and continue in a new chat. The active
technical goal remains:

```text
P0: implement a real SLED single-pass linear verifier.
P0: implement real SpecEdge draft-side graph/KV tree expansion.
P0: keep SpecEdge 2-2 and 6-6 correctness fixed.
P1: wire official-style server queue/static batch and pipeline-aware draft depth.
P1: implement true qwen3_graph; do not count HF cached fallback as graph.
```

The newest user correction was simply:

```text
真正 graph verifier
```

Meaning: stop trying to make HF fallback look like the solution. The next
window should implement a true graph/KV verifier backend.

### Current Runtime State

3090 workspace:

```text
/home/chajiahao/data/specDec
env: /home/chajiahao/miniconda3/envs/edge-specdec
```

A100 workspace:

```text
ssh alias: a100-specdec
repo: /data/chajiahao/specDec
env: /data/chajiahao/miniconda3/envs/specedge
target model: /data/chajiahao/hf_models/Qwen3-14B
service: http://172.16.11.62:8010
```

Current A100 process:

```text
pid 1076170
/data/chajiahao/miniconda3/envs/specedge/bin/python scripts/a100_target_service.py \
  --model-path /data/chajiahao/hf_models/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8010 \
  --device cuda \
  --attn-implementation eager \
  --backend hf_cached
```

Current health response:

```json
{
  "status": "ok",
  "linear_backend": "a100_http_linear",
  "tree_backend": "a100_http_tree",
  "model_backend": "hf_cached",
  "model_backend_capabilities": {
    "backend_name": "hf_cached",
    "backend_fallback": false,
    "supports_kv_cache": true,
    "supports_cuda_graph": false
  }
}
```

Important: this is **not** a true graph verifier. It is only the current running
service so experiments can hit A100.

### Latest Code Changes In This Window

Touched files in this last stretch:

```text
scripts/a100_target_service.py
src/specplatform/model/loader.py
src/specplatform/model/qwen3_graph.py
src/specplatform/model/transformers.py
tests/test_model_interface.py
```

What changed:

- `a100_target_service.py` now accepts `--backend`, defaulting to `hf_cached`.
- `load_causal_lm_runner(...)` forwards `attn_implementation`.
- `qwen3_graph` fallback forwards `attn_implementation`.
- `CachedTransformersCausalLMRunner` was patched experimentally:
  - clones cached `past_key_values` before reuse;
  - passes explicit `position_ids`, `cache_position`, and attention mask;
  - latest local version builds a 4D additive causal mask for cached tail
    forwards.
- Tests were added around loader forwarding and cached-runner behavior.

Latest local focused test after the experimental transformer edits:

```text
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  -m unittest \
  tests.test_model_interface.CausalLMRunnerInterfaceTest.test_cached_transformers_greedy_uses_kv_cache_with_explicit_positions \
  tests.test_model_interface.CausalLMRunnerInterfaceTest.test_cached_transformers_linear_verify_passes_explicit_tail_positions \
  tests.test_model_interface.CausalLMRunnerInterfaceTest.test_cached_transformers_linear_verify_clones_mutable_prefix_cache \
  tests.test_model_interface.CausalLMRunnerInterfaceTest.test_cached_transformers_linear_verify_reuses_prefix_kv

Ran 4 tests in 1.239s OK
```

Latest full suite **before** the very latest 4D-mask edits:

```text
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python -m unittest discover -s tests
Ran 118 tests in 6.272s OK
```

Next window should rerun the full suite before claiming stability.

### Critical Finding: HF Cached Single-Pass Is Not Exact For Qwen3

This is the most important handoff fact.

We tried to make SLED linear verifier faster using HF `past_key_values`:

```text
prefix prefill + one tail forward over all draft tokens
```

Even with cloned cache objects and explicit `position_ids/cache_position/4D
causal mask`, Qwen3-14B on the current HF path still disagrees with
token-by-token target generation on `mixed-001`.

Prompt:

```text
介绍一下 speculative decoding 的核心思想。
```

Tokenized prefix:

```text
[109432, 65736, 47116, 43589, 100185, 100383, 1773]
```

Target-only `/generate_greedy` result:

```text
[10956, 22160, 47116, 54851, 101053, 18493, 101951, 102064]
```

But `/verify_linear` with a single cached tail forward accepts this alternate
path:

```text
[10956, 22160, 47116, 54851, 101053, 100751, 100627, 101951]
bonus: 102064
linear_forward_batch_kind: linear_single_pass_kv_cache
```

And verifying the target-only path itself mismatches at token index 5:

```text
verified prefix:
[10956, 22160, 47116, 54851, 101053, 100751, ...]

expected target-only token at that position:
18493
```

Conclusion:

```text
Current HF cached "single-pass" fallback is not semantically safe for Qwen3.
Do not use it as the final SLED single-pass verifier.
Do not report it as qwen3_graph.
```

The correct next direction is a true graph/KV verifier, closer to official
SpecEdge: custom Qwen3 attention that writes/reads an explicit KV cache with
`cache_batch_indices`, `cache_seq_indices`, graph-captured fixed shapes, and
explicit tree/linear attention masks.

### Official SpecEdge Implementation Notes

Official repo clone:

```text
/tmp/specedge-official
HEAD seen earlier: 1edcaf02ffc41a7b57726450c5357ed216a3b9bc
```

Key official files to study/adapt:

```text
/tmp/specedge-official/src/model/qwen3.py
/tmp/specedge-official/src/model/cache.py
/tmp/specedge-official/src/specedge/engine/graph.py
/tmp/specedge-official/src/strategy/edge_draft/specexec.py
/tmp/specedge-official/src/strategy/edge_verify/specexec.py
/tmp/specedge-official/src/strategy/server_verify/specexec/grpc.py
```

Official shape:

- `model/qwen3.py` defines a custom `Qwen3ForCausalLM`.
- `model/cache.py` defines explicit tensor KV cache:
  `[layer, batch, kv_head, seq, head_dim]`.
- `GraphEngine` / `BatchGraphEngine` capture CUDA graphs using static tensors:
  - `input_ids`
  - `position_ids`
  - `cache_batch_indices`
  - `cache_seq_indices`
  - `attention_mask`
  - custom `KVCache`
- Server and edge verification use the same graph engine boundary.

This is the right foundation for `src/specplatform/model/qwen3_graph.py`.

### Recommended Next Steps

1. Freeze the HF cached verifier as fallback only.
   - It may be useful for smoke tests, but should be labeled unsafe for exact
     Qwen3 multi-token verification.
   - Consider adding a guard flag like `linear_single_pass_semantic_safe=False`.

2. Implement real `qwen3_graph` in `src/specplatform/model/qwen3_graph.py`.
   - Either vendor/adapt the official Qwen3 model/cache/graph components into
     `src/specplatform/model/`, or create a thin adapter that imports them if
     vendored.
   - Do not silently fallback when experiments request true graph.
   - If `allow_fallback=False`, fail fast.

3. Define a minimal graph verifier milestone:
   - load custom Qwen3 target;
   - prefill prompt into explicit KV cache;
   - verify one linear draft with fixed shape;
   - compare against token-by-token target-only on `mixed-001`;
   - only then mark SLED single-pass as correct.

4. After linear graph verifier is correct, extend the same backend to SpecEdge
   tree verifier:
   - packed tree `input_ids`;
   - tree `position_ids`;
   - `cache_batch_indices`;
   - `cache_seq_indices`;
   - tree attention mask;
   - KV reorder/gather for accepted path.

5. Re-run correctness before performance:
   - local full tests;
   - A100 `/health`;
   - manual mixed-001 probe;
   - rc2/rc6 `target_only,specedge_pipeline,sled_async`;
   - only then regenerate plots.

### Most Recent Experiment Artifacts

Useful but not final:

```text
experiments/request_draft_compare_after_sled_singlepass/latest
experiments/request_draft_compare_after_sled_singlepass_safe_greedy/latest
```

Do not use the second one as final evidence. It was part of diagnosing
full-prefix versus KV-cache semantic drift.

Earlier correctness-fixed but pre-HF-cached-single-pass run:

```text
experiments/request_draft_compare_final_correctness/latest
```

At that time rc2/rc6 SpecEdge and SLED matched target-only, but SLED was using
safe `linear_prefix_batch`, not true single-pass graph verification.

### Dirty Worktree Warning

The repo is dirty and contains many pre-existing edits. Do not revert unrelated
files. Current `git status --short` includes many modified and untracked files,
including:

```text
scripts/a100_target_service.py
src/specplatform/model/transformers.py
src/specplatform/model/loader.py
src/specplatform/model/qwen3_graph.py
src/specplatform/verification/tree.py
src/specplatform/verification/linear.py
tests/test_model_interface.py
tests/test_specedge_tree_core.py
configs/
experiments/ is ignored
```

If you need to revert the experimental HF-cached 4D mask work, do it with a
small targeted patch only after inspecting current code. Never use destructive
git reset/checkout on the whole tree.

This is the current handoff for the speculative decoding experiment platform.
It is written for the server workspace, not for the older computer/Windows
workspace mentioned in earlier notes.

## First Read

- Active 3090 workspace: `/home/chajiahao/data/specDec`
- A100 mirror/workspace: `/data/chajiahao/specDec`
- Treat the 3090 workspace as the current source of truth unless the user says
  otherwise.
- The git worktree is intentionally dirty. Do not run `git reset`,
  `git checkout --`, `git clean`, or destructive sync commands unless the user
  explicitly asks.
- The current platform is no longer only a minimal speculative decoding demo.
  It now contains a broad experiment skeleton for target-only, linear
  speculative decoding, SpecEdge, DiP-SD, SLED, multi-draft workers, timing
  charts, and matrix experiments.
- A100 verifier is currently healthy at `http://172.16.11.62:8010/health`.
- Latest full tests in this handoff session:
  - 3090: `Ran 104 tests in 4.987s OK`
  - A100: `Ran 104 tests in 6.405s OK`

## Machine And Path Map

3090 server:

```text
repo:      /home/chajiahao/data/specDec
env:       /home/chajiahao/miniconda3/envs/edge-specdec
draft 0.6B /home/chajiahao/data/hf_models/Qwen3-0.6B
draft 1.7B /home/chajiahao/data/hf_models/Qwen3-1.7B
GPUs:      8 x NVIDIA GeForce RTX 3090
```

A100 server:

```text
ssh alias: a100-specdec
repo:      /data/chajiahao/specDec
env:       /data/chajiahao/miniconda3/envs/specedge
target:    /data/chajiahao/hf_models/Qwen3-14B
HTTP:      http://172.16.11.62:8010
```

Current 3090 Python snapshot:

```text
python 3.11.15
torch 2.6.0+cu124
transformers 4.51.3
torch.cuda.is_available() == True
torch.cuda.device_count() == 8
```

Current A100 verifier process, as checked during handoff:

```text
/data/chajiahao/miniconda3/envs/specedge/bin/python scripts/a100_target_service.py \
  --model-path /data/chajiahao/hf_models/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8010 \
  --device cuda \
  --attn-implementation eager
```

## Git State

Remote and branch:

```text
origin https://github.com/971027212/SpecDec.git
branch main
HEAD 606c0b3 Harden minimal speculative decoding loop
```

The current work is mostly uncommitted. Important modified or new areas include:

```text
configs/
data/sample_prompts_mixed.jsonl
scripts/3090_speculative_smoke.py
scripts/3090_specedge_smoke.py
scripts/a100_target_service.py
scripts/render_timing_charts.py
scripts/run_experiment_matrix.py
src/specplatform/core/
src/specplatform/draft/
src/specplatform/methods/
src/specplatform/metrics/
src/specplatform/model/
src/specplatform/runtime/
src/specplatform/schedulers/
src/specplatform/timing/
src/specplatform/verification/
tests/
README.md
PROJECT_STRUCTURE.md
HANDOFF_3090.md
```

`experiments/` is ignored by git and contains local evidence/artifacts. Preserve
or summarize important runs before moving machines or cleaning the workspace.

## What The Platform Can Do Now

Implemented method surfaces:

- `target_only`
  - A100 target greedy generation over HTTP.
  - Used as the correctness reference.

- `linear`
  - Basic speculative decoding with greedy draft tokens.
  - A100 linear verifier checks draft token by token and returns accepted prefix,
    mismatch correction, and optional bonus token.

- `tree_stop_wait`
  - SpecEdge-style top-k tree draft and tree verification in a stop-wait loop.
  - Includes tree schemas, target choice events, root guard accounting, and
    fallback labels when a backend cannot truly fuse tree attention.

- `specedge_pipeline`
  - Async/pipeline runtime for SpecEdge-style proactive drafting.
  - Supports background verify future, proactive single-head expansion, and
    reconcile/reuse/discard metadata.
  - This is the branch for SpecEdge overlap experiments.

- `dip_sd`
  - DiP-SD-facing planning policy.
  - Current reproduction path uses greedy linear local draft plus central A100
    batched linear verify.
  - Implements multi-request batch verification, joint batch assignment, and
    draft-length planning.

- `sled`
  - SLED-facing planning policy.
  - Supports heterogeneous draft worker registry, one stable edge-device draft
    stream per request, confidence-triggered dynamic drafting, and shared A100
    batched linear verify.

Cross-cutting implemented pieces:

- Explicit 8-GPU draft worker registry.
- Per-worker model path, device, backend, draft type, and speed profile.
- Warmup events that update observed draft speed profiles.
- HTTP timing schema for client serialize/deserialize, response validation,
  server total, target forward events, and network/queue residuals.
- Batch schemas and endpoints:
  - `/verify_linear_batch`
  - `/verify_tree_batch`
- A100 `verify_linear_batch` now does real batched linear verification where
  safe, buckets variable-length prefixes, and deduplicates identical prefixes.
- Timing artifacts:
  - `phase_events.jsonl`
  - `phase_events.csv`
  - `phase_summary.csv`
  - `request_results.json`
  - `smoke_output.json`
  - `combined_summary.json`
- Plot suite:
  - `timeline_gantt`
  - `compact_timeline_distribution`
  - `phase_breakdown`
  - `round_waterfall`
  - `overlap_concurrency`
  - `worker_batch_lanes`
  - `http_verify_breakdown`
  - `network_breakdown`
  - `proactive_reuse_chart`
  - `timing_audit`

## Important Architecture Boundaries

Keep this separation if continuing development:

```text
runtime/       orchestrates sessions, plans, draft jobs, verify batches, accept, append
methods/       candidate strategies, acceptance policies, planning policies
draft/         draft generation only
verification/  verifies proposals and returns facts; does not mutate sessions
schedulers/    request selection, worker assignment, batch construction
model/         local causal LM backends and capability reporting
timing/        spans/events/attribution; no correctness decisions
metrics/       artifacts, summaries, plots
```

The synchronous runtime should not branch on method names for algorithm
behavior. Method differences should enter through strategies, policies, and
verifier/candidate abstractions.

## Current SpecEdge Status

Implemented:

- Tree candidate strategy and tree acceptance.
- Stop-wait tree speculative decoding.
- Async pipeline runtime with `VERIFY_IN_FLIGHT` and proactive drafting.
- Proactive single expansion head.
- Reconcile logic that can reuse proactive suffix when aligned.
- A100 tree batch schema and service endpoints.
- Timing for server and client verify breakdown.
- Pipeline lane and overlap charts.

Important limitations:

- The `qwen3_graph` backend currently reports fallback behavior rather than a
  true optimized CUDA graph / KV-managed serving implementation.
- Tree attention may fall back depending on backend capability; timing labels
  must be inspected before claiming fused tree attention speedup.
- Current experiments should be interpreted as platform/reproduction evidence,
  not paper-level absolute throughput.
- SpecEdge is the only method currently attached to the async proactive runtime.
  DiP-SD/SLED are kept on their original-style stop-wait/batch reproduction path
  unless deliberately experimenting with cross-method optimization.

## Current DiP-SD/SLED Status

The user explicitly wants original-method reproduction before later
optimization. Current code follows that split:

- DiP-SD and SLED use `LinearCandidateStrategy`, `GreedyPrefixAcceptancePolicy`,
  and `HttpLinearVerifierClient`.
- They do not use SpecEdge tree candidate logic.
- They do not use SpecEdge proactive single-head drafting in the reproduction
  report.

DiP-SD implemented pieces:

- Central server batch verification.
- Multi-user batch verification.
- Batched linear target verification.
- Joint batch assignment planning.
- Joint draft-length planning.

DiP-SD partial/missing:

- Current latest run did not observe multi-device local drafting for DiP-SD.
- Phase-level draft/verify pipeline is not active in the current DiP-SD run.
- This is not yet a full distributed multi-client reproduction.

SLED implemented pieces:

- Shared server batch verification.
- Heterogeneous draft worker registry.
- Single edge-device draft stream per request.
- Confidence-triggered dynamic drafting.
- Shared server verifies candidate batch.
- Batched linear target verification.
- Edge-device worker assignment planning.

SLED partial/missing:

- Workers are local 3090 processes/devices, not physically separate edge
  devices.
- The system is still an experiment runner, not a production SLED serving
  stack.

Zero-draft fallback note:

- DiP-SD can still select draft depth `0` when observed draft cost is worse
  than target verification.
- SLED no longer uses zero-draft fallback in the strict paper-reproduction path;
  it emits a local draft stream and relies on confidence-triggered verification.
- This prevents forced negative-speedup runs on the current fallback backend.
- Treat this as a cost-model guard for the experiment platform, not as proof of
  paper-level speculative draft speedup.
- For strict forced-draft ablations, add or set `allow_zero_draft_fallback:
  false` in the method config and expect it may be slower on the current backend.

## Latest Verified Smoke

Latest short real 3090-to-A100 run:

```text
experiments/3090_a100_dipsd_sled_repro_fix_v6_restarted/latest
```

Command used:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/3090_specedge_smoke.py \
  --config configs/specedge_dipsd_sled_8gpu_smoke.yaml \
  --methods target_only,dip_sd,sled \
  --max-new-tokens 2 \
  --run-id 3090_a100_dipsd_sled_repro_fix_v6_restarted \
  --output-dir experiments/3090_a100_dipsd_sled_repro_fix_v6_restarted/latest \
  --no-plots
```

Plots were rendered after the run with `scripts/render_timing_charts.py`.

Key result from `combined_summary.json`:

```text
matches_target_only:
  dip_sd: true
  sled:   true

target_only:
  http_total_ms:              2153.316071
  target_forward_call_count:  16
  avg_verify_batch_size:      1.0

dip_sd:
  runtime_round_total_ms:     1220.275530
  http_total_ms:              1161.887145
  target_forward_call_count:  14
  avg_verify_batch_size:      8.0
  draft_worker_count:         1
  avg_candidate_count:        1.0

sled:
  runtime_round_total_ms:     1138.307752
  http_total_ms:              1114.045711
  target_forward_call_count:  14
  avg_verify_batch_size:      16.0
  draft_worker_count:         2
  avg_candidate_count:        2.0
```

Useful plot paths:

```text
experiments/3090_a100_dipsd_sled_repro_fix_v6_restarted/latest/target_only/plots/
experiments/3090_a100_dipsd_sled_repro_fix_v6_restarted/latest/dip_sd/plots/
experiments/3090_a100_dipsd_sled_repro_fix_v6_restarted/latest/sled/plots/
```

Open first:

```text
compact_timeline_distribution.png
worker_batch_lanes.png
http_verify_breakdown.png
network_breakdown.png
timing_audit.txt
```

## Important Configs

Current config files:

```text
configs/specedge_v1_smoke.yaml
configs/specedge_pipeline_smoke.yaml
configs/specedge_pipeline_multidraft_smoke.yaml
configs/specedge_dipsd_sled_multidraft_smoke.yaml
configs/specedge_dipsd_sled_real_heterogeneous_smoke.yaml
configs/specedge_dipsd_sled_8gpu_smoke.yaml
```

Use `configs/specedge_dipsd_sled_8gpu_smoke.yaml` for 8-GPU 3090 worker
experiments. It defines eight draft workers alternating Qwen3-0.6B and
Qwen3-1.7B across `cuda:0` to `cuda:7`.

Note: using an 8-worker registry does not mean every method/run will use all
eight workers. The scheduler may select fewer workers when cost-aware planning
or zero-draft fallback says more workers would hurt.

## Commands

Check A100 health:

```bash
curl --noproxy '*' -sS --max-time 5 http://172.16.11.62:8010/health
```

Start/restart A100 verifier:

```bash
ssh a100-specdec
cd /data/chajiahao/specDec
PYTHONPATH=src nohup /data/chajiahao/miniconda3/envs/specedge/bin/python \
  scripts/a100_target_service.py \
  --model-path /data/chajiahao/hf_models/Qwen3-14B \
  --host 0.0.0.0 \
  --port 8010 \
  --device cuda \
  --attn-implementation eager \
  > /tmp/specdec_a100_service.log 2>&1 &
```

Run 3090 tests:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  -m unittest discover -s tests -v
```

Run A100 tests:

```bash
ssh a100-specdec 'cd /data/chajiahao/specDec && \
  PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  /data/chajiahao/miniconda3/envs/specedge/bin/python \
  -m unittest discover -s tests -v'
```

Sync 3090 to A100:

```bash
rsync -a src scripts tests configs data README.md PROJECT_STRUCTURE.md \
  a100-specdec:/data/chajiahao/specDec/
```

Use `--delete` only if the user explicitly wants the A100 workspace to be made
identical to the 3090 workspace.

Run the latest DiP-SD/SLED/target-only smoke:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/3090_specedge_smoke.py \
  --config configs/specedge_dipsd_sled_8gpu_smoke.yaml \
  --methods target_only,dip_sd,sled \
  --max-new-tokens 2 \
  --run-id 3090_a100_dipsd_sled_next \
  --output-dir experiments/3090_a100_dipsd_sled_next/latest \
  --plot-formats png,svg
```

Run SpecEdge V1 smoke:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/3090_specedge_smoke.py \
  --config configs/specedge_v1_smoke.yaml \
  --use-sample-prompts \
  --run-id 3090_a100_specedge_next \
  --output-dir experiments/3090_a100_specedge_next/latest \
  --plot-formats png,svg
```

Render charts offline:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/render_timing_charts.py \
  --input-dir experiments/<run>/latest/<method> \
  --formats png,svg
```

Run matrix experiments:

```bash
cd /home/chajiahao/data/specDec
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/run_experiment_matrix.py \
  --base-config configs/specedge_dipsd_sled_8gpu_smoke.yaml \
  --output-dir experiments/method_matrix_next/latest \
  --request-counts 1,2,4,8,16 \
  --draft-worker-counts 1,2,4,8 \
  --depths 1,2,4,8 \
  --network-profiles observe,low_uplink,high_rtt \
  --draft-worker-mode explicit \
  --worker-speed-profile heterogeneous \
  --resume \
  --rerun-mismatches \
  --continue-on-error
```

## Timing Interpretation

Use the 3090 monotonic timestamps as the only global timeline. A100 timings are
nested server breakdowns inside 3090-side HTTP spans. Do not compare A100
`perf_counter` timestamps directly against 3090 timestamps.

Main timing phases:

```text
setup.load_draft_model
setup.warm_draft_worker
scheduler.plan
draft.generate
draft.token_forward
verify.batch_total
verify.http_total
verify.client_serialize
verify.client_deserialize
verify.response_validate
accept.apply
session.append
runtime.round_total
artifact.write
plot.render
```

For real elapsed phase cost, prefer:

```text
event_scope=system
span_kind=leaf
```

Do not double-count aggregate or attribution-only events.

Chart reading:

- `compact_timeline_distribution`: first glance at per-round major phase share.
- `timeline_gantt`: full detailed timeline.
- `worker_batch_lanes`: request/draft/A100 lane view.
- `http_verify_breakdown`: server/HTTP verification breakdown.
- `network_breakdown`: serialize/upload/server/downlink/residual view.
- `overlap_concurrency`: overlap/concurrency behavior.
- `timing_audit.txt`: warnings and coverage checks.

## Correctness Gates

For smoke experiments:

```text
combined_summary.matches_target_only.<method> == true
target mismatches == 0
negative_network_residual_count == 0, unless explicitly investigating clock/measurement issues
timing_audit has no unexpected warning
```

For DiP-SD/SLED reproduction runs, also inspect:

```text
avg_verify_batch_size
target_forward_call_count
linear_forward_batch_kinds
draft_worker_count
avg_candidate_count
method_reproduction.<method>.partial_or_missing
```

For SpecEdge pipeline runs, inspect:

```text
overlap_ratio
proactive_draft_event_count
proactive_reused_token_count
proactive_discarded_token_count
server_idle_gap_ms
tree_forward_batch_kinds
```

## Known Gaps And Risks

- The repo is not clean/committed. This is the biggest operational risk.
- `qwen3_graph` is currently fallback-labeled. Do not claim CUDA graph speedups
  until a true optimized backend is wired and verified.
- Current DiP-SD positive result is largely from server batching and avoiding
  harmful draft, not from fast local draft-token speedup.
- DiP-SD is not yet a full distributed multi-client implementation.
- SLED uses local 3090 GPUs as heterogeneous worker stand-ins, not physical edge
  devices.
- SpecEdge async pipeline exists, but paper-level performance requires stronger
  serving support: KV cache discipline, real batched tree attention, CUDA graph
  or vLLM/TensorRT-LLM-like serving, and lower overhead.
- Network `uplink_mbps`, `downlink_mbps`, and `rtt_ms` are modeled/observed by
  default. No real `tc/netem` shaping is active unless explicitly added later.
- Experiments in `experiments/` are ignored by git; preserve summaries manually
  if they matter.
- `rg` was previously unavailable on this machine. If still unavailable, use
  `find` and `grep`.

## Recommended Next Work

1. Create a branch or commit checkpoint for the current uncommitted platform.
2. Keep zero-draft fallback scoped to DiP-SD practical experiments; SLED strict
   reproduction should stay on confidence-triggered local drafting.
3. Run two DiP-SD/SLED experiment families:
   - DiP-SD practical cost-aware fallback enabled
   - SLED strict dynamic-drafting/batched-verification comparison
4. Improve real draft serving speed before expecting paper-style speculative
   speedup from local draft tokens.
6. For SpecEdge, continue from real fused tree/pipeline serving rather than only
   charts or schema work.
7. For matrix experiments, compare method results with correctness gates first,
   then timing.

## Quick Troubleshooting

A100 service not reachable:

```bash
curl --noproxy '*' -sS --max-time 5 http://172.16.11.62:8010/health
ssh a100-specdec 'tail -80 /tmp/specdec_a100_service.log'
ssh a100-specdec 'ps -eo pid,ppid,stat,etime,pcpu,pmem,cmd | grep -E "a100_target_service|8010" | grep -v grep'
```

3090 CUDA looks wrong:

```bash
nvidia-smi
/home/chajiahao/miniconda3/envs/edge-specdec/bin/python -c \
  "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

Smoke output mismatches target-only:

```text
1. Open combined_summary.json.
2. Compare generated_ids_by_method per request.
3. Inspect request_results.json for the first mismatching request.
4. Check tokenizer/model paths and EOS ids.
5. Re-run with one prompt and max_new_tokens=1 or 2.
6. Inspect verifier response metadata and target_forward_events.
```

Plots missing:

```bash
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/render_timing_charts.py \
  --input-dir experiments/<run>/latest/<method> \
  --formats png,svg
```

## Final Caution

This handoff describes the current experimental platform, not a polished
released package. The code is valuable but midstream: keep the dirty worktree
safe, verify after every sync, and separate "paper reproduction", "practical
guardrails", and "future optimizations" when reporting results.
