# Three-Method Tuning and Comparison Workflow

This workflow is for performance / paper-reproduction comparisons among:

```text
specedge_pipeline
sled_async
dip_sd
```

Correctness smoke can run before tuning.  Performance conclusions should not be
made until each method has been tuned under the same fairness envelope and then
evaluated on a locked configuration.

## Fairness Envelope

Keep these fixed during tuning and final comparison:

```text
same tuning prompt source for all methods
separate held-out prompt source for final comparison
same A100 target verifier
same explicit 3090 draft-worker pool
same max_new_tokens
same network profile
same correctness gate against target_only
```

Each method may tune only its own core parameters.

## Tuning Runner

Use `scripts/run_method_tuning.py` to expand method-specific tuning grids into
ordinary `scripts/3090_specedge_smoke.py` configs.

Dry-run the grid first:

```bash
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/run_method_tuning.py \
  --output-dir experiments/three_method_tuning/latest \
  --max-candidates-per-method 1 \
  --tuning-prompts-file data/sample_prompts_mixed.jsonl \
  --heldout-prompts-file data/sample_prompts_heldout.jsonl \
  --dry-run
```

Run the tuning grid:

```bash
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/run_method_tuning.py \
  --base-config configs/four_method_compare_1xa100.yaml \
  --output-dir experiments/three_method_tuning/latest \
  --request-count 8 \
  --tuning-prompts-file data/sample_prompts_mixed.jsonl \
  --heldout-prompts-file data/sample_prompts_heldout.jsonl \
  --heldout-request-count 8 \
  --draft-worker-count 8 \
  --max-new-tokens 8 \
  --network-profile observe \
  --continue-on-error
```

The default small grid is:

```text
SpecEdge:
  tree.max_depth = 4,8
  tree.branch_width = 4,8
  tree.max_budget = 20
  pipeline.proactive_depth = 4,8

SLED:
  sled.max_speculation_tokens = 4,8
  sled.confidence_threshold = 0.4,0.5,0.6
  sled.batch_size = 4,8
  sled.async.proactive_tokens = 4,8

DiP-SD:
  dip_sd.max_draft_length = 4,8
  dip_sd.solver = paper_milp_or_dinkelbach
  calibration off by default
```

If DiP-SD calibration has already been collected, add it as a tuning axis:

```bash
--dip-sd-calibration-profiles none,experiments/<calibration-run>/dip_sd/dip_sd_latency_calibration.json
```

## Selection Rule

The runner selects the best candidate per method only when:

```text
status == completed
matches_target_only == true
effective_total_ms > 0
```

Among those candidates, it chooses the lowest `effective_total_ms`.

Artifacts written by the runner:

```text
tuning_configs/<method>/*.yaml
tuning_runs/<method>/<candidate>/
tuning_summary.csv
tuning_summary.json
best_tuning.json
locked_three_method_compare.yaml
next_commands.md
```

`locked_three_method_compare.yaml` is written only after all three methods have
at least one correctness-clean candidate and a held-out prompt file is
available.  The locked config points `data.sample_prompts` at the held-out file,
so final comparison does not reuse the tuning prompts.

## Final Comparison

Use the locked config as the base for a held-out matrix:

```bash
PYTHONPATH=src /home/chajiahao/miniconda3/envs/edge-specdec/bin/python \
  scripts/run_experiment_matrix.py \
  --base-config experiments/three_method_tuning/latest/locked_three_method_compare.yaml \
  --output-dir experiments/three_method_locked_matrix/latest \
  --methods target_only,specedge_pipeline,sled_async,dip_sd \
  --request-counts 1,2,4,8,16 \
  --draft-worker-counts 1,2,4,8 \
  --max-new-token-counts 8,16,32,64 \
  --depths locked \
  --network-profiles observe,low_uplink,high_rtt \
  --resume \
  --rerun-mismatches \
  --continue-on-error
```

`--depths locked` preserves the tuned `tree.max_depth` and
`pipeline.proactive_depth` from `locked_three_method_compare.yaml`.  Use this
final matrix, not the tuning grid, for performance claims.
