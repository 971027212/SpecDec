# 三方法 locked held-out 矩阵阶段性诊断

更新时间：2026-06-02

## 当前结论

不建议现在继续把 240 个 cell 全部跑完。

原因不是 correctness 不稳定，而是相反：当前已经完成的 69 个完整 cell 全部
`matches_target_only=true`，说明 verifier 修复后正确性已经比较稳。现在继续全矩阵
主要是在消耗 GPU 确认“现有实现整体慢于 target_only”。更有价值的下一步是先优化
方法本身，再重新跑缩小版矩阵验证收益。

## 已完成矩阵范围

正式 held-out 矩阵目录：

```text
/home/chajiahao/data/specDec/experiments/three_method_locked_matrix_verifyfix/latest
```

已完成完整 summary：

```text
completed_cells: 69
correctness_clean_cells: 69
mismatch_count: 0
```

矩阵原计划：

```text
request_count: 1,2,4,8,16
draft_worker_count: 1,2,4,8
max_new_tokens: 8,16,32,64
network_profile: observe,low_uplink,high_rtt
total_cells: 240
```

当前主要覆盖了：

- `request_count=1` 的全部 48 个 cell。
- `request_count=2` 的前 21 个完整 cell。

## locked 较优超参

来自：

```text
/home/chajiahao/data/specDec/experiments/three_method_tuning_verifyfix/latest/best_tuning.json
```

SpecEdge：

```text
tree.max_depth = 4
tree.branch_width = 8
tree.max_budget = 20
specedge.official = true
pipeline.proactive_depth = 8
```

SLED：

```text
sled.batch_size = 8
sled.confidence_threshold = 0.6
sled.max_speculation_tokens = 4
sled.async.proactive_tokens = 8
```

DiP-SD：

```text
dip_sd.max_draft_length = 4
dip_sd.solver = paper_milp_or_dinkelbach
dip_sd.plan_cache_enabled = true
dip_sd.steady_state_enabled = true
dip_sd.calibration_enabled = false
```

补充验证：已经单独做过一次 DiP-SD calibration on/off 小网格，见后文
“DiP-SD calibration 验证”。

## 当前 speedup vs target_only

下面是 69 个完整 held-out cell 的阶段性统计。speedup 的定义是：

```text
target_only_effective_total_ms / method_effective_total_ms
```

大于 1 才表示比 target_only 快。

| method | mean speedup | median | best | worst |
| --- | ---: | ---: | ---: | ---: |
| SpecEdge | 0.657x | 0.614x | 0.992x | 0.467x |
| SLED | 0.662x | 0.654x | 0.909x | 0.478x |
| DiP-SD | 0.465x | 0.431x | 0.701x | 0.304x |

阶段性结论：

- 三个方法当前整体都没有超过 `target_only`。
- `request_count=2` 比 `request_count=1` 明显更接近 target_only，说明 batching 场景是有希望的。
- 当前最接近 target_only 的 cell 是 `rc2_dw2_dlocked_mt16_observe`：
  - SpecEdge：0.992x
  - SLED：0.909x
  - DiP-SD：0.646x

## 分维度观察

按 request_count：

| request_count | cells | SpecEdge | SLED | DiP-SD |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 48 | 0.567x | 0.616x | 0.407x |
| 2 | 21 | 0.866x | 0.774x | 0.603x |

解释：

- 单请求时 speculative 方法没有足够 batch/reuse 空间，overhead 容易超过 target call 节省。
- 两请求时 SpecEdge 已明显接近 target_only，但还没有稳定超过。

按 max_new_tokens：

| max_new_tokens | cells | SpecEdge | SLED | DiP-SD |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 18 | 0.709x | 0.682x | 0.519x |
| 16 | 18 | 0.657x | 0.675x | 0.473x |
| 32 | 18 | 0.616x | 0.666x | 0.431x |
| 64 | 15 | 0.638x | 0.631x | 0.436x |

解释：

- 当前实现没有随着生成长度变长稳定放大收益。
- SpecEdge/SLED 的 draft/proactive/idle 开销仍然吃掉 target call savings。
- DiP-SD 的 stage wait 和规划/调度开销仍然偏重。

## 为什么现在不建议继续跑满 240 cell

1. correctness 已经足够证明当前 verifier 修复方向正确。
   - 69 个完整 held-out cell 全部 match。
   - 继续跑满主要增加 confidence，不会改变“当前实现整体慢”的事实。

2. full matrix 会越来越慢。
   - 后续 `request_count=4/8/16`、`max_new_tokens=64` 会显著增加耗时。
   - DiP-SD 在大并发下 solver/planner 和 stage wait 可能继续放大。

3. 已经发现明确优化方向。
   - SpecEdge：接近 target_only，但 server idle 和 proactive 丢弃仍明显。
   - SLED：target call 没有减少，主要靠 overlap，收益不够。
   - DiP-SD：batch size 没起来，linear verifier call 基本等于 target_only，planner 优势没有体现。

所以当前最合理的策略是：

```text
暂停 full matrix
保留 69 个 clean held-out cell
先做方法优化
再跑一个小而集中的优化验证矩阵
```

## 方法本身需要优化吗

需要。当前不是单纯“参数没调好”，而是实现/系统路径还没有把论文方法的核心收益完全释放出来。

### SpecEdge 优先优化点

观察：

- 最好 cell 已经到 0.992x，说明它最接近 target_only。
- 多请求时 verify batch size 可以到约 2。
- tree target calls 有下降，但 server idle gap 和 proactive discarded tokens 仍然高。

优先优化：

1. 降低 server idle gap。
   - 让下一批 draft 更早 ready。
   - 对 pipeline depth/proactive depth 做按 RTT/verify time 的自适应，而不是固定 locked 值。

2. 提高 proactive reuse 的有效收益。
   - 当前 proactive alignment 不等于有效加速；很多 proactive token 仍被丢弃。
   - 需要统计 retained subtree/token 数，而不是只看 proactive draft 数。

3. 降低 tree draft payload 和 guard overhead。
   - branch_width=8 在某些场景可能 payload 偏大。
   - 可以尝试 branch_width=4 + 更精确 proactive，或动态 branch_width。

### SLED 优先优化点

观察：

- SLED 当前平均 0.662x，最好 0.909x。
- 它的 target forward call 基本仍接近 target_only。
- 优势主要来自 async overlap，不是减少 target calls。

优先优化：

1. 重新校准 confidence threshold。
   - 当前 `0.6` 是小网格最优，但不一定对 held-out 长输出最佳。
   - 应该记录 acceptance/confidence 曲线，做 per-model/per-task threshold。

2. 改 static queue 策略。
   - 当前 queue batch 有时提升 batch size，但也增加 idle/wait。
   - 需要在 `queue_max_wait_ms` 上调小网格，而不是只用默认/null。

3. 增强 proactive reuse。
   - 只有 parent 全接受且 prefix 完全对齐才复用，条件较苛刻。
   - 可以考虑更细粒度地复用 prefix-aligned 的一段，而不是全丢。

### DiP-SD 优先优化点

观察：

- DiP-SD 当前平均 0.465x，最好 0.701x。
- 很多 cell 的 avg verify batch size 仍是 1.0。
- target calls 基本等于 target_only，说明 speculative savings 没有出来。
- paper MILP backend 是通的，但规划收益没有转成执行收益。

优先优化：

1. latency calibration 已验证，但还不够。
   - 当前 locked config 是 calibration off。
   - 用已产生的 `dip_sd_latency_calibration.json` 重跑后，best speedup 只从约 0.100x 提升到 0.126x。
   - calibration 能降低 draft ready wait 和 server idle，但没有消除约 30s 的 online solver/planner 开销。

2. 减少 online solver 进入 hot path。
   - 采用更强 plan cache。
   - 对相同 request shape 预求 plan。
   - 或先跑 no-planner-cost sensitivity，把算法调度收益和 solver 成本拆开。

3. 修正 batch/stage 执行收益。
   - 当前多请求下 avg verify batch size 没明显上去。
   - 要检查 preferred_batches 到 scheduler/runtime 的落地，确认 DiP-SD 真的形成多用户 verify batch。

## DiP-SD calibration 验证

小网格目录：

```text
/home/chajiahao/data/specDec/experiments/dip_sd_calibration_tuning_verifyfix/latest
```

使用的 calibration profile：

```text
/home/chajiahao/data/specDec/experiments/three_method_locked_matrix_verifyfix/latest/runs/rc2_dw2_dlocked_mt16_observe/dip_sd/dip_sd_latency_calibration.json
```

profile 推荐参数：

```text
dip_sd_draft_beta = 65.28746926666665
dip_sd_draft_c = 0.0
dip_sd_verify_beta = 3.4492080640198424
dip_sd_verify_c = 1.3940573138835278e-09
```

调参脚本也修了一个实际问题：calibration profile 现在会写成绝对路径，否则候选 config
会把相对路径按 config 所在目录解析，导致 `FileNotFoundError`。

候选结果：

| max_draft_length | calibration | match | method ms | target ms | speedup | solver/planner ms | draft-ready wait ms | avg verify batch |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | off | true | 37734.940 | 3786.050 | 0.100x | 28359.703 | 4682.649 | 3.516 |
| 4 | on | true | 34559.991 | 4351.203 | 0.126x | 29905.735 | 1002.930 | 3.780 |
| 8 | off | true | 40729.294 | 4474.060 | 0.110x | 31354.022 | 4899.786 | 3.516 |
| 8 | on | true | 35020.223 | 4406.346 | 0.126x | 30440.602 | 903.894 | 3.780 |

结论：

- calibration 是有效的：它让 draft-ready wait 从约 4.7-4.9s 降到约 0.9-1.0s。
- 但 calibration 没有带来数量级改善：best speedup 仍只有 0.126x。
- 核心瓶颈转移得很清楚：`dip_sd_solver_total_ms` 仍接近 30s，占据绝大部分 runtime。
- 因此 DiP-SD 下一步不应继续扩大普通超参网格，而应优先优化/旁路 online solver 成本。

## 推荐下一步

不要继续 240-cell full matrix。建议切到下面的优化顺序：

1. DiP-SD solver/planner hot-path 优化。
   - calibration-first 已完成，结论是有改善但远不够。
   - 先实现/验证 no-online-solver 或 stronger-plan-cache 路径，把调度收益和 solver 成本拆开。

2. SpecEdge focused sweep。
   - 固定 `request_count=2,4`，`max_new_tokens=16,32`。
   - 调 `branch_width=4,8`、`proactive_depth=4,8,12`。
   - 目标是压 server idle gap 和 discarded proactive tokens。

3. SLED queue/threshold sweep。
   - `confidence_threshold=0.55,0.6,0.65`
   - `queue_max_wait_ms=0,2,5`
   - `batch_size=4,8`

4. 再跑一个小矩阵：

```text
request_count: 2,4,8
draft_worker_count: 2,4,8
max_new_tokens: 16,32
network_profile: observe,high_rtt
```

如果小矩阵里某个方法稳定超过 target_only，再回头跑完整 240-cell final matrix。
