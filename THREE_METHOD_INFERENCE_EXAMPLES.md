# 三种论文方法与三种代码方法的一次推理例子

更新时间：2026-06-02

这份说明接手 `HANDOFF_3090.md`，但不把 handoff 里的实验结论当作最终结论。原因是当前代码已经修过 linear verifier 的 causal-safe 语义，旧的三方法调参和旧的 DiP-SD 数字都可能受影响。下面先按论文和代码分别解释一次推理怎么发生，最后总结不足。

参考材料：

- DiP-SD 论文：`/home/chajiahao/.codex/attachments/3796c4f9-364b-4a07-a4b7-4687df2394e3/Xu 等 - 2026 - DiP-SD Distributed Pipelined Speculative Decoding for Efficient LLM Inference at the Edge.pdf`
- SLED 论文：`/home/chajiahao/.codex/attachments/f70b2c02-f532-4df1-a841-581aed03fbdb/Li 等 - 2025 - SLED A Speculative LLM Decoding Framework for Efficient Edge Serving.pdf`
- SpecEdge 论文：`/home/chajiahao/.codex/attachments/654ce2e2-76cd-4b79-8933-bb54890026a4/Park 等 - 2025 - SpecEdge Scalable Edge-Assisted Serving Framework for Interactive LLMs.pdf`
- Handoff：`/home/chajiahao/data/specDec/HANDOFF_3090.md`
- 当前调参流程：`/home/chajiahao/data/specDec/THREE_METHOD_TUNING.md`

## 先回答：什么时候可以做三方法对比

可以先做 correctness 对比，但不能马上下性能结论。性能对比应等这三步完成：

1. 在同一个 tuning prompt set 上，为 SpecEdge、SLED、DiP-SD 各自找一组 correctness-clean 的较优配置。
2. 锁定配置，写入 `locked_three_method_compare.yaml`，不能在最终矩阵里继续调参。
3. 在 held-out prompt set 上跑最终矩阵，所有 speculative 方法必须 `matches_target_only=true`，再比较速度、吞吐、target call、verify batch、idle gap。

不一定要找全局最优配置，那成本太高；但必须找公平约束下的较优配置。否则三方法对比会变成“谁的参数更随便”，不是“谁的方法更好”。

## 当前已经完成的前置工作

本轮已经在 verifier 修复后重新跑了三方法 tuning，共 34 个候选：

- SpecEdge：8 个候选全部完成，全部 match target_only。
- SLED：24 个候选全部完成，全部 match target_only。
- DiP-SD：2 个候选全部完成，全部 match target_only。

当前 locked config 是：

- SpecEdge：`depth=4, branch_width=8, max_budget=20, official=true, proactive_depth=8`。
- SLED：`batch_size=8, confidence_threshold=0.6, max_speculation_tokens=4, proactive_tokens=8`。
- DiP-SD：`solver=paper_milp_or_dinkelbach, max_draft_length=4, min_batch_count=2, plan_cache=true, steady_state=true`。

对应文件：

- tuning 结果：`/home/chajiahao/data/specDec/experiments/three_method_tuning_verifyfix/latest/best_tuning.json`
- locked 配置：`/home/chajiahao/data/specDec/experiments/three_method_tuning_verifyfix/latest/locked_three_method_compare.yaml`
- tuning summary：`/home/chajiahao/data/specDec/experiments/three_method_tuning_verifyfix/latest/tuning_summary.csv`

还跑了一个 held-out sanity cell：

- 目录：`/home/chajiahao/data/specDec/experiments/three_method_locked_sanity_verifyfix/latest`
- cell：`request_count=1, draft_worker_count=1, max_new_tokens=8, network_profile=observe`
- 结果：`specedge_pipeline=true, sled_async=true, dip_sd=true`

所以当前状态是：可以开始正式 held-out 三方法性能矩阵；但还不能把 tuning 数字当最终性能结论。

## 一、论文 DiP-SD 的一次推理例子

假设有 4 个用户请求：

- 用户 A：问“解释 TCP slow start”
- 用户 B：问“写一个 Python 排序函数”
- 用户 C：问“总结一段客服对话”
- 用户 D：问“翻译一段英文”

边缘侧有 4 台用户设备，每台设备本地跑 draft model，例如 Qwen3-1.7B；边缘服务器跑 target model，例如 Qwen3-32B。DiP-SD 的核心不是单个请求怎么 draft，而是“多用户这一轮应该怎么分批、每个用户 draft 多长、服务器怎么流水验证”。

一次 speculative round 可以这样走：

1. 系统收集每个用户的状态。
   - A/B/C/D 都有自己的 prefix length `i_m`。
   - 每个用户有估计 acceptance rate `alpha_m`。
   - 每个用户有本地 draft latency 参数 `c_m, beta_m`。
   - 服务器有 verify latency 参数 `c_v, beta_v` 和显存上限。

2. DiP-SD solver 决定 batch 数 `N`。
   - 如果 `N=1`，所有用户一起 verify，batch 大但可能等慢设备。
   - 如果 `N=4`，每个用户单独 verify，等待少但服务器 batch 利用率低。
   - 论文算法扫描 `N=2..M`，比如当前选出 `N=2`。

3. solver 决定用户到 batch 的分配 `x_mn`。
   - 例子：batch 0 = A,C；batch 1 = B,D。
   - 这个分配考虑每个用户 draft 完成时间、通信延迟、prefix 长度、verify padding 后的成本。

4. solver 决定每个用户的 draft length `l_m`。
   - 例子：A draft 4 个 token，B draft 6 个，C draft 4 个，D draft 5 个。
   - draft 太短，target call 省得少；draft 太长，acceptance 下降且 verify padding 成本上升。
   - 论文用 expected accepted tokens `u_m(l_m)` 估计收益。

5. 用户设备并行本地 draft。
   - A 本地生成 `[a1,a2,a3,a4]`。
   - B 本地生成 `[b1..b6]`。
   - C 本地生成 `[c1..c4]`。
   - D 本地生成 `[d1..d5]`。
   - 这一步是 device-level distributed drafting。

6. 服务器按 pipeline stage 验证。
   - stage 0 验证 batch 0：A,C。
   - stage 1 验证 batch 1：B,D。
   - batch 内按最大 draft length 和最大 prefix length 做 padding；服务器一次 forward 验证多个用户的多个 draft token。

7. target 返回接受结果。
   - A 可能前 3 个 draft token 匹配，第 4 个不匹配，target 给出纠偏 token `a4_target`。
   - C 可能 4 个都匹配，target 额外给一个 bonus token。
   - 各用户把 accepted tokens 和 bonus/correction token 追加到 prefix。

8. 进入下一轮。
   - 新 prefix 改变，remaining token 改变。
   - DiP-SD 可以重新规划，也可以复用 cached plan。

论文 DiP-SD 的功能模块：

- 用户侧 draft 模块：每个设备独立生成 draft tokens，不在用户侧 batch。
- latency model 模块：用 affine latency 估计 draft/verify 时间。
- throughput objective 模块：最大化 expected accepted tokens / pipeline span。
- batch-count scan 模块：枚举 `N=2..M`。
- association subproblem 模块：固定 draft length，求用户到 batch 的分配。
- draft-length subproblem 模块：固定 batch 分配，求每个用户 draft 多长；论文用 Dinkelbach 处理分式目标。
- edge-server verify 模块：按 batch 验证 draft tokens。
- memory constraint 模块：确保 target model 参数和 KV cache 不超过服务器显存。

## 二、论文 SLED 的一次推理例子

假设有 3 台边缘设备：

- RPi 5 跑 1B draft model。
- Jetson Orin Nano 跑 3B draft model。
- 小型边缘 GPU 跑 7B draft model。

它们共享同一个 tokenizer，服务器上有一个更大的 target model。

一次推理可以这样走：

1. 每台设备本地接收 prompt 并 tokenize。
   - 设备 1 的用户问：“Summarize home security system status.”
   - 设备 2 的用户问：“Identify the upcoming traffic sign.”
   - 设备 3 的用户问：“Draft a short email.”

2. 设备上的 speculation controller 启动 dynamic drafting。
   - draft model 每生成一个 token，就看该 token 的 confidence score。
   - 如果 confidence 高于阈值 `c_th`，继续 draft 下一个 token。
   - 如果 confidence 低于阈值，停止本地扩展，请求服务器 verify。

3. 形成 verification request。
   - 设备 1 可能发送 3 个 draft token。
   - 设备 2 可能发送 5 个 draft token。
   - 设备 3 可能只发送 2 个 draft token。
   - SLED 接受异构 draft model，只要 tokenizer 一致。

4. 服务器侧 request queue 收集请求。
   - batch planner 等到固定 batch size，或者等到策略允许发 batch。
   - 不同请求 draft token 长度不同，服务器 padding 后统一进入 target model。

5. verification executor 批量验证。
   - target model 一次 forward 验证多个设备发来的 draft tokens。
   - 返回每个请求 accepted prefix、rejected position、correction token。

6. 设备等待 verify 时继续 async drafting。
   - 设备 1 发送 `[x1,x2,x3]` 后不空等，而是继续生成 `[x4,x5]`。
   - 如果服务器确认 `[x1,x2,x3]` 全接受，那么 `[x4,x5]` 可以进入下一轮 verify 队列。
   - 如果中间 token 被拒绝，`[x4,x5]` 就丢弃，因为 prefix 已经变了。

7. timeout/fallback。
   - 如果服务器太久不返回，设备可以合并最近产生的 draft tokens 重新请求。
   - 连续失败超过阈值时，论文提到可以把本地 draft token 作为 fallback 发给用户。

论文 SLED 的功能模块：

- edge device draft model：每台设备本地轻量 LLM。
- speculation controller：根据 confidence threshold 决定何时请求 verify。
- request queue：接收来自多设备的 verify 请求。
- batch planner：把多个请求组成 target batch。
- verification executor：服务器 target model 批量验证。
- async drafting 模块：verify in-flight 时设备继续 draft。
- timeout/fallback 模块：处理网络波动或服务器失败。
- system monitor：观察 GPU 利用率、VRAM、队列长度、平均延迟等。

## 三、论文 SpecEdge 的一次推理例子

假设有 2 个用户，每个用户旁边有一张 edge GPU；服务器有 A100 target model。

一次推理可以这样走：

1. 用户 A 的 edge GPU 做 initial tree drafting。
   - 当前 prefix 是 `P_A`。
   - draft model 生成一棵 tree，而不是一条线性 token 序列。
   - 例子：root 后可能有候选 token `t1/t2/t3`，每个候选下面继续扩展。

2. edge 把 draft tree 发给服务器验证。
   - 网络里传的是 token ids 和 tree structure，不传 hidden states。
   - 服务器用 target model 做 tree verification。

3. edge 不等服务器返回，继续 proactive draft tree expansion。
   - SpecEdge 论文强调：不是所有 leaf 都继续扩展。
   - 它选择 cumulative logprob 最高的一条 leaf path 作为 expansion head。
   - 例如最高路径是 `t2 -> t7 -> t9`，edge 就从这个位置继续 draft deeper subtree。

4. 服务器返回 accepted path 和 bonus token。
   - 如果 target 接受路径 `t2 -> t7 -> t9`，并且 bonus token 正好等于 proactive subtree 的第一个 token，这叫 complete draft alignment。
   - complete alignment 时，edge 保留 proactive subtree，下一轮直接从更深位置继续。
   - 如果不 alignment，proactive work 丢弃，从 target 确认的新 prefix 重新建树。

5. 多用户 pipeline-aware scheduling。
   - 用户 A 的 tree 在服务器 verify 时，用户 B 的 edge GPU 正在 draft。
   - 用户 B 的 tree 到达服务器时，A 的 verify 刚好完成。
   - 论文目标是让 `server verification time ≈ edge drafting time + network RTT`，减少 server bubble。

论文 SpecEdge 的功能模块：

- initial tree drafting：生成 tree-shaped candidate tokens。
- proactive edge drafting：verify in-flight 时继续生成。
- expansion head selector：选择最高累计 logprob 的 leaf path。
- post-verification update：根据 accepted path 和 bonus token 决定保留或丢弃 proactive subtree。
- edge KV cache update：把 accepted tokens 写入 edge draft model state。
- server tree verification：服务器只负责 target verify。
- pipeline-aware scheduler：交错多用户 verify，校准 draft depth 以匹配 verify time 和 RTT。
- heterogeneous batch verify：处理不同 prefix length/tree shape 的请求。

## 四、代码 SpecEdge `specedge_pipeline` 的一次推理例子

当前代码入口是 `/home/chajiahao/data/specDec/scripts/3090_specedge_smoke.py` 里的 `specedge_pipeline` 分支。它和论文 SpecEdge 相似，但不是完全等价复刻。

假设运行 2 个请求、2 个 3090 draft worker、A100 target service：

1. runner 创建 sessions。
   - 每个 prompt 变成 `GenerationSession`。
   - session 保存 `prompt_ids`、`prefix_ids`、`remaining_tokens`、`step_idx`。

2. draft registry 选择 tree-capable worker。
   - `draft_registry.runners_for("tree")` 返回 TopK tree draft runners。
   - 每个 worker 可能映射到一张 3090，模型可以是 Qwen3-0.6B 或 Qwen3-1.7B。

3. `AsyncPipelineRuntimeEngine` 开始 round 0。
   - planning policy 先给每个 active session 分配 draft job。
   - scheduler 产生 `draft_jobs` 和 `verify_batches`。

4. `SpecEdgeOfficialCandidateStrategy` 生成 tree proposal。
   - worker 对当前 prefix 做 `generate_tree_batch(...)`。
   - 生成 `CandidateProposal(shape="tree")`。
   - metadata 里带 `prefix_ids`、`tree_node_count`、`tree_max_depth`、`official_specedge_state`。

5. A100 HTTP tree verifier 验证。
   - `HttpTreeVerifierClient` 把 proposal 发到 A100。
   - A100 qwen3_graph backend 走 `tree_forward_batch` 或 tree attention。
   - 返回 accepted node ids、rejected node ids、bonus token、target choices。

6. verify in-flight 时 proactive tree draft 同时发生。
   - `SpecEdgeOfficialProactiveDraftPolicy` 或 `SpecEdgeProactiveDraftPolicy` 会从已有 tree 中选择扩展位置。
   - 生成 proactive proposal。

7. acceptance 写回。
   - `SpecEdgeTreeAcceptancePolicy` 根据 accepted node ids 找出对应 token。
   - 如果有 bonus token，也追加。
   - `session.append_tokens(...)` 写入最终输出。

8. official state/reconcile 处理复用。
   - `OfficialSpecEdgeDraftState` 保留每个 request 的 slot、tree nodes、node statuses、draft batch row。
   - 如果 proactive subtree 和新 prefix 对齐，下一轮可以复用。
   - 如果不对齐，丢弃并从新 prefix 重新 draft。

代码 SpecEdge 的主要模块：

- `scripts/3090_specedge_smoke.py`：方法入口、config 解析、artifact 写出。
- `src/specplatform/runtime/async_pipeline.py`：异步 verify + proactive draft runtime。
- `src/specplatform/methods/specedge_tree.py`：tree candidate、proactive、acceptance、reconcile。
- `src/specplatform/methods/specedge_official.py`：official-style persistent tree state。
- `src/specplatform/verification/tree.py`：tree verifier schema 到 target model 的转换。
- `src/specplatform/model/qwen3_graph.py`：A100/3090 qwen3 graph backend。

## 五、代码 SLED `sled_async` 的一次推理例子

当前代码入口是 `/home/chajiahao/data/specDec/scripts/3090_specedge_smoke.py` 里的 `sled_async` 分支。

假设 4 个请求、4 个 draft workers：

1. runner 创建 4 个 `GenerationSession`。
   - 每个 request 都有独立 prefix。
   - `max_new_tokens` 控制最多生成多少 token。

2. `SLEDPlanningPolicy` 分配 edge device。
   - 每个 request 固定绑定一个 draft worker。
   - 如果有 worker speed profile，会按相对速度做 weighted assignment。
   - planning metadata 记录 `request_worker_assignment` 和 estimated ready time。

3. `SLEDDynamicCandidateStrategy` 本地 dynamic draft。
   - 调用 `generate_tokens_until_confidence_drop(...)`。
   - token confidence 高就继续，低就停止。
   - 受 `sled.max_speculation_tokens` 和 `sled.confidence_threshold` 控制。

4. scheduler 组成 verify batch。
   - `sled.batch_size` 控制 batch 大小。
   - 如果开启 static queue，还会按 queue wait/padding 规则重新组织 verify batch。

5. `AsyncPipelineRuntimeEngine` 后台发起 verify。
   - `HttpLinearVerifierClient` 把 linear proposal 发给 A100。
   - A100 返回 accepted prefix length、verified tokens、bonus/correction token。

6. verify in-flight 时继续 proactive draft。
   - `SLEDAsyncDraftPolicy` 用 `prefix + parent draft tokens` 继续本地生成。
   - 这部分先不写入 session，只作为 possible next-round cache。

7. `GreedyPrefixAcceptancePolicy` 写回 target 对齐结果。
   - 如果前 3 个 token 被 target 接受，就写入前 3 个。
   - 如果第 4 个 mismatch，就写入 target correction token。
   - 如果全接受且 allow bonus，写入 bonus；当前 `sled_async` 配置里通常关闭 full-match bonus，以减少语义/调试复杂度。

8. `SLEDAsyncReconcilePolicy` 决定 proactive tokens 是否可复用。
   - 只有 parent proposal 全接受、没有 rejected token、没有 bonus 干扰，并且 proactive prefix 等于当前 committed prefix，才复用。
   - 否则丢弃 proactive tokens。

代码 SLED 的主要模块：

- `src/specplatform/methods/sled.py`：dynamic candidate、async proactive、reconcile、planning。
- `src/specplatform/runtime/async_pipeline.py`：后台 verify 和 proactive draft 的 overlap。
- `src/specplatform/schedulers/sled_queue.py`：SLED static queue 相关逻辑。
- `src/specplatform/verification/linear.py`：linear verify response 到 accepted prefix 的转换。
- `src/specplatform/methods/linear.py`：`GreedyPrefixAcceptancePolicy`。

## 六、代码 DiP-SD `dip_sd` 的一次推理例子

当前代码入口是 `/home/chajiahao/data/specDec/scripts/3090_specedge_smoke.py` 里的 `dip_sd` 分支。它比旧 handoff 中的实现更接近论文：有 paper-MILP/Dinkelbach solver、plan cache、latency calibration、steady-state prefetch。

假设 4 个请求、4 个 draft workers：

1. runner 创建 sessions，并选择 greedy draft workers。
   - `draft_registry.runners_for("greedy")` 返回线性 draft worker。
   - candidate strategy 是 `LinearCandidateStrategy(proposal_prefix="dip-sd-linear")`。

2. `DiPSDPlanningPolicy` 收集用户参数。
   - 每个 request 分配一个 worker。
   - 根据 worker metadata 估计 draft speed/quality。
   - 读取 `dip_sd_alpha`、`dip_sd_comm_latency_ms`、`dip_sd_draft_c/beta`、`dip_sd_verify_c/beta`。

3. `DiPSDSolver` 求解计划。
   - 扫描 batch count。
   - 对固定 batch count，求 user-to-batch assignment。
   - 再求每个 request 的 integer draft length。
   - 输出 `preferred_batches` 和 `draft_lengths`。

4. scheduler 把 PlanHints 变成可执行计划。
   - 例如 batch 0 = request A,C；batch 1 = request B,D。
   - A draft 4 token，B draft 8 token，C draft 4 token，D draft 8 token。

5. `DistributedBatchPipelineRuntimeEngine` 并行提交 draft jobs。
   - 每个 worker 本地 `generate_tokens(...)`。
   - draft 完成后形成 linear `CandidateProposal`。

6. runtime 按 stage 等待并 verify。
   - stage 0 等 A,C 的 draft 都 ready，然后发给 A100 linear verifier。
   - stage 1 等 B,D ready，再发下一批。
   - 如果服务器空等 draft，会记录 `pipeline.draft_ready_wait`。

7. A100 linear verifier 返回 causal-safe 结果。
   - 当前修复后，qwen3_graph linear verify 是 prefix-step verification。
   - 它逐 token 用当前 prefix 验证，避免旧 single-pass 语义跳 token。

8. acceptance 写回。
   - `GreedyPrefixAcceptancePolicy` 根据 accepted prefix + correction/bonus 产出 output tokens。
   - session append 后进入下一 round。

9. steady-state prefetch。
   - 如果本轮有 token 写入且 session 未结束，runtime 可以提前提交下一轮 draft。
   - 当前代码记录 prefetch 的 `prefix_ids` 和 `step_idx`，只有 prefix 完全匹配才复用，避免 stale draft。

代码 DiP-SD 的主要模块：

- `src/specplatform/methods/dip_sd/model.py`：论文 cost model、expected accepted tokens、memory/latency 公式。
- `src/specplatform/methods/dip_sd/solver.py`：batch count scan、MILP/Dinkelbach/enumeration solver。
- `src/specplatform/methods/dip_sd/policy.py`：把 solver 结果转换为 runtime `PlanHints`。
- `src/specplatform/runtime/distributed_pipeline.py`：stage-level distributed draft + central batch verify。
- `src/specplatform/methods/dip_sd/calibration.py`：从真实 phase events 拟合 latency 参数。
- `src/specplatform/verification/linear.py`：linear verifier 和 accepted prefix 语义。

## 七、现有代码方法的不足

SpecEdge 代码不足：

- proactive tree reuse 依赖 prefix/tree state 对齐，失败时会丢弃较多 proactive work。
- 当前实现更偏工程复现，不完全等价论文里的全部 scheduling/calibration 策略。
- tree draft 和 qwen3_graph state 复杂，worker affinity、draft batch row、KV gather/reorder 都是易错点。
- 小 `max_new_tokens` 场景下，draft/proactive/scheduling overhead 可能大于省下的 target calls。

SLED 代码不足：

- 当前 `sled_async` 是 lossless greedy-prefix 验证版本，和论文里带 probabilistic acceptance/fallback 的完整边缘服务语义不完全一样。
- dynamic confidence threshold 的好坏强依赖 draft model calibration。
- async proactive 只有在 prefix 完全对齐时才复用，mismatch 时工作浪费。
- static queue/padding 可以提高 batch，但会增加等待时间，短输出时很容易拖慢。

DiP-SD 代码不足：

- paper-MILP/Dinkelbach solver 进入在线 timed path 时，短输出负载下 solver overhead 会非常重。
- latency model 需要真实 calibration；不校准时 planner 可能选出理论上好、实际很慢的 plan。
- prefix-step linear verifier 修复了正确性，但比旧 single-pass 更慢，性能数字必须重跑。
- plan cache 只能缓解相同 request shape，复杂动态负载下缓存命中有限。
- stage pipeline 可能出现 server idle gap 或 draft_ready_wait。

平台共同不足：

- 目前主要是单 A100 target service，target 仍可能成为瓶颈。
- 网络 profile 多是 observe/modeled，不等价真实公网长期波动。
- tuning 只是在有限网格内找较优配置，不代表全局最优。
- correctness gate 是对齐 `target_only` greedy 输出，不能证明所有采样式 speculative decoding 分布完全一致。

## 八、论文方法本身的不足

DiP-SD 论文不足：

- 依赖 acceptance 独立同分布/几何模型，真实 LLM token acceptance 往往和上下文、任务、draft model 强相关。
- 依赖 affine latency profiling，换硬件、batch shape、CUDA graph、KV cache 后都要重拟合。
- MILP/Dinkelbach 求解开销在论文数值模拟中不一定进入在线 hot path。
- 论文实验偏数值仿真，真实系统中的 HTTP、queue、warmup、kernel cache、线程争用会改变结论。

SLED 论文不足：

- 更像系统 position paper，调度/优化问题没有 DiP-SD 那样形式化。
- static batching 简单但可能牺牲 tail latency。
- fallback release 本地 draft tokens 可能破坏严格 lossless 目标，除非只作为用户体验降级策略。
- 对 tokenizer 一致、设备可信、网络稳定性、安全隔离的工程要求没有完全展开。

SpecEdge 论文不足：

- 高收益依赖 draft-target alignment；draft model 太弱时 proactive subtree 容易浪费。
- 需要 edge GPU 参与且足够可信，真实部署会遇到计费、调度、安全、失败恢复问题。
- RTT 过高时收益下降，论文也显示需要按 RTT 调 draft depth。
- 成本效率受云厂商价格、GPU 可用性、模型大小和业务负载影响很大。
