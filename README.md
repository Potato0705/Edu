# Edu / WISE-AES

当前仓库已经从早期的 **WISE-PACE hidden-evidence protocol evolution** 收敛到新的研究主线：

```text
Representation-Guided Budgeted Anchor Protocol Repair for LLM-Based Essay Scoring
简称：BAPR
```

核心问题不再是“anchors 是否有用”。已有工作 *Anchor Is the Key* 已经说明 full anchors 对 LLM-based AES 很强。我们现在研究的是：

```text
If anchors are the key, which anchors matter?
```

也就是：在固定本地 LLM、固定评分任务、有限 anchor budget `k` 下，如何选择和修复 anchor bank，使同一个 LLM 在最终测试时直接输出更好的 raw score。

## 当前主线

BAPR 的研究对象是一个评分协议：

```text
P = (I, A)

I: scoring instruction / rubric
A: anchor bank
M: fixed local LLM
```

最终预测始终是：

```text
y_hat = M_score(x | I, A)
```

约束非常重要：

- 不训练 final scorer。
- 不做 test-time calibration。
- 不把 hidden evidence 当最终评分器。
- 不使用 test labels 选择 anchors、repair anchors 或调 parser。
- `final_pace_calibrated=false`。
- representation / diagnostics 只允许用于 validation-time anchor selection、repair 和 audit。
- final test 只报告 selected anchor protocol 下的 raw LLM score。

## 为什么切换主线

早期 WISE-PACE 尝试过：

- PACE hidden diagnostic
- hidden-guided mutation priority
- contrastive anchors
- neural hidden evidence
- instruction / anchor protocol evolution

但 Gate 3 rescue 显示：hidden-guided mutation priority 虽然真实改变了决策路径，但没有带来稳定收益；contrastive anchors 也不稳定。因此当前结论是：

- hidden evidence 保留为 audit / ablation。
- contrastive anchors 暂停主线，最多作为 negative ablation。
- PACE calibrator 不进入主线。
- 主线转向 **coverage-constrained / retrieval-grounded / representation-aware anchor selection and repair**。

## BAPR v1 逻辑链

BAPR v1 是一个最小、可审计的一步 anchor-bank repair 框架：

```text
1. 固定 instruction I 和本地 LLM M
2. 将 validation split 成 V_diag 和 V_sel
3. 只用 train + V_diag 初始化 parent anchor bank A0
4. 用 A0 在 V_diag 上评分，得到 boundary failure profile
5. 根据 failure profile 生成最多 3 个 child anchor banks
6. 在 V_sel 上评估 parent 和 children
7. 用 guarded selection 选择最终 A*
8. 在 test 上只用 A* 做 raw LLM scoring
```

数据边界：

```text
train: anchor candidate pool
V_diag: 诊断 failure profile，允许影响 A0 和 child generation
V_sel: 只用于 guarded selection，不允许影响 A0 或 child generation
test: 只用于最终评估
```

## Repair Operators

BAPR v1 当前支持 4 类 repair operator：

```text
REBALANCE_SCORE_BANDS
REPLACE_WORST_BAND_ANCHOR
DECOMPRESS_EXTREME_ANCHORS
REMOVE_CONFUSING_OR_REDUNDANT_ANCHOR
```

每个 child 必须改善和其 operator trigger 相关的 boundary metric，不能靠无关指标偶然变好进入最终选择。

例如：

- `DECOMPRESS_EXTREME_ANCHORS` 需要改善 high recall、max recall、range coverage、SCI 或 tail under-score 等相关指标。
- `REBALANCE_SCORE_BANDS` 需要改善 coverage、score TV 或 worst-band MAE。
- empty band metric 记为 `None`，不会触发 repair severity。

如果所有 children 都被拒绝，BAPR 会返回 parent A0，并仍然写出完整输出文件。

## Guarded Selection

child 需要同时满足：

- 不能超过允许的 QWK drop。
- 不能超过允许的 MAE increase。
- anchor band coverage 不能退化。
- anchor score range 不能明显退化。
- 必须改善 operator 相关 boundary metric。

默认配置见：

[configs/anchor_budget_bapr_v1.yaml](configs/anchor_budget_bapr_v1.yaml)

## 关键输出文件

每个 BAPR run 会写出：

```text
anchor_bank.json
anchor_metrics.json
anchor_selection_trace.jsonl
predictions.csv
prediction_distribution.csv
score_boundary_metrics.json
summary.md

bapr_val_split.json
bapr_parent_anchor_bank.json
bapr_parent_metrics.json
bapr_failure_profile.json
bapr_repair_candidates.jsonl
bapr_guarded_selection.csv
bapr_final_anchor_bank.json
bapr_repair_trace.jsonl
```

其中 `bapr_parent_anchor_bank.json` 和 `bapr_final_anchor_bank.json` 用来明确记录 parent A0 与最终 A* 是否不同。

## 当前代码入口

核心实现：

- [scripts/bapr_repair.py](scripts/bapr_repair.py)
  BAPR 纯逻辑模块。不得调用 LLM，不实例化 `LocalLlamaBackend`。

- [scripts/run_anchor_budget_experiment.py](scripts/run_anchor_budget_experiment.py)
  anchor-budget / BAPR 实验入口。BAPR 只通过 clean integration function 接入，不把完整 repair pipeline 塞进 `build_anchor_bank()`。

- [tests/test_bapr_repair.py](tests/test_bapr_repair.py)
  BAPR toy/mock 单元测试，不调用真实 LLM。

## 本地验证

运行全部测试：

```bash
pytest -q
```

只跑 BAPR 测试：

```bash
pytest tests/test_bapr_repair.py -q
```

运行 fake-scoring smoke，不加载本地 LLM：

```bash
python scripts/run_anchor_budget_experiment.py \
  --config configs/anchor_budget_bapr_v1.yaml \
  --fold 0 \
  --methods bapr_repair_k_anchor \
  --ks 9 \
  --fake-scoring \
  --output-root logs/bapr_fake_smoke \
  --no-resume-existing
```

fake smoke 的目的只是验证 split、repair、guard、输出文件和 summary 链路，不代表真实模型分数。

## 真实 debug 实验

P1 fold0 BAPR v1 debug：

```bash
WISE_PACE_COMMIT=$(git rev-parse HEAD) \
PYTHONUNBUFFERED=1 python scripts/run_anchor_budget_experiment.py \
  --config configs/anchor_budget_bapr_v1.yaml \
  --fold 0 \
  --methods bapr_repair_k_anchor \
  --ks 9 \
  --output-root logs/anchor_budget_bapr/bapr_v1_p1_fold0_real \
  --no-resume-existing
```

当前不建议直接跑 full-fold。先看 P1 fold0 debug 是否说明：

- child repair 是否真实产生；
- guarded selection 是否合理；
- A* 是否不同于 A0；
- A* 是否改善 V_sel 上的目标 boundary metrics；
- raw test QWK / MAE / high recall 是否没有明显崩溃；
- token cost 是否可控。

## Anchor-Budget 历史结论

当前已经得到的研究判断：

1. P8 是 ultra-wide score range stress-test。10-60 分制下，当前 compact anchor protocol 不稳定，`no_anchor` 反而强于多种 anchor 方法。P8 暂时作为 limitation，不进入主线 full-fold。
2. P1/P2/P7 上，coverage-constrained representation-guided anchor selection 有过稳定信号，但 seed/split sensitivity 明显。
3. `retrieval_grounded_stratified_rep_k_anchor_v21` 更稳，但 representation replacement 经常过少，独立贡献仍不足。
4. `retrieval_grounded_no_rep_k_anchor` ablation 用来判断收益是否主要来自 coverage-constrained retrieval grounding。
5. 因此当前 BAPR v1 的重点不是继续堆 representation，而是验证 anchor-bank repair 是否能在 validation-only 边界约束下稳定改善 anchor protocol。

## 不应该做的事

当前阶段不要：

- 跑 5-fold。
- 跑 P1-P8 full。
- 为 P8 继续堆 prompt/anchor 补丁。
- 恢复 hidden-evidence mutation 主线。
- 把 PACE calibrated score 当 final result。
- 用 test set 修改 anchor selection、repair、parser 或 guard。
- 在没有 fake smoke 和单元测试通过前启动 GPU 实验。

## 下一步最小实验

建议顺序：

1. BAPR v1 P1 fold0 real debug：确认 repair chain 是否真实运行。
2. 若 P1 debug 有正信号，再做 same-split baseline 对照：`parent A0` vs `BAPR A*` vs `retrieval_grounded_no_rep`。
3. 若 repair chain 和 baseline 对照都成立，再考虑 P1/P2/P7 小规模 sanity；仍不直接进入 full-fold。

只有当 BAPR 能证明：

```text
failure profile
→ repair operator
→ child anchor bank
→ V_sel boundary improvement
→ final raw test not collapsing
```

才有资格进入更大规模实验。
