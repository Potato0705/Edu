# Edu / WISE-AES

当前仓库的研究主线已经从早期的 WISE-PACE hidden-evidence protocol evolution，收敛到：

```text
Which Anchors Matter?
Stability- and Influence-Aware Anchor Protocol Learning for LLM-Based Essay Scoring
```

当前实现名：

```text
BAPR-SI: Budgeted Anchor Protocol Repair with Stability and Influence
```

核心问题不是“anchors 是否有用”。已有工作 *Anchor Is the Key* 已经说明 full anchors 对 LLM-based AES 很强。我们现在研究的是：

```text
If anchors are the key, which anchors matter, and why?
```

在固定本地 LLM、固定 scoring instruction、有限 anchor budget `k` 下，BAPR-SI 只在 validation-time 使用 anchor-level stability / influence 估计来选择和修复 anchor bank。最终 test 阶段仍然只让同一个 LLM 在选出的 anchor protocol 下直接输出 raw score。

## 研究边界

评分协议：

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

严格约束：

- 不训练 final scorer。
- 不做 test-time calibration。
- 不把 hidden evidence 当最终评分器。
- 不使用 test labels 选择 anchors、repair anchors、调 parser 或调 guard。
- `final_pace_calibrated=false`。
- stability / influence / diagnostics 只允许用于 validation-time anchor selection、repair 和 audit。
- final test 只报告 selected anchor protocol 下的 raw LLM score。

## 为什么切换到 BAPR-SI

早期 WISE-PACE 尝试过 hidden diagnostic、hidden-guided mutation、contrastive anchors、neural evidence 和 instruction evolution。实验显示这些机制目前没有稳定独立贡献，且容易让论文主线变得混乱。

BAPR v1 已经证明了一步 anchor-bank repair 的机制链条可以跑通，但其技术贡献仍偏工程化。BAPR-SI 将主贡献上移到两个可审计估计器：

```text
1. Anchor Stability Estimator
2. Anchor Influence Estimator
```

它们回答两个更清晰的问题：

- limited validation labels 下，anchor selection 为什么会不稳定？
- 哪些 anchors 是稳定有用的，哪些 anchors 会伤害 score-boundary grounding？

## BAPR-SI 方法链

```text
1. 从 train pool 中构造 candidate anchors。
2. 将 validation 分成 V_diag 和 V_sel。
3. 只用 train + V_diag 估计 anchor stability。
4. 用 stability-aware selector 构造 parent anchor bank A0。
5. 用 A0 在 V_diag 上打分，得到 boundary failure profile。
6. 估计 parent anchors 的 proxy influence。
7. 只替换 negative-influence anchor，加入 high-stability candidate。
8. 在 V_sel 上用 guarded selection 选择 A*。
9. 在 test 上只用 A* 做 raw LLM scoring。
```

数据边界：

```text
train: anchor candidate pool
V_diag: stability / influence / failure diagnosis / child generation
V_sel: guarded selection only
test: final raw-score evaluation only
```

## Technical Contributions

### Anchor Stability Estimator

对 `V_diag` 做 bootstrap subsplits，重复运行同一 retrieval / coverage selector，记录每个 anchor 的：

```text
selection_frequency
mean_rank
rank_variance
redundancy_score
stability_score
```

直觉：一个 anchor 如果只在某一次 validation sample 上偶然被选中，它不应该成为稳定 parent bank 的核心。

### Anchor Influence Estimator

第一版实现了纯逻辑 proxy influence，不调用 LLM。它结合：

```text
anchor stability
same-band overrepresentation
text redundancy
failure profile pressure
score-band / high-tail / compression diagnostics
```

输出 anchor failure type：

```text
harmful_compression_anchor
high_tail_suppressor
boundary_confuser
redundant_low_influence_anchor
useful_stabilizing_anchor
```

后续可扩展为 leave-one-anchor-out influence，但 LOO scoring 应只由 runner 调用，`scripts/anchor_influence.py` 本身仍保持纯逻辑。

### Stability-Influence Repair

BAPR-SI 只允许替换 negative-influence anchor：

```text
remove harmful / unstable anchor
add high-stability candidate from matching score band
guard on V_sel
```

child 必须通过 guard：

- QWK drop 不超过容忍度。
- MAE increase 不超过容忍度。
- anchor coverage 不退化。
- anchor stability 不明显退化。
- 必须改善与 trigger 相关的 boundary / anchor metric。

## 当前入口

核心模块：

- `scripts/anchor_stability.py`  
  纯逻辑 stability estimator，不调用 LLM。

- `scripts/anchor_influence.py`  
  纯逻辑 influence estimator 和 influence repair child generator，不调用 LLM。

- `scripts/bapr_repair.py`  
  BAPR guard、failure profile、legacy repair operators。

- `scripts/run_anchor_budget_experiment.py`  
  统一实验入口，负责打分、split、输出文件和 BAPR-SI 集成。

新增方法名：

```text
stability_retrieval_k_anchor
bapr_si_k_anchor
```

默认配置：

```text
configs/anchor_budget_bapr_si.yaml
```

## 关键输出文件

普通 anchor run：

```text
anchor_bank.json
anchor_metrics.json
anchor_selection_trace.jsonl
predictions.csv
prediction_distribution.csv
score_boundary_metrics.json
summary.md
```

BAPR-SI run 额外输出：

```text
bapr_val_split.json
bapr_parent_anchor_bank.json
bapr_parent_metrics.json
bapr_failure_profile.json
bapr_repair_candidates.jsonl
bapr_guarded_selection.csv
bapr_final_anchor_bank.json
bapr_repair_trace.jsonl

anchor_stability_scores.csv
stability_trace.jsonl
anchor_influence_scores.csv
anchor_influence_trace.jsonl
anchor_loo_influence_scores.csv
anchor_influence_child_alignment.csv
bapr_si_repair_trace.jsonl
```

## 本地验证

运行全部测试：

```bash
pytest -q
```

只跑 anchor / BAPR 相关测试：

```bash
pytest tests/test_anchor_budget.py tests/test_bapr_repair.py -q
```

BAPR-SI fake-scoring smoke，不加载本地 LLM：

```bash
python scripts/run_anchor_budget_experiment.py \
  --config configs/anchor_budget_bapr_si.yaml \
  --fold 0 \
  --methods bapr_si_k_anchor \
  --ks 9 \
  --fake-scoring \
  --output-root logs/bapr_si_fake_smoke \
  --no-resume-existing
```

fake smoke 只用于验证 split、stability、influence、repair、guard、输出文件和 summary 链路，不代表真实模型分数。

## 真实 debug 实验

P1 fold0 BAPR-SI debug：

```bash
WISE_PACE_COMMIT=$(git rev-parse HEAD) \
PYTHONUNBUFFERED=1 python scripts/run_anchor_budget_experiment.py \
  --config configs/anchor_budget_bapr_si.yaml \
  --fold 0 \
  --methods bapr_si_k_anchor \
  --ks 9 \
  --output-root logs/anchor_budget_bapr_si/bapr_si_p1_fold0_real \
  --no-resume-existing
```

当前不建议直接跑 full-fold。先验证：

- stability estimator 是否降低 anchor selection variance。
- influence estimator 是否能定位 harmful anchors。
- BAPR-SI A* 是否优于 A0。
- BAPR-SI 是否能接近或超过 retrieval same-k baseline。
- V_sel 改善是否转化到 raw test，而不是只在 V_diag 上过拟合。

## 暂不推进的方向

当前阶段不要做：

- 5-fold。
- P1-P8 full。
- P8 主线实验。
- hidden-evidence mutation 主线。
- contrastive anchors 主线。
- PACE calibrated score final result。
- instruction mutation。
- trained scorer。
- test-time calibration。
- 用 test set 修改 anchor selection、repair、parser 或 guard。

## 下一步最小实验

推荐顺序：

1. BAPR-SI P1 fold0 fake-scoring smoke，确认输出完整。
2. BAPR-SI P1 fold0 real debug，对比 A0 vs A*。
3. 同 split baseline attribution：`retrieval_k_anchor`、`stability_retrieval_k_anchor`、`BAPR-SI-A0`、`BAPR-SI-A*`、`full_static_anchor`。

只有当下面链条成立，才考虑 P1/P2/P7 小规模 sanity：

```text
anchor stability
-> stable parent bank
-> anchor influence estimate
-> targeted child repair
-> V_sel boundary improvement
-> raw test does not collapse
```
