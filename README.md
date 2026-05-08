# WISE-PACE

WISE-PACE 是一个面向 LLM 自动作文评分（Automated Essay Scoring, AES）的研究原型。项目从 WISE-AES 出发，把原本主要依赖 raw validation QWK 的 prompt / anchor 进化，推进为 hidden-evidence diagnostic guided protocol evolution。

当前研究目标不是训练一个 test-time 后处理校准器，而是寻找更好的评分协议：

```text
P* = <I*, E*>

I: scoring instruction / evolved rubric
E: reference anchors / static exemplars
M: local target LLM, also used for hidden representation extraction
```

最终成功指标始终是目标 LLM 在 `<I*, E*>` 下直接输出的 raw score。PACE 只作为 protocol evolution 过程中的 hidden-evidence diagnostic / mutation guidance，不作为最终 test score。

## 当前状态

主线已经推进到 Phase 4：

- 主实验路径使用本地 Llama backend，不再依赖 OpenRouter。
- `M_score = M_hidden = M_meta = local Llama`。
- `final_pace_calibrated` 默认保持 `false`。
- selection 是 raw-first guarded selection。
- PACE calibrated QWK 不作为主 selection reward，不作为最终 test 结果。
- high-score / max-score audit、parent-child mutation audit、training curve、candidate comparison 已经落盘。
- 新增 neural hidden evidence、dual-view anchor encoding、diagnostic-only PACE、staged validation、PACE-on-demand 和 protocol cache 基础设施。

## 方法逻辑链

WISE-PACE 当前按以下逻辑工作：

```text
1. 构造候选评分协议 P = <I, E>
2. 用本地 LLM M 在 P 下直接给 validation essays 打 raw score
3. 计算 raw validation QWK、MAE、score distribution、high/max recall
4. 对 top / triggered candidates 抽取 scoring-context hidden states
5. 用 hidden-anchor geometry 和 raw error pattern 生成 PACE diagnostics
6. 根据 dominant ErrorType 选择 mutation policy
7. 对 I 或 E 做受控 mutation
8. 用 validation-only signals 选择下一代 elite
9. 最终 test 前固定 primary candidate
10. 在 test set 上只报告 raw LLM scoring 结果
```

核心原则：

```text
PACE 可以指导 protocol evolution。
PACE 不能替代 raw scoring。
```

这样可以避免方法退化成“WISE-AES 后面接一个强 calibrator”。我们真正想证明的是：hidden evidence 能帮助找到更好的 `I*` 和 `E*`，从而让目标 LLM 本身在该协议下打出更合理的分数。

## Scoring-Context Encoding

为了让 target essay 和 anchor essay 的 hidden representations 可比较，当前使用统一的 scoring-context encoding。

Target essay：

```text
Scoring Instruction:
<I>

Score Range:
<score_min>-<score_max>

Reference Anchors:
Score: <anchor_score>
Essay: <anchor_essay>

Essay to Represent:
<x>

Representation Target:
Encode the essay to be scored.
```

Anchor essay 有两种 view：

- text-only view：不输入 known score，用于表示 anchor 文本本身。
- score-conditioned view：输入 known score，用于表示 score-conditioned reference anchor。

neural evidence 支持：

```text
anchor_view: none / text / score / dual
```

dual view 同时使用 text-only anchor hidden 和 score-conditioned anchor hidden。

## Evidence 设计

当前支持两种 evidence mode：

```text
compact_manual
neural_hidden
```

### compact_manual

保留早期 compact evidence，包含 raw score、anchor-relative hidden geometry、reasoning text、objective / uncertainty / stability 等手工特征块。该模式主要用于兼容旧实验和 z-block ablation。

### neural_hidden

主线新增 `pace/neural_evidence.py`：

```text
target_hidden
anchor_hidden_text
anchor_hidden_score
anchor_scores
reasoning_hidden
y_raw embedding
```

这些输入经过 frozen neural projection、target-to-anchor attention、block-preserving heads，输出固定维度的 `z_neural`。neural_hidden 模式不把 word count、keyword count、hedge count 等手工特征作为模型输入。

当前 neural evidence 仍属于轻量诊断模块，不是最终评分器。它的主要用途是帮助 mutation routing 和 hidden-error attribution。

## Mutation Policy

PACE diagnostics 会映射到 mutation type：

```text
under_score_high_hidden      -> high_tail_instruction_mutation
over_score_low_hidden        -> negative_constraint_mutation
anchor_confusion             -> anchor_slot_mutation
reasoning_score_contradiction -> score_mapping_mutation
raw_collapse                 -> score_distribution_mutation
boundary_ambiguity           -> boundary_clarification_mutation
true max underprediction     -> max_score_contrastive_mutation
general_error                -> general_reflection_mutation
```

当前启用 mutation diversity quota，避免所有 child 都变成 high-tail mutation。parent-child audit 会记录：

- parent / child signature
- parent / child raw QWK
- parent / child high/max recall
- mutation_type / requested_mutation_type / actual_mutation_type
- instruction_changed / anchors_changed
- dominant_error_type_used
- validation_stage 和 delta_comparable

## 成本控制

当前成本优化包括：

- `llm.max_new_tokens_scoring`
- `llm.max_new_tokens_reflection`
- `llm.max_new_tokens_induction`
- staged validation：mini / mid / full validation
- PACE-on-demand trigger
- protocol score cache
- PACE result cache
- diagnostic-only PACE，可跳过 calibrator

关键配置：

```yaml
pace:
  selection_influence: "diagnostic_only"
  diagnostic_only_skip_calibrator: true
  diagnostic_sample_size: 32
  final_pace_calibrated: false
```

`diagnostic_only_skip_calibrator: true` 时，PACE 只计算 raw hidden-error diagnostics、anchor geometry 和 ErrorType，不训练 Coral calibrator，也不计算 calibrated test result。

## 最新实验记录

### P1 fold0 cost-aware gate

Raw-only cost-aware gate：

```text
exp_dir: logs/exp_20260508_152214_fold0
validation QWK: 0.2405
raw test QWK: 0.3207
high recall: 0.4444
max recall: 1.0000
runtime: 50.4 min
tokens: 1.11M
```

WISE-PACE diagnostic-only gate：

```text
exp_dir: logs/exp_20260508_174135_fold0
validation QWK: 0.3684
raw test QWK: 0.3565
raw test MAE: 1.1875
high recall: 0.6111
max recall: 0.0000
runtime: 66.2 min
tokens all: 2.05M
PACE tokens: 0.89M
```

主要观察：

- WISE-PACE gate 在 val QWK、test QWK、MAE、high recall 上均好于 raw-only gate。
- 有效 child 来自 `boundary_clarification_mutation`，full-val raw QWK 相比 parent 提升约 `+0.128`。
- max-score recall 仍然不稳定；test 中 true max 样本很少，不能过度解读。
- calibrator probe 不稳定且昂贵，因此后续主线改为 diagnostic-only skip calibrator。

### neural diagnostic smoke

```text
exp_dir: logs/exp_20260508_190507_fold0
config: configs/phase4_neural_smoke.yaml
runtime: 7.7 min
validation QWK: 0.2411
raw test QWK: 0.2771
PACE tokens: 84k
diagnostic_only_skip_calibrator: true
```

该 smoke 的目的不是追求分数，而是验证 neural hidden diagnostic 路径能在真实本地 Llama 上端到端跑通。

## 当前风险

当前模型还不能直接认为已经达到 EMNLP main 级别的定型状态。主要风险：

- high-score / max-score recall 仍是最大短板。
- base 8B Llama 经常把分数压到 7/10 附近，存在 score range compression。
- PACE calibrator probe 不稳定，已经从主路径降级为可选诊断。
- neural evidence 仍需更多 ablation 证明 anchor-relative hidden evidence 的贡献。
- 需要 raw-only、random mutation、no-anchor、anchor-only、OPRO-lite / APE-lite 等 baseline。
- 需要 P1-P8 smoke 和更强模型对照，不能只凭 P1 fold0 下结论。

## 推荐运行命令

运行测试：

```bash
pytest tests/test_phase4_diagnostics.py tests/test_neural_evidence.py tests/test_prompt_profiler.py -q
```

raw-only cost-aware gate：

```bash
python wise_aes.py --config configs/phase4_raw_only_cost_aware_mid.yaml --fold 0
```

WISE-PACE cost-aware gate：

```bash
python wise_aes.py --config configs/phase4_cost_aware_mid.yaml --fold 0
```

neural hidden smoke：

```bash
python wise_aes.py --config configs/phase4_neural_smoke.yaml --fold 0
```

P1-P8 smoke dry-run：

```bash
python scripts/run_phase4_smoke_all_prompts.py --prompts 1 2 3 4 5 6 7 8 --fold 0 --dry-run
```

分析单个实验：

```bash
python scripts/analysis/summarize_phase4_run.py logs/<exp_dir>
python scripts/analysis/export_candidate_table.py logs/<exp_dir>
```

对比 raw-only 和 WISE-PACE：

```bash
python scripts/analysis/compare_raw_vs_wisepace.py \
  --raw-only-exp logs/<raw_exp_dir> \
  --wisepace-exp logs/<wisepace_exp_dir>
```

Prompt profile：

```bash
python scripts/analysis/profile_asap_prompt.py --prompt 1 --fold 0 --config configs/phase4_smoke.yaml
```

## 下一步最小实验建议

不要马上跑 5-fold。建议先跑：

1. `phase4_cost_aware_mid.yaml` 新 no-calibrator 版，对比 `logs/exp_20260508_174135_fold0`，确认成本下降且 mutation 信号不丢。
2. P1-P8 smoke 实跑，每个 prompt 只跑小规模，检查 score range 和 high-tail 风险是否泛化。
3. raw-only vs WISE-PACE cost-aware gate 再做 2-3 个 seed，确认 `boundary_clarification_mutation` 的提升不是单次偶然。

只有当这些最小实验稳定后，再启动单个完整 full fold。
