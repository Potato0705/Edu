# WISE-PACE

WISE-PACE 是一个面向 LLM 自动作文评分（Automated Essay Scoring, AES）的研究原型。
它在 WISE-AES 的基础上，把原来主要依赖 raw validation score 的 prompt / anchor 进化，
扩展为 hidden-evidence-guided 的评分协议进化。

当前项目目标是：

```text
寻找最优评分协议 P* = <I*, E*>

I: scoring instruction / evolved rubric，即评分指令和进化后的评分规则
E: reference anchors / static exemplars，即参考 anchor 作文集合
M: local target LLM，同时用于直接评分和 hidden representation 提取
```

最终成功指标仍然是目标 LLM 在进化得到的 `<I*, E*>` 下直接输出的 raw score。
PACE 只作为协议选择和协议进化信号，不作为最终测试阶段的后处理校准器。

## 当前进度

当前实现状态：

- 主实验已经切换到本地模型后端，不再依赖 OpenRouter。
- scoring、reflection、rewrite、induction、hidden representation extraction 和 PACE fitness 都通过本地 Llama backend 执行。
- PACE 在进化过程中用于评估候选评分协议，信号包括 hidden evidence、anchor geometry、stability 和 cost。
- final PACE-calibrated inference 默认关闭，因为当前研究目标是让 `<I*, E*>` 下的 LLM raw scoring 变强，而不是训练一个更强的后处理 calibrator。
- anchor selection 和 anchor mutation 已改为 score-band based，设计目标是泛化到 ASAP P1-P8，而不是只针对 P1。
- 每次实验会保存 training curve、generation snapshot、anchor mutation log 和 final candidate comparison，便于后续诊断。

最新单个完整 fold 实验：

```text
Config: configs/phase4_full_fold.yaml
Prompt / fold: ASAP P1 fold 0
Model: Meta-Llama-3.1-8B-Instruct local bf16
Runtime: 865.6 minutes on RTX 5090 32GB
Validation primary QWK: 0.5219
Raw test QWK: 0.4780
Raw test MAE: 1.3445
Final PACE tokens: 0
```

这次实验的主要诊断：

- 最好的 raw validation 协议在第 1 代已经出现，并在后续代中保留下来。
- PACE / protocol quality 后续有小幅提升，但 raw validation QWK 没有继续上升。
- 高分段预测仍然是最大短板。P1 test 中真实 12 分作文有 14 篇，模型只预测出 4 篇。
- 当前方法已经可以端到端运行，但在高分段处理和 mutation 有效性进一步改善之前，还不适合直接投入昂贵的完整 multi-fold 实验。

## 当前模型的逻辑链

WISE-PACE 当前按下面的逻辑工作：

```text
1. 构造一个候选评分协议 P = <I, E>
2. 使用本地 LLM M 在协议 P 下给 validation essays 直接打分
3. 计算 raw validation QWK 和 raw score distribution diagnostics
4. 在同一个 scoring context 下提取 LLM hidden states
5. 构造证据向量 z，包含 raw score、hidden geometry、anchor relation、uncertainty 和 stability 信号
6. 在 calibration items 上训练一个轻量 PACE probe
7. 在 held-out fitness items 上评估 top candidate protocols
8. 综合 raw QWK、guarded PACE signal、anchor geometry 和 protocol-quality guards
9. 选择 elite protocols
10. 基于错误样本和 hidden-evidence diagnostics 变异 I 和 E
11. 重复若干 generations
12. 在测试集评估前，只用 validation-only signals 选择候选协议
13. 在 test set 上使用 raw LLM scoring 评估最终候选协议
```

核心设计原则是：

```text
PACE 可以指导 protocol evolution。
PACE 不能成为最终 scoring model。
```

这个原则是为了避免方法退化成“WISE-AES 后面接一个强 calibrator”。
我们真正想证明的是：hidden evidence 能帮助我们找到更好的 `I*` 和 `E*`，
从而让目标 LLM 本身在该协议下直接打出更合理的分数。

## 评分协议定义

一个评分协议定义为：

```text
P = <I, E>
```

其中：

- `I` 是评分指令，包括 evolved rubric、score range contract 和 operational calibration rules。
- `E` 是固定数量的 reference anchors，每个 anchor 包含一篇作文和它的已知人工分数。

raw scoring function 是：

```text
y_raw = M(x | I, E)
```

目标优化问题是：

```text
P* = argmax_P QWK(y_raw(P), y)
```

PACE 改变的是搜索过程中如何选择和变异候选协议。
它不替代 `y_raw` 成为最终答案。

## Scoring-Context Encoding

为了让 target essay 和 anchor essay 的 hidden representations 可比较，
当前使用统一的 scoring-context encoding 模板。

对 target essay：

```text
Scoring Instruction:
<I>

Score Range:
<score_min>-<score_max>

Reference Anchors:
Score: <low_score>
Essay: <e_low>

Score: <mid_score>
Essay: <e_mid>

Score: <high_score>
Essay: <e_high>

Essay to Represent:
<x>

Representation Target:
Encode the essay to be scored.
```

对 anchor essay，例如 low anchor：

```text
Scoring Instruction:
<I>

Score Range:
<score_min>-<score_max>

Reference Anchors:
Score: <low_score>
Essay: <e_low>

Score: <mid_score>
Essay: <e_mid>

Score: <high_score>
Essay: <e_high>

Essay to Represent:
<e_low>

Known Score:
<low_score>

Representation Target:
Encode this reference essay as a low-score anchor.
```

当前实现会尽量对 essay span 的 last-layer hidden states 做 mean pooling。
如果 span localization 不可用，则退回到更稳健的 sequence pooling。

这个设计有两个目的：

- target 和 anchors 都在同一个 instruction、score range 和 anchor context 下编码。
- anchor 的人工分数标签会进入上下文，使 anchor representation 成为 score-conditioned representation。

## 证据向量 z

证据向量 `z` 是一个 compact protocol-quality feature vector。
它不是最终预测器，而是用来判断一个 `<I, E>` 是否形成了稳定、合理的评分尺度。

当前 `z` 由以下信息构成：

- raw score features：raw predicted score、normalized score、到 score boundaries 的距离。
- hidden state features：target representation 的统计信息和投影后的 hidden 信息。
- anchor relation features：target essay 到 low / mid / high anchors 的距离或相似度。
- ordinal geometry features：target 是否更接近与 raw prediction 对应的 score-band anchors。
- score-consistency features：raw score 与 anchor-relative evidence 是否一致。
- distribution / stability features：score collapse ratio、prediction bias、TV distance、high-score recall。
- enhanced evidence features：供 PACE 和 mutation feedback 使用的额外诊断信号。

精确的 feature layout 目前在：

```text
pace/pace_fitness.py
```

当前设计判断：

- `z` 已经能用于 selection diagnostics 和部分 mutation feedback。
- 但 `z` 仍然是后续重点优化对象。
- 下一步应该把 `z` 明确拆成 named feature blocks，并记录每个 block 对 protocol selection 的贡献。

## I 和 E 的进化方式

### Instruction I

`I` 通过本地 LLM 的 reflection 和 rewrite 进化：

```text
validation errors + evidence diagnostics -> reflection feedback -> rewritten rubric
```

当前重要约束：

- score range normalization 会把 evolved rubric 拉回当前数据集的分数范围。
- rubric 被视为 holistic ordinal scoring instruction，而不是 subcriteria point-sum checklist。
- final score 必须是当前 prompt score range 内的整数。
- operational calibration text 会被追加到 rubric 中，减少 score-scale drift。

### Anchors E

`E` 的选择和变异基于真实 score band：

```text
low anchor
mid/high 附近的 boundary anchor
high 附近的 boundary anchor
high/top anchor
```

实现上避免 P1-only 规则。
score bands 来自当前 prompt 的 `score_min`、`score_max` 和当前训练池中的 observed train scores。

高分段处理包括：

- 当 top-score 样本数量足够时，优先保留 observed top-score anchors。
- anchor mutation 时设置 top-score preservation probability。
- 当 high-score recall 或 max-score recall collapse 时施加额外 penalty。

anchor mutation 可以使用 hidden-evidence representative selection：

```text
从同一 score band 构造 candidate pool
-> 用 evidence / diagnostic signal 打分
-> 可选 hidden rerank
-> 替换对应 anchor slot
```

## Selection 和 Guards

当前 selection 是 raw-first guarded selection：

```text
raw_val_qwk
raw_adjusted_qwk
PACE fitness on top candidates
anchor geometry
protocol quality
distribution guard
high-score guard
max-score guard
overfit guard
```

selection policy 是保守的：

- 如果一个候选协议 PACE 很强但 raw validation 明显弱，它不能替代 raw-best protocol。
- PACE 只能在 raw validation 竞争力足够时提供小幅 selection lift。
- 退化的 PACE 结果会 fallback 到 raw scoring。
- 最终候选协议必须在 test evaluation 之前用 validation-only signals 选出。

final evaluation 会比较多个候选族，例如：

```text
best_raw_val
best_raw_guarded
best_pace_guarded
best_pareto
```

如果多个候选族实际上是同一个协议，结果文件会标记 duplicate candidates。

## 代码结构

```text
wise_aes.py                         WISE-PACE 主进化脚本
pace/
  llm_backend.py                    本地 Llama backend，用于生成和 hidden states
  pace_fitness.py                   PACE evidence、calibrator probe、fitness 和 diagnostics
configs/
  default.yaml                      向后兼容默认配置
  phase4_smoke.yaml                 快速 smoke test 配置
  phase4_evidence_mutation.yaml     stability / pilot experiment 配置
  phase4_full_fold.yaml             单个完整 fold 实验配置
logs/                               本地实验日志，git ignored
models/                             本地模型，git ignored
data/                               ASAP 数据，git ignored
```

## 运行实验

Smoke test：

```bash
python wise_aes.py --config configs/phase4_smoke.yaml --fold 0
```

Stability pilot：

```bash
python wise_aes.py --config configs/phase4_evidence_mutation.yaml --fold 0
```

单个完整 fold：

```bash
python wise_aes.py --config configs/phase4_full_fold.yaml --fold 0
```

在 32GB GPU 上，当前 P1 full-fold 使用本地 Llama-3.1-8B backend 的运行时间约为 14-15 小时。

## 当前研究风险

当前模型还没有成熟到可以直接大规模跑完整 multi-fold。
已知风险包括：

- high-score 和 max-score recall 仍然偏弱。
- mutation 经常无法让 child protocol 超过第一代中已经出现的强 parent protocol。
- PACE signal 有时能提升 protocol quality，但不一定提升 raw test QWK。
- `z` 需要更清晰的 feature-block attribution。
- 当前 full-fold 结果只覆盖 P1 fold 0，仍然需要 P1-P8 和 multi-fold 证据。
- WISE-AES raw-only、OPRO-lite、APE-lite、random mutation、hidden-only z 等 baseline 和 ablation 还需要系统运行。

## 下一步工作

最高优先级：

1. 加强 high-score 和 max-score anchor coverage，同时避免过拟合到 P1。
2. 改进 mutation，让 child protocols 更稳定地超过 parent protocols。
3. 把 `z` 拆成清晰的 named feature blocks，并记录每个 block 对 selection 的贡献。
4. 在相同 compute budget 下运行 raw-only 和 evidence-guided ablations。
5. 先扩展到 P1-P8 smoke / stability checks，再决定是否投入完整 multi-fold training。

