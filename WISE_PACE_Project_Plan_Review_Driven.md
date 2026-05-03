# WISE-PACE 项目计划书（Review-Driven EMNLP Main 版本）

> 项目目标：将原始 **WISE-AES** 从“输出层 prompt / anchor 进化框架”升级为 **WISE-PACE: Hidden-Evidence-Guided Scoring Protocol Evolution for LLM-based Automated Essay Scoring**。
> 核心思想：不再只依据 LLM 最终输出分数选择评分协议，而是利用 LLM 评分过程中的 hidden state、anchor-relative geometry、reasoning consistency、uncertainty 和 cost signals 来指导 \(I^*, E^*\) 的联合进化。
> 最终目标：形成一个可以冲击 EMNLP Main 的统一方法，而不是“WISE-AES + PACE-AES 后处理”的模块拼接。

---

## 1. 项目背景与上次评审暴露的问题

上一版 WISE-AES 的主要贡献是：把 LLM-based AES 建模为评分协议优化问题，通过进化搜索联合优化：

\[
P = \langle I, E \rangle
\]

其中：

- \(I\)：scoring instruction / evolved rubric
- \(E\)：global reference anchors / static exemplars

原始目标是：

\[
P^* = \arg\max_{P} QWK(y_{raw}(P), y)
\]

也就是选择在验证集上 raw QWK 最高的 instruction-anchor 组合。

但上次评审指出了几个关键问题：

1. **Artifact transparency 不足**
   evolved prompts、anchor sets、rubric trajectories 没有系统展示，读者无法判断方法是否真的学到了新的评分逻辑。

2. **Novelty 不够硬**
   容易被理解为 genetic algorithm + prompt rewriting + exemplars 的组合，和一般 automated prompt optimization 方法边界不够清晰。

3. **缺少关键 baseline**
   缺少 OPRO / APE / EvoPrompt 等 automated prompt optimization baseline，也缺少 human prompt engineering baseline。

4. **计算成本没有充分量化**
   需要 wall-clock time、LLM calls、token counts、API cost、per-run cost、inference cost。

5. **generalizability 证据不足**
   主要实验集中在 single model + ASAP，难以支撑“general principles”。

6. **anchor selection / representative essay selection 不够清楚**
   初始化和 mutation 机制不够透明，尤其 anchor mutation 当前更像从全训练集随机替换，不够符合“global scoring scale”的叙事。

7. **same model as optimizer and target 未充分讨论**
   meta-optimizer 和 scoring model 使用同一个 LLM，需要明确为 self-optimization setting，并设计相关 ablation 或讨论。

---

## 2. 新版本核心定位

新版本不应再写成：

```text
WISE-AES 生成 I* 和 E*
PACE-AES 在后面校准 y_raw
```

而应写成：

```text
WISE-PACE:
WISE 负责生成候选评分协议 P=<I,E>
PACE 负责用 hidden evidence 评价协议质量
PACE 的评价结果反过来指导 WISE 的 selection、mutation 和 evolution
最终更高效地找到更稳定的 I* 和 E*
```

核心定义：

> **WISE-PACE uses anchor-relative hidden evidence from the target LLM to evaluate and guide scoring protocol evolution.**

中文表达：

> **WISE-PACE 利用目标 LLM 在评分过程中的 anchor-relative hidden representations 判断评分协议是否真正形成稳定评分标尺，并用该信号指导 \(I\) 与 \(E\) 的联合进化。**

---

## 3. 最终研究问题

新版论文应围绕以下研究问题组织：

### RQ1：WISE-PACE 是否能比 WISE-AES 找到更好的评分协议？

比较：

```text
WISE-AES raw-QWK evolution
vs.
WISE-PACE hidden-evidence-guided evolution
```

核心指标：

```text
Avg QWK
MAE
Accuracy
per-prompt QWK
```

---

### RQ2：WISE-PACE 是否能用更少 generations / LLM calls 达到相同或更高性能？

比较：

```text
同样 generation 下谁更强
同样 QWK 目标下谁更省 calls
同样 call budget 下谁找到更好的 P*
```

核心指标：

```text
generations-to-target-QWK
LLM calls
tokens
wall-clock time
API cost
```

---

### RQ3：hidden evidence 是否真的提供了比 raw score 更有用的 protocol selection signal？

消融：

```text
raw QWK only
PACE QWK only
raw + PACE
raw + PACE + anchor geometry
raw + PACE + anchor geometry + stability
```

---

### RQ4：anchor geometry 是否能解释为什么某些 \(E^*\) 更好？

分析：

```text
low/mid/high anchors 在 hidden space 中是否更分离
作文 hidden states 是否更靠近正确 band anchor
hidden distance 是否与真实分数呈现 ordinal correlation
```

---

### RQ5：WISE-PACE 是否比通用 prompt optimization baseline 更适合 AES？

必须比较：

```text
OPRO-lite / APE-lite / EvoPrompt-lite
WISE-AES
WISE-PACE
```

重点说明：

```text
通用 APO 优化的是文本 prompt；
WISE-PACE 优化的是评分协议 P=<I,E>，
并使用 hidden evidence 评价协议是否形成稳定评分标尺。
```

---

## 4. 模型选择计划

### 4.1 主模型：Llama 系列

主实验建议保持与上一版一致，方便继承已有 WISE-AES 结果：

```text
Llama-3.1-8B-Instruct 或 Llama-3.1-8B-Instruct-compatible local backend
```

用途：

```text
scoring model M_score
meta-optimizer M_meta
Local-WISE replay hidden-state extractor
```

默认设置：

```text
M_meta = M_score
```

即 self-optimization setting。

---

### 4.2 补充模型：至少一个不同 family 的 7B/8B 模型

为回应 single-model generalizability 质疑，至少补一个小规模验证：

候选：

```text
Qwen2.5-7B-Instruct
Mistral-7B-Instruct
Gemma 系列 instruct model
```

建议优先级：

```text
1. Qwen2.5-7B-Instruct
2. Mistral-7B-Instruct
3. Gemma instruct
```

最小验证规模：

```text
2 prompts × 2 folds × 2 models
```

目标不是完整 SOTA，而是证明：

```text
WISE-PACE 的相对收益不只存在于 Llama-8B。
```

---

### 4.3 stronger meta-optimizer ablation

评审提到 stronger meta-optimizer 是自然设计轴。

建议做一个小规模实验：

```text
Self-meta:
M_meta = M_score = Llama-8B

Strong-meta:
M_meta = stronger LLM
M_score = Llama-8B
```

可选 stronger meta：

```text
Llama-70B API
Qwen-max / GPT-class model if accessible
Claude / GPT only if cost and policy允许
```

如果资源不足，至少在 1-2 prompts 上做 pilot。

---

## 5. 数据集计划

### 5.1 主数据集：ASAP

主实验仍以 ASAP 为核心。

原因：

```text
1. 与上一版 WISE-AES 结果可比
2. AES 领域经典 benchmark
3. 8 prompts 分数范围和题型差异明显
4. 适合分析 prompt-specific protocol evolution
```

实验设置：

```text
8 prompts × 5 folds
```

每个 prompt 内部：

```text
train / val / test split
val 再切 calib / fitness，用于 WISE-PACE evolution
```

---

### 5.2 外部数据集：建议至少做一个补充验证

为回应 external validity，建议考虑：

#### 方案 A：TOEFL11

优点：

```text
文本评分任务经典
可测试跨语域/跨水平泛化
```

风险：

```text
数据格式、评分维度和 ASAP 不完全一致，需要额外适配
```

#### 方案 B：EssayJudge subset

优点：

```text
更新、更贴近 LLM-based AES 评测
可回应最新 reviewer 对外部 benchmark 的期待
```

风险：

```text
需要确认数据可得性、标签形式、任务设置
```

#### 方案 C：ASAP cross-prompt robustness

如果外部数据集来不及，最低限度做：

```text
ASAP 内部 prompt transfer / cross-prompt robustness
```

例如：

```text
在 prompt A 上 evolved protocol 的部分诊断是否迁移到 prompt B？
或者仅比较 WISE-PACE 在不同 score range prompts 上是否稳定收益。
```

但这不能完全替代外部数据集。

---

## 6. 方法改进总览

最终 WISE-PACE 包含 5 个核心模块：

```text
1. WISE Protocol Generator
2. Local-WISE Replay
3. PACE Evidence Evaluator
4. Review-Driven Protocol Fitness
5. Evidence-Aware Evolution
```

---

## 7. 模块一：WISE Protocol Generator

### 7.1 功能

生成候选评分协议：

\[
P_i = \langle I_i, E_i \rangle
\]

其中：

```text
I_i = candidate scoring instruction
E_i = candidate global anchors
```

---

### 7.2 Instruction 初始化

使用 high/low contrastive induction：

```text
sort train essays by score
select bottom-N and top-N essays
combine with official rubric
ask LLM to induce operational scoring instruction
```

输出：

```text
I_0
```

---

### 7.3 Anchor 初始化

必须升级为清晰的分层策略：

```text
按训练集分数分布划分 Low / Mid / High pools
从每个 pool 中选择一个 anchor
E_0 = {e_low, e_mid, e_high}
```

推荐分层方式：

```text
score percentile:
Low: bottom 33%
Mid: 33%-66%
High: top 33%
```

或者按 score range：

```text
Low: [min, min+1/3 range]
Mid: middle third
High: upper third
```

建议固定一种并写清楚。

---

### 7.4 Anchor Mutation 改进

原始方式：

```text
anchor 被随机替换，新候选来自整个 train set
```

新版必须改成：

```text
stratum-preserving anchor replacement
```

即：

```text
low anchor 只从 Low pool 替换
mid anchor 只从 Mid pool 替换
high anchor 只从 High pool 替换
```

强版本进一步使用 hidden geometry 排序候选：

```text
representativeness
separation
ordinal consistency contribution
```

---

## 8. 模块二：Local-WISE Replay

### 8.1 功能

对每个 candidate protocol \(P\)，使用本地 LLM 复现 WISE scoring：

```text
essay + I + E → local LLM → y_raw + reasoning + h(x)
```

---

### 8.2 为什么需要本地模型

API 模型通常只能返回：

```text
文本输出
final_score
```

而 WISE-PACE 需要：

```text
hidden states
```

所以必须用本地 HuggingFace/transformers backend，并打开：

```text
output_hidden_states=True
```

---

### 8.3 输出

每篇 essay 记录：

```text
essay_id
y_true
y_raw
reasoning_text
hidden_state
parse_status
wallclock_sec
prompt_tokens
completion_tokens
```

---

## 9. 模块三：PACE Evidence Evaluator

PACE 构造 evidence vector：

\[
z = [y_{raw}; r_{emb}; r_s; f_o; u]
\]

---

### 9.1 \(y_{raw}\)

WISE 原始分数。

---

### 9.2 \(r_{emb}\)：anchor-relative hidden evidence

输入：

```text
h(x)
h(e_low)
h(e_mid)
h(e_high)
```

构造：

```text
h(x)
h(x) - h(e_low)
h(x) - h(e_mid)
h(x) - h(e_high)
cos(h(x), h(e_low))
cos(h(x), h(e_mid))
cos(h(x), h(e_high))
L2(h(x), h(e_low))
L2(h(x), h(e_mid))
L2(h(x), h(e_high))
```

直觉：

```text
目标作文在 LLM hidden space 中更像低分、中分还是高分 anchor？
```

---

### 9.3 \(r_s\)：reasoning features

从 reasoning text 中抽取：

```text
positive grading cues
negative grading cues
organization / grammar / evidence / coherence mentions
score inconsistency
是否多次提到不同 score
```

---

### 9.4 \(f_o\)：objective features

作文客观特征：

```text
word count
sentence count
paragraph count
average sentence length
type-token ratio
punctuation statistics
rough grammar/spelling proxies if available
```

---

### 9.5 \(u\)：uncertainty features

不确定性：

```text
hedging words
parse failure
score mention conflict
raw score near boundary
low confidence pattern
```

---

## 10. 模块四：Review-Driven Protocol Fitness

### 10.1 MVP fitness

第一阶段实现：

\[
Fitness(P)=0.7 QWK(y_{PACE},y)+0.3 QWK(y_{raw},y)
\]

其中：

```text
y_PACE = lightweight PACE calibrator output
```

---

### 10.2 完整 fitness

最终 EMNLP 版本：

\[
Fitness(P)=
\alpha QWK(y_{PACE}, y)
+\beta QWK(y_{raw}, y)
+\gamma S_{anchor}
+\delta S_{stability}
-\lambda Cost(P)
\]

建议初始权重：

```text
alpha = 0.60
beta  = 0.25
gamma = 0.10
delta = 0.05
lambda = 根据 cost scale 调节
```

---

### 10.3 calib / fitness split

不能在同一批数据上训练 PACE calibrator 又计算 fitness。

因此 validation set 内部再切：

```text
calib_items：训练 lightweight PACE calibrator
fitness_items：计算 protocol fitness
```

例如：

```text
calib_split_ratio = 0.5
```

严格要求：

```text
test set 不进入 evolution
```

---

## 11. Anchor Geometry Score

这是新版最重要的技术创新之一。

### 11.1 目标

判断 \(E=\{e_{low},e_{mid},e_{high}\}\) 是否真的构成稳定评分尺子。

---

### 11.2 Anchor Separation

\[
S_{sep}=\min(d(e_{low},e_{mid}), d(e_{mid},e_{high}), d(e_{low},e_{high}))
\]

或平均距离：

\[
S_{sep}=\frac{d_{lm}+d_{mh}+d_{lh}}{3}
\]

---

### 11.3 Ordinal Consistency

对每篇作文 \(x\)，判断其 hidden state 是否更接近正确 band anchor：

\[
S_{ord}=\frac{1}{N}\sum_i \mathbb{1}[\arg\min_b dist(h(x_i),h(e_b)) = band(y_i)]
\]

---

### 11.4 Monotonicity

定义：

\[
s(x)=sim(h(x),h(e_{high}))-sim(h(x),h(e_{low}))
\]

希望 \(s(x)\) 与真实分数 \(y\) 正相关。

计算：

```text
Spearman(s(x), y)
```

---

### 11.5 最终组合

\[
S_{anchor}=0.3S_{sep}+0.4S_{ord}+0.3S_{mono}
\]

MVP 可先只实现：

```text
S_anchor = S_sep
```

---

## 12. Protocol Stability Score

可选强版本。

### 12.1 组成

```text
raw score collapse penalty
hidden band compactness
adjacent band separation
uncertainty penalty
reasoning-score consistency
```

---

### 12.2 目的

避免选择：

```text
raw QWK 偶然较高但分数坍缩严重
或 hidden representation 混乱
或 reasoning 与 score 矛盾的 protocol
```

---

## 13. 模块五：Evidence-Aware Evolution

### 13.1 动机

原始 evolution 只是：

```text
错误案例 → LLM reflection → 改 instruction
随机替换 anchors
```

新版要变成：

```text
错误案例 + hidden mismatch + anchor geometry + reasoning contradiction
→ 决定改 instruction、换 anchor、还是淘汰 protocol
```

---

### 13.2 错误类型

| Error Type | 现象 | 改进动作 |
|---|---|---|
| Over-score with low hidden evidence | raw 分数偏高，但 hidden 更接近 low/mid | instruction 加强扣分规则 |
| Under-score with high hidden evidence | raw 分数偏低，但 hidden 更接近 high | instruction 加强高分识别 |
| Anchor confusion | anchors hidden distance 太近 | 替换 anchor |
| Reasoning-score contradiction | reasoning 说弱但给高分 | 改 scoring template |
| Raw collapse | 大量作文给同一分 | 增加分档区分描述 |
| Boundary ambiguity | 错误集中在相邻分数档 | 增加边界规则 / 边界 anchors |

---

### 13.3 Evidence-aware instruction mutation

mutation prompt 应包含：

```text
predicted score
gold score
raw error
closest anchor
hidden similarity pattern
reasoning contradictions
suggested correction target
```

---

### 13.4 Evidence-aware anchor replacement

如果发现：

```text
mid anchor 与 high anchor hidden distance 太近
```

则：

```text
从 Mid pool 中找一个更代表 mid band 且与 high anchor 更分离的 candidate
```

---

## 14. Cost-Aware Evolution

### 14.1 Top-K PACE Evaluation

每代：

```text
1. 所有 candidates 先算 raw QWK
2. 选 top-K candidates
3. 只对 top-K 跑 PACE-guided fitness
4. 其他 candidates 使用 raw QWK 或低优先级 fitness
```

推荐：

```text
population_size = 8
top_k_pace = 3
```

---

### 14.2 Early Rejection

先评估少量 mini-set：

```text
8 或 16 篇 essays
```

如果出现：

```text
raw score collapse
hidden anchor 无区分
uncertainty 高
early QWK 极低
```

则不再完整评估。

---

### 14.3 Cost logging

必须记录：

```text
LLM calls
hidden forward passes
prompt tokens
completion tokens
wall-clock time
cache hit count
cache miss count
API cost estimate
```

---

## 15. Baseline 对比计划

这是新版必须重点补齐的部分。

### 15.1 基础 LLM baseline

必须有：

```text
Zero-shot rubric prompting
Few-shot prompting
Static official rubric + fixed examples
```

---

### 15.2 AES LLM baseline

必须有：

```text
MTS
```

因为上一版主要对比它，且审稿人关注 narrow margin。

---

### 15.3 Prompt optimization baseline

至少必须有一个：

```text
OPRO-lite
APE-lite
EvoPrompt-lite
```

推荐实现顺序：

```text
1. OPRO-lite
2. APE-lite
3. EvoPrompt-lite optional
```

要求：

```text
same model
same validation set
same call budget or same generation budget
```

---

### 15.4 Human prompt engineering baseline

建议至少设计一个 controlled baseline：

```text
官方 rubric
+ 人工根据 train/val 错误迭代 2-3 次
+ 固定 anchors
```

如果没有专家参与，可以做：

```text
manual strong prompt baseline
```

但必须诚实命名，不要称为 expert if no expert.

---

### 15.5 原方法与变体 baseline

必须有：

```text
WISE-AES original
WISE-AES with stratum-preserving anchors only
WISE + post-hoc PACE
WISE-PACE MVP
WISE-PACE full
```

---

## 16. 结果指标

### 16.1 主指标

```text
Quadratic Weighted Kappa (QWK)
```

AES 主指标。

---

### 16.2 辅助指标

```text
MAE
Accuracy
Pearson / Spearman correlation
per-prompt QWK
macro average QWK
standard deviation across folds
```

---

### 16.3 稳定性指标

```text
score variance under repeated generations
temperature sensitivity
raw score collapse rate
cross-band error share
boundary error rate
```

---

### 16.4 成本指标

```text
LLM calls
prompt tokens
completion tokens
total tokens
wall-clock time
estimated API cost
cost per prompt/fold
cost to target QWK
```

---

### 16.5 hidden evidence 指标

```text
anchor_geometry_score
anchor separation
ordinal consistency
monotonicity Spearman
hidden band compactness
adjacent band overlap
```

---

## 17. 必须生成的表格

### Table 1：主实验结果

| Method | Avg QWK | MAE | Acc | Std across folds |
|---|---:|---:|---:|---:|
| Zero-shot |  |  |  |  |
| Few-shot |  |  |  |  |
| MTS |  |  |  |  |
| OPRO-lite |  |  |  |  |
| WISE-AES |  |  |  |  |
| WISE + post-hoc PACE |  |  |  |  |
| WISE-PACE MVP |  |  |  |  |
| WISE-PACE full |  |  |  |  |

---

### Table 2：per-prompt QWK

| Method | P1 | P2 | P3 | P4 | P5 | P6 | P7 | P8 | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

---

### Table 3：成本效率

| Method | Generations | LLM Calls | Tokens | Wall-clock | Cost | Avg QWK |
|---|---:|---:|---:|---:|---:|---:|

---

### Table 4：generation-to-target-QWK

| Method | Target QWK | Generations Needed | Calls Needed | Tokens Needed |
|---|---:|---:|---:|---:|

---

### Table 5：消融实验

| Variant | Avg QWK | ΔQWK | Cost | Interpretation |
|---|---:|---:|---:|---|
| WISE-PACE full |  |  |  |  |
| - PACE-guided fitness |  |  |  |  |
| - anchor geometry |  |  |  |  |
| - r_emb |  |  |  |  |
| - evidence-aware mutation |  |  |  |  |
| - early rejection |  |  |  |  |

---

### Table 6：anchor geometry 分析

| Method | Anchor Separation | Ordinal Consistency | Monotonicity | Avg QWK |
|---|---:|---:|---:|---:|

---

### Table 7：artifact transparency

| Prompt | Initial Rubric Structure | Evolved Rubric Structure | Main Changes | Anchor Scores |
|---|---|---|---|---|

---

### Table 8：model robustness

| Model | Method | Avg QWK | Δ over WISE | Notes |
|---|---|---:|---:|---|

---

### Table 9：temperature consistency

| Method | Temp | Score Variance | QWK Mean | QWK Std |
|---|---:|---:|---:|---:|

---

## 18. 必须生成的图

### Figure 1：WISE-PACE 总框架图

必须突出闭环：

```text
Protocol P=<I,E>
→ Local-WISE Replay
→ Hidden Evidence
→ PACE-guided Fitness
→ Evolution Selection/Mutation
→ P*
```

---

### Figure 2：WISE-AES vs WISE-PACE 进化曲线

```text
x-axis: generation
y-axis: val QWK / combined fitness
```

展示：

```text
WISE-PACE 更快收敛
或同代数 QWK 更高
```

---

### Figure 3：cost-efficiency curve

```text
x-axis: LLM calls / tokens / cost
y-axis: QWK
```

展示：

```text
WISE-PACE 在更低成本下达到同等 QWK
```

---

### Figure 4：hidden anchor geometry visualization

使用 UMAP / t-SNE：

```text
essay hidden states colored by true score band
low/mid/high anchors marked as stars
compare WISE-AES vs WISE-PACE
```

---

### Figure 5：rubric evolution case study

展示：

```text
I0 → intermediate I → I*
```

突出：

```text
compression
expansion
negative constraints
boundary sharpening
```

---

### Figure 6：error type distribution

比较：

```text
WISE-AES errors
WISE-PACE errors
```

错误类型：

```text
over-score
under-score
boundary error
cross-band error
raw collapse
reasoning-score contradiction
```

---

## 19. 实施阶段计划

### Phase 0：Review-driven planning

不改代码，只生成计划文档：

```text
docs/reviewer_issue_mapping.md
docs/wise_pace_method_plan.md
docs/wise_pace_implementation_roadmap.md
docs/experiment_matrix.md
```

---

### Phase 1：WISE-PACE MVP

实现：

```text
pace_guided fitness
calib / fitness split
lightweight PACE calibrator
raw_qwk fallback
generation artifact logging
cost logging
protocol signature cache
```

---

### Phase 2：Anchor geometry + anchor replacement

实现：

```text
anchor geometry score
stratum-preserving anchor mutation
hidden-guided anchor candidate ranking
anchor trajectory logging
```

---

### Phase 3：Cost-aware evolution

实现：

```text
top-k PACE evaluation
early rejection
generation-to-target-QWK analysis
cost summary scripts
```

---

### Phase 4：Evidence-aware mutation

实现：

```text
hidden mismatch diagnostics
reasoning-score contradiction detection
evidence-aware rubric mutation prompt
evidence-aware anchor replacement rule
```

---

### Phase 5：Baselines and analysis

实现：

```text
OPRO-lite / APE-lite
manual strong prompt baseline
temperature consistency script
hidden geometry visualization
artifact export script
```

---

## 20. 最终 EMNLP Main 成功标准

最终版本至少应达到：

```text
1. WISE-PACE full 平均 QWK > WISE-AES
2. WISE-PACE full > WISE + post-hoc PACE
3. WISE-PACE 在相同 QWK 下需要更少 generations / calls
4. anchor geometry 消融有效
5. r_emb 消融有效
6. 至少一个 automated prompt optimization baseline 被击败
7. artifact 完整展示：I0, I*, E*, anchor IDs, anchor snippets
8. 成本表完整
9. 至少一个小规模 model robustness 实验
10. 无 test leakage
```

---

## 21. 论文贡献写法

最终贡献建议写成：

### Contribution 1

We formulate LLM-based AES as a scoring protocol evolution problem over instructions and global anchors.

### Contribution 2

We introduce anchor-relative hidden evidence to evaluate whether a scoring protocol forms a stable internal scoring scale in the target LLM representation space.

### Contribution 3

We propose WISE-PACE, a hidden-evidence-guided and cost-aware evolution framework that uses PACE-derived protocol fitness to discover better \(I^*\) and \(E^*\) with fewer generations.

### Contribution 4

We provide systematic artifact-level analysis, cost accounting, and comparisons against prompt optimization baselines to clarify when and why protocol evolution improves LLM-based AES.

---

## 22. 最终一句话总结

> **WISE-PACE 把 LLM 作文评分从“优化一个 prompt”升级为“优化一个可审查、可诊断、可进化的评分协议”。它利用 LLM 评分过程中的 hidden evidence 判断评分协议是否真的形成 low/mid/high 评分标尺，并用该信号指导 instruction 和 anchors 的联合进化，从而提高性能、降低搜索成本，并增强方法的解释性和可复现性。**
