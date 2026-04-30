# PACE-AES 实验设计清单

更新时间：2026-04-22

## 1. 文档目的

本清单用于统一后续实验安排，避免重复跑已经完成或可以复用的部分，并把论文需要证明的核心问题拆成可执行的实验任务。

本阶段的总原则只有一句话：

**不把算力浪费在重复验证 WISE-AES 已知结论上，而是集中证明 PACE-AES 为什么有效、有效在哪里、是否具有稳定性和可迁移性。**

---

## 2. 当前研究主张

### 2.1 核心问题

我们当前要回答的问题不是“WISE-AES 能不能继续涨分”，而是：

**在 WISE-AES 已经得到较优 scoring protocol 的前提下，剩余误差是否主要来自 score-boundary misalignment，以及 PACE-AES 能否系统性修复这一问题。**

### 2.2 论文主线

建议把方法主线收敛为以下三点：

1. **提出并实证 Protocol-to-Score Misalignment**
   即：protocol alignment 不等于 score alignment，WISE-AES 的残余误差大量集中在 band boundary 附近。

2. **提出 anchor-relative latent evidence**
   将 evolved anchors 从 prompt 内的静态示例，提升为表示空间中的相对评分坐标。

3. **提出 protocol-conditioned ordinal calibrator**
   用显式有序边界建模，而不是普通回归式后处理，专门修正 boundary 附近的偏差。

### 2.3 不作为当前主贡献的点

以下内容可保留为增强项，但不建议作为当前论文主创新：

1. Bilevel anchor-calibrator refinement
2. Uncertainty-aware abstention / routing
3. Information-theoretic view

这些内容只有在实验收益明确、叙事不分散的前提下，再考虑提升为主结果。

---

## 3. 当前可复用资产

### 3.1 可以直接复用的结果

1. 现有 WISE-AES Layer 1 日志与 champion protocol
2. ASAP 上已有的 WISE-AES 预测结果
3. RQ0 诊断结果
   当前已有：
   - `results/rq0/summary.json`
   - `cross_band_error_share = 0.8697`
   - `gate_pass = true`
4. 已完成的本地 / 服务器侧 PACE 代码基础设施
   - `pace/llm_backend.py`
   - `pace/datasets/asap.py`
   - `pace/evidence.py`
   - `pace/calibration.py`
   - `pace/experiments/build_anchor_cache.py`
   - `pace/experiments/train_pace_calibrator.py`

### 3.2 当前不需要重新跑的部分

**原则上不重新全量跑 WISE-AES。**

理由：

1. PACE-AES 的定位是 Layer 2，而不是重做 Layer 1
2. 当前最需要证明的是 boundary-aware calibration 的有效性
3. 重跑 WISE-AES 不会直接回答本工作最关键的问题

### 3.3 只有在以下情况才考虑补跑 WISE-AES

1. 旧日志不完整，无法支撑最终主表
2. 某些 prompt / fold 缺失，导致主实验不能闭环
3. 最终决定修改 Layer 1 本身，而不是只做 Layer 2

---

## 4. 研究问题（RQ）设计

建议论文主体围绕以下 RQ 展开。

### RQ0：WISE-AES 的残余误差是否主要体现为 Protocol-to-Score Misalignment？

目标：

1. 证明 WISE-AES 的主要错误并非完全随机
2. 证明很多误差集中在 band boundary 附近
3. 为 Layer 2 的必要性提供直接实证动机

当前状态：

1. 已有初步结果
2. 需要补图表和按 prompt 的细分分析

### RQ1：PACE-AES 是否能在固定的 WISE-AES protocol 上稳定提升评分性能？

目标：

1. 证明在不改动 Layer 1 的条件下，Layer 2 仍可带来稳定提升
2. 证明提升不仅体现在 overall QWK，也体现在 boundary-sensitive 指标

### RQ2：PACE-AES 的提升是否来自 anchor-relative evidence，而不是 generic hidden-state feature？

目标：

1. 区分 “有 hidden states” 和 “有 anchor-relative hidden evidence” 这两个概念
2. 证明 anchors 在表示空间中确实起结构性作用

### RQ3：为什么必须使用 ordinal calibrator，而不是普通 regression / classification head？

目标：

1. 证明 PACE 不是“随便接个后处理头”
2. 证明显式有序边界建模对 boundary correction 是必要的

### RQ4：PACE-AES 是否真正依赖于 WISE-AES 演化得到的 protocol / anchors？

目标：

1. 防止 reviewer 认为这是一个与 WISE 解耦的普通校准器
2. 证明 Layer 2 的效果与 Layer 1 的 protocol quality 有结构性关联

### RQ5：PACE-AES 的校准收益是否主要集中在 boundary-near 样本上？

目标：

1. 验证方法与论文动机一致
2. 避免只报 overall QWK 而看不出方法究竟修复了什么

---

## 5. 主实验与消融实验清单

## 5.1 主比较模型

建议主表至少包含以下方法：

1. **WISE-AES raw**
   使用 Layer 1 原始输出分数，不做 Layer 2

2. **WISE + 简单后处理**
   例如：
   - `y_raw` only linear / MLP regressor
   - 或 `y_raw + objective features`

3. **WISE + hidden-state only calibrator**
   使用 `h(x)`，但不使用 anchor-relative residual

4. **WISE + anchor-relative + non-ordinal head**
   使用 anchor-relative 特征，但头部仍然是普通 regression

5. **PACE-AES full**
   使用：
   - `y_raw`
   - anchor-relative latent evidence
   - structured reasoning features
   - objective features
   - uncertainty features
   - ordinal calibrator

如果篇幅允许，可再加一个更轻的版本：

6. **PACE-AES light**
   仅使用 `y_raw + anchor-relative hidden evidence + ordinal head`

这个版本有助于证明“最核心增益来自哪里”。

---

## 5.2 特征级消融

建议使用同一个 calibrator 框架，只替换输入特征，形成一组干净的 ablation。

### A 组：输入特征消融

1. `A0`: `y_raw` only
2. `A1`: `y_raw + objective features`
3. `A2`: `y_raw + h(x)`
4. `A3`: `y_raw + h(x) + reasoning features`
5. `A4`: `y_raw + anchor-relative hidden evidence`
6. `A5`: `PACE full`

关键比较：

1. `A2 vs A4`
   证明不是只要有 hidden states 就行，而是 anchor-relative evidence 才是关键

2. `A4 vs A5`
   证明 reasoning / objective / uncertainty 是增强项，而不是主效应伪装

---

## 5.3 头部结构消融

在输入特征固定的前提下，比较不同头部：

1. Linear regression
2. MLP regression
3. Multi-class classification head
4. CORAL / CORN ordinal head

目标：

1. 证明有序边界建模是必要的
2. 证明 PACE 的优势不只是“特征更强”，而是“特征 + ordinal decoding”共同作用

---

## 5.4 Anchor 来源消融

在其他设置固定时，比较不同 anchor 方案：

1. Random anchors
2. Score-stratified static anchors
3. WISE evolved anchors
4. Optional：bilevel refined anchors

目标：

1. 证明 Layer 2 不是与 Layer 1 脱耦的普通校准器
2. 证明 evolved anchors 确实更适合后续 calibrator 使用

---

## 5.5 Boundary-focused 分析

这一组不是额外模型，而是必须做的分析视角。

建议单独汇报：

1. Boundary-near subset 的 QWK / MAE
2. Cross-band error share
3. Same-band vs cross-band correction
4. Off-by-1 与 off-by-2+ 的变化
5. 不同 prompt 上的边界修正收益

目的：

1. 让实验结果直接对齐方法动机
2. 避免 reviewer 认为只是 overall QWK 的偶然上涨

---

## 6. 数据集安排

### 6.1 主数据集

**ASAP 作为主实验数据集。**

理由：

1. 当前 WISE-AES Layer 1、RQ0、PACE 代码都已围绕 ASAP 对齐
2. ASAP 上已有最完整的旧结果与工程基础
3. 最容易形成主线闭环

### 6.2 泛化数据集

仅在 ASAP 主线跑顺后，再增加 1 个额外数据集做泛化验证。

优先级建议：

1. Cambridge-FCE
2. DREsS
3. ASAP++

策略：

1. 不复制全部消融矩阵
2. 只跑主方法与关键 baseline
3. 目标是证明方法不是 ASAP 专属技巧

---

## 7. 指标与统计报告

主指标建议如下。

### 7.1 主指标

1. QWK
2. MAE

### 7.2 核心辅助指标

1. Cross-band error share
2. Same-band error share
3. Off-by-1 band share
4. Off-by-2+ band share
5. Mean band distance

### 7.3 校准 / 稳定性相关指标

1. Boundary-near subset MAE
2. Boundary-near subset QWK
3. Pearson / Spearman
   仅用于 local-vs-openrouter sanity gate，不作为最终主指标

### 7.4 如果做 uncertainty 扩展

1. ECE
2. Selective risk-coverage curve
3. Reject top-X% uncertain samples 后的 QWK

---

## 8. 图表规划

建议提前对齐论文图表，避免实验做完后才发现证据形态不够。

### Figure 1

PACE-AES 整体方法框架图

### Figure 2

RQ0：Protocol-to-Score Misalignment 诊断图

建议包含：

1. error histogram
2. per-prompt cross-band ratio
3. band-distance CDF

### Figure 3

Anchor-relative latent evidence 示意图

重点强调：

1. `h(x)`
2. `h(e_low/mid/high)`
3. residual embeddings `h(x)-h(e_k)`
4. ordinal calibrator

### Table 1

主结果表：WISE vs PACE

### Table 2

特征级消融

### Table 3

头部结构消融

### Table 4

Anchor 来源消融

### Optional Figure 4

Boundary-near subset 上的收益可视化

---

## 9. 执行顺序

这是最重要的实验排期部分。

### 第 1 阶段：验证主线是否成立

先做小规模 pilot：

1. 数据集：ASAP
2. Prompt：`1 / 3 / 4 / 7 / 8`
3. Fold：先做 `fold0`

目标：

1. 确认 PACE pipeline 稳定
2. 初筛最有效的特征组合
3. 初筛最有效的 calibrator head

### 第 2 阶段：主消融

在 pilot 结论稳定后，扩大到：

1. 数据集：ASAP
2. Prompt：`1 / 3 / 4 / 7 / 8`
3. Fold：`5 folds`

目标：

1. 跑完 RQ1-RQ5 的主体结果
2. 完成主表与核心消融表

### 第 3 阶段：正式主表

如果资源允许，再扩展到：

1. ASAP 全部 prompts
2. 只保留最关键 baseline

目标：

1. 完成最终主结果表
2. 避免在全量设置中重复跑庞大消融矩阵

### 第 4 阶段：泛化验证

仅在主线已稳定后进行：

1. 选 1 个额外数据集
2. 只跑主方法与关键对照

---

## 10. 实验停止条件与取舍原则

为了避免实验发散，需要明确停止条件。

### 10.1 不继续深挖的情况

以下情况出现时，不建议继续扩大该方向：

1. 某个增强模块收益持续小于 `0.003 ~ 0.005 QWK`
2. 结果不稳定，跨 fold 方差过大
3. 无法解释其收益来自哪里

### 10.2 可以降级为附录或补充实验的情况

1. Bilevel refinement 收益很小
2. Uncertainty 模块只在少数 prompt 有效
3. 泛化数据集收益不稳定但趋势一致

### 10.3 必须做扎实的核心部分

以下部分不能弱：

1. RQ0 诊断
2. `h(x)` vs anchor-relative residual 的消融
3. regression head vs ordinal head 的消融
4. boundary-near 样本分析

---

## 11. 工程执行原则

### 11.1 统一复用 evidence cache

后续所有消融，尽量共享同一份 `evidence_cache.pt`。

原因：

1. 最贵的是本地 Llama 前向与 hidden-state 抽取
2. calibrator 训练本身很便宜
3. 先抽特征、后多次训练轻量头部，能显著节省成本

### 11.2 不重复抽取相同 anchor hidden states

anchor hidden states 应统一缓存，按：

1. dataset
2. prompt
3. fold
4. backend signature

进行管理。

### 11.3 所有对比必须控制 protocol 一致

除非实验目标就是比较 anchor / protocol 本身，否则：

1. 同一组实验必须固定 `P*`
2. 只允许改变：
   - 特征输入
   - calibrator 结构
   - 训练目标

这样实验结论才干净。

---

## 12. 当前建议的最小可行实验集

如果只做一版最小但足够写论文的实验，建议先完成以下内容：

1. RQ0 诊断图与统计表
2. ASAP 上 `5 prompts x fold0` 的 pilot
3. `h(x)` vs anchor-relative residual 的消融
4. regression head vs ordinal head 的消融
5. WISE raw vs PACE full 的主对比
6. boundary-near subset 分析

如果这组结果成立，再扩展到：

1. `5 prompts x 5 folds`
2. 1 个额外数据集

---

## 13. 结论性决策

当前实验策略确定如下：

1. **不重新全量跑 WISE-AES**
2. **将 WISE-AES 视为固定 Layer 1**
3. **后续算力优先投入到 Layer 2 的因果证据链**
4. **主线先集中在 ASAP**
5. **先做 pilot，再做 full 5-fold，再决定是否扩全 prompt 和跨数据集**

---

## 14. 下一步立即执行事项

1. 在 AutoDL 上完成项目代码、模型、日志上传
2. 跑通 ASAP `prompt 1 / fold 0` 的 PACE 主流程
3. 固定一版 pilot 的主比较模型与消融列表
4. 确认主实验输出格式：
   - summary
   - per-prompt summary
   - per-essay prediction
   - boundary analysis
5. 开始积累可直接写进论文表格的结果

