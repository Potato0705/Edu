项目名称: WISE-AES (Weakly-supervised Integrated Scoring Evolution)

1. 任务背景

我们要针对 ASAP (Automated Student Assessment Prize) 数据集的 Prompt 8 构建一个自动化评分系统。
- Prompt 8 类型: Narrative (记叙文)。
- 题目: "Laughter" (讲述一次笑声在其中扮演重要角色的经历)。
- 评分范围: 0-60分 (ASAP 原始数据中的 domain1_score)。在代码中我们需要将其归一化或直接预测该分数。

2. 核心算法架构

由于我们没有大量的人工标注数据用于微调 BERT，我们将采用 基于 LLM 的进化优化 方法。系统分为三个核心模块：

模块 A: 初始化 (Instruction Induction)

不要使用人工编写的 Prompt。系统必须通过观察数据自动生成初始评分标准（Rubric）。
- 输入: 从训练集中随机抽取的少量样本（例如 5 篇高分，5 篇低分）。
- 过程: 让 LLM 分析这些样本的差异，反推评分标准。
- 输出: 一段初始的 System Prompt（包含评分维度和标准）。

模块 B: 提示词进化 (基于反馈的文本优化)

这是对 Prompt 文本部分的优化（类似 TextGrad/ProTeGi，但不计算数学梯度）。
- 机制:
1. 用当前 Prompt 对验证集打分。
2. 找出预测偏差最大的案例（Bad Cases）。
3. 让 LLM 生成自然语言反馈（例如：“模型未能识别出讽刺性的幽默，导致给分过低”）。
4. 让 LLM 根据反馈修改 Prompt 文本。

模块 C: 上下文示例进化 (基于遗传算法的 ICL)

这是对 Prompt 中 Few-Shot 示例（In-Context Learning Exemplars）的优化。
- 搜索空间: 训练集中的所有文章。
- 基因型: 一个包含 $K$ 个文章 ID 的列表（例如 ``)。
- 进化操作:
  - 交叉 (Crossover): 交换两个 Prompt 的示例集合。
  - 变异 (Mutation): 随机替换集合中的某篇范文。
- 目标: 找到最能帮助 LLM 理解评分逻辑的那一组范文。

3. 数据集结构模拟 (Mock Data)

由于环境限制，假设我们通过以下 Pandas DataFrame 结构读取数据：

columns: ['essay_id', 'essay_text', 'domain1_score', 'essay_set']essay_set == 8