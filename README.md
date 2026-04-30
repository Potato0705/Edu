# Edu / WISE-AES / PACE-AES

本仓库用于自动作文评分（Automated Essay Scoring, AES）实验，基础代码来自 WISE-AES，并在此基础上新增了本地复现实验、PACE-AES Layer-2 校准器，以及 Prompt-Aware Automatic Recipe Selector（PARS）。

## 项目目标

当前实验重点是：在不改动 WISE Layer-1 的协议进化、anchor 选择和 raw scoring 逻辑的前提下，利用 Layer-1 产生的 hidden evidence，对 Local-WISE replay 的原始分数进行二阶段校准。

核心对照包括：

- `Local-WISE replay`：本地复现 WISE-AES 的 raw scoring 结果。
- `PACE-AES`：基于 hidden evidence 的 Layer-2 ordinal calibrator。
- `Manual Prompt-wise Recipe`：人工指定每个 prompt 的 Layer-2 recipe。
- `PARS`：基于 train/val diagnostics 自动选择 recipe 的 prompt-aware selector。

## 方法概览

PACE-AES 的输入来自 WISE Layer-1 评分过程：

- `y_raw`：Local-WISE 生成的原始分数。
- `r_emb`：LLM hidden state 构成的 residual evidence 表示。
- `z`：由 raw score、hidden evidence、anchor relation、ordinal features 和 uncertainty features 拼接得到的 Layer-2 输入向量。

Layer-2 使用 CORAL-style ordinal calibrator 输出校准后的分数。训练目标包括 ordinal loss、soft-QWK surrogate，以及可选的 boundary-aware MMD separation loss。

PARS 在每个 `prompt × fold` 上执行两阶段选择：

1. `Rule Gate`：根据 score range、Local-WISE error pattern、raw score collapse、`r_emb` adjacent overlap 等 diagnostics 缩小候选 recipe。
2. `Inner-Val Selection`：只用 train/val，在候选 recipe 中按 QWK 选择；当 QWK 差距小于 tolerance 时，优先 MAE 更低，再比较 ACC 和 recipe 简单度。

该流程不使用 test label 做选择，test set 只用于最终报告。

## 目录结构

```text
pace/
  calibration.py                         # Layer-2 ordinal calibrator、decode、QWK/MMD loss
  evidence.py                            # hidden evidence / feature vector 构建
  llm_backend.py                         # 本地 Llama backend
  datasets/asap.py                       # ASAP 数据加载与 fold split
  experiments/
    build_anchor_cache.py                # anchor hidden cache 构建
    train_pace_calibrator.py             # 单个 prompt/fold 的 Layer-2 训练
    sweep_pace_calibrator.py             # 参数 sweep
    run_auto_recipe_selector.py          # PARS 自动 recipe 选择入口
  selector/
    recipe_library.py                    # R0-R7 固定 recipe library
    diagnostics.py                       # prompt-level diagnostics
    rule_gate.py                         # rule-gated candidate selection
    auto_select.py                       # inner-val selection + final retrain/eval
configs/
  selector_default.yaml                  # selector 默认配置，默认关闭
```

大型实验产物不会进入 Git：

- `models/`
- `data/`
- `cache/`
- `logs/`
- `results/`

## 环境

项目使用 Python 3.12。基础依赖见 `pyproject.toml`。

```bash
uv sync
```

服务器离线运行时通常需要：

```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 常用命令

### 构建 anchor cache

```bash
python -m pace.experiments.build_anchor_cache \
  --model-path /path/to/Meta-Llama-3.1-8B-Instruct \
  --logs-root /path/to/logs \
  --prompts 1 2 3 4 5 6 7 8 \
  --folds 0 1 2 3 4 \
  --cache-root /path/to/cache/pace_anchor_cache \
  --dataset asap
```

### 训练单个 Layer-2 calibrator

```bash
python -u -m pace.experiments.train_pace_calibrator \
  --model-path /path/to/Meta-Llama-3.1-8B-Instruct \
  --logs-root /path/to/logs \
  --prompt 7 \
  --fold 1 \
  --cache-root /path/to/cache/pace_anchor_cache \
  --out-dir /path/to/results/pace_fixed_recipe_5fold/p7_mmd_v1 \
  --epochs 25 \
  --batch-size 32 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --hidden-dim 512 \
  --dropout 0.1 \
  --lambda-qwk 2.0 \
  --decode-mode blend_round \
  --blend-alpha 0.65 \
  --max-delta-frac 0.10 \
  --mmd-enable \
  --lambda-sep 0.05 \
  --mmd-sample-mode boundary_only \
  --mmd-boundary-mode raw_boundary \
  --mmd-raw-boundary-epsilon 2.0 \
  --mmd-num-bands 3 \
  --mmd-project-dim 128
```

### 运行 PARS

PARS 只读取已有的 `evidence_cache.pt`，不会重新调用 WISE Layer-1。

```bash
python -u -m pace.experiments.run_auto_recipe_selector \
  --evidence-search-roots \
    /path/to/results/pace_fixed_recipe_5fold \
    /path/to/results/pace_struct_runs \
    /path/to/results/pace_calibration_bf16 \
  --out-dir /path/to/results/pars_v2_5fold \
  --prompts 1 2 3 4 5 6 7 8 \
  --folds 0 1 2 3 4 \
  --mode pars \
  --recipe-library v2 \
  --epochs 25 \
  --batch-size 32 \
  --tie-qwk-tolerance 0.005 \
  --device cuda
```

主要输出：

- `selector_features.csv`
- `selector_decisions.csv`
- `selector_summary.json`
- `five_fold_main_results.csv`
- `five_fold_main_results_by_prompt.csv`
- `five_fold_main_results_overall.csv`

## 当前 PARS v2 规则

PARS v2 保留基础 recipe `R0-R4`，并为宽分域 raw-collapse 场景加入 `R5-R7`。

关键约束：

- `score_span < 20` 的 prompt 只允许选择 `R0/R1/R2`，避免 P1-P6 被误路由到 P7/P8 的 wide recipe。
- `score_span >= 20` 的 prompt 才允许 `R3/R4/R5/R6/R7`。
- 当 train/val 上出现 `raw_mode_share` 高或 `raw_std_frac` 低时，触发 raw-collapse gate，开放更强修正能力的 recipe。

这个约束用于避免非宽分域 prompt 误选 blend/wide recipe，同时保留 P7/P8 的 collapse rescue 能力。

## 注意事项

- 不要提交 ASAP 原始数据、模型权重、cache 或实验输出。
- `evidence_cache.pt` 是 PARS 的必要输入，但属于实验产物，默认不进 Git。
- 完整 5-fold PARS 需要每个 `prompt × fold` 都已经生成对应的 `evidence_cache.pt`。
