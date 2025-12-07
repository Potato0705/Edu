# ASAP 数据集

请从 Kaggle 下载 ASAP 数据集并放置在此目录。

## 下载方式

1. 访问 Kaggle 竞赛页面: https://www.kaggle.com/c/asap-aes/data
2. 下载 `training_set_rel3.tsv` 文件
3. 重命名为 `asap_prompt8.tsv` 或修改 `config.yaml` 中的路径

## 或使用命令行下载

```bash
# 安装 kaggle CLI (如果没有)
pip install kaggle

# 配置 Kaggle API (需要 ~/.kaggle/kaggle.json)
kaggle competitions download -c asap-aes

# 解压并重命名
unzip asap-aes.zip
mv training_set_rel3.tsv asap_prompt8.tsv
```

## 数据格式

TSV 文件，包含以下列:
- essay_id: 文章 ID
- essay_set: Prompt 编号 (1-8)
- essay: 文章文本
- domain1_score: 评分

Prompt 8 (Laughter) 的评分范围是 0-60 分。
