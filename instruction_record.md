# Run 5-fold Experiment

```
uv run run_5fold_experiment.py --config configs/official.yaml --resume
```

# Run 5-fold Evaluation

```
uv run eval_5fold.py --gen 25 --dirs logs/exp_fold0 logs/exp_fold1 logs/exp_fold2 logs/exp_fold3 logs/exp_fold4
```

uv run eval_5fold.py --gen 25 --dirs logs/exp_20251227_143916_fold0 logs/exp_20251227_202247_fold1 logs/exp_20251227_224414_fold2 logs/exp_20251228_005153_fold3 logs/exp_20251228_030806_fold4