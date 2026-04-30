"""Batch hyperparameter sweep launcher for the PACE-AES calibrator.

This script reuses `pace.experiments.train_pace_calibrator` as the single-run
worker and adds:

1. A small, curated hyperparameter grid for QWK stability.
2. Resume-friendly skip logic: existing `summary.json` results are reused.
3. Automatic metric aggregation against:
   - Original WISE-AES log reference (`final_result.json:test_qwk`)
   - Local-WISE replay (`y_raw`)
   - PACE-AES (`y_pred`)

The intended first use is a fold0 pilot sweep over prompts that currently show
QWK instability (e.g. P2/P7/P8), but the script supports arbitrary prompt/fold
lists.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_GRID: List[Dict] = [
    {
        "id": "s0_lr1e-3_lq0p25_h512_d0p1",
        "lr": 1.0e-3,
        "lambda_qwk": 0.25,
        "hidden_dim": 512,
        "dropout": 0.1,
    },
    {
        "id": "s1_lr3e-4_lq0p25_h512_d0p1",
        "lr": 3.0e-4,
        "lambda_qwk": 0.25,
        "hidden_dim": 512,
        "dropout": 0.1,
    },
    {
        "id": "s2_lr1e-4_lq0p25_h512_d0p1",
        "lr": 1.0e-4,
        "lambda_qwk": 0.25,
        "hidden_dim": 512,
        "dropout": 0.1,
    },
    {
        "id": "s3_lr3e-4_lq1p0_h512_d0p1",
        "lr": 3.0e-4,
        "lambda_qwk": 1.0,
        "hidden_dim": 512,
        "dropout": 0.1,
    },
    {
        "id": "s4_lr3e-4_lq2p0_h512_d0p1",
        "lr": 3.0e-4,
        "lambda_qwk": 2.0,
        "hidden_dim": 512,
        "dropout": 0.1,
    },
    {
        "id": "s5_lr3e-4_lq1p0_h256_d0p1",
        "lr": 3.0e-4,
        "lambda_qwk": 1.0,
        "hidden_dim": 256,
        "dropout": 0.1,
    },
    {
        "id": "s6_lr3e-4_lq1p0_h128_d0p2",
        "lr": 3.0e-4,
        "lambda_qwk": 1.0,
        "hidden_dim": 128,
        "dropout": 0.2,
    },
    {
        "id": "s7_lr1e-4_lq1p0_h256_d0p2",
        "lr": 1.0e-4,
        "lambda_qwk": 1.0,
        "hidden_dim": 256,
        "dropout": 0.2,
    },
]


def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--logs-root", type=str, default="logs")
    p.add_argument("--cache-root", type=str, default="cache/pace_anchor_cache")
    p.add_argument("--base-out-dir", type=str, default="results/pace_sweeps")
    p.add_argument("--sweep-name", type=str, default="pilot_qwk_stability")
    p.add_argument("--prompts", type=int, nargs="+", required=True)
    p.add_argument("--folds", type=int, nargs="+", required=True)
    p.add_argument("--gen", type=int, default=25)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--weight-decay", type=float, default=1.0e-4)
    p.add_argument("--max-new-tokens", type=int, default=768)
    p.add_argument(
        "--decode-mode",
        type=str,
        default="threshold",
        choices=["threshold", "expected_round", "blend_round"],
    )
    p.add_argument("--blend-alpha", type=float, default=1.0)
    p.add_argument("--max-delta-frac", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-every", type=int, default=25)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--limit-val", type=int, default=0)
    p.add_argument("--limit-test", type=int, default=0)
    p.add_argument("--python-bin", type=str, default=sys.executable)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--rebuild-evidence", action="store_true")
    p.add_argument("--no-resume-evidence", action="store_true")
    p.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Do not launch new runs. Only summarize existing sweep outputs.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rerun even if the target summary.json already exists.",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining runs if one setting crashes.",
    )
    return p.parse_args()


def _find_exp_dir(logs_root: Path, prompt: int, fold: int) -> Path:
    candidates: List[Path] = []
    for d in sorted(logs_root.glob(f"exp_*_fold{fold}")):
        cfg_path = d / "config.yaml"
        if not cfg_path.exists():
            continue
        try:
            with cfg_path.open("r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            continue
        if int(cfg.get("data", {}).get("essay_set", -1)) == prompt:
            candidates.append(d)
    if not candidates:
        raise FileNotFoundError(
            f"No exp dir under {logs_root} for prompt={prompt} fold={fold}"
        )
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _load_old_wise_qwk(logs_root: Path, prompt: int, fold: int) -> float:
    exp_dir = _find_exp_dir(logs_root, prompt, fold)
    final_path = exp_dir / "final_result.json"
    with final_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return float(obj["test_results"]["test_qwk"])


def _compute_split_metrics(df: pd.DataFrame, col: str) -> Dict[str, float]:
    qwk = float(cohen_kappa_score(df["y_true"], df[col], weights="quadratic"))
    mae = float(mean_absolute_error(df["y_true"], df[col]))
    acc = float(accuracy_score(df["y_true"], df[col]))
    return {"qwk": qwk, "mae": mae, "acc": acc}


def _write_setting_manifest(path: Path, setting: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(setting, f, ensure_ascii=False, indent=2)


def _train_one(
    *,
    python_bin: str,
    setting: Dict,
    prompt: int,
    fold: int,
    args: argparse.Namespace,
    setting_out_dir: Path,
) -> None:
    summary_path = setting_out_dir / f"p{prompt}_fold{fold}" / "summary.json"
    if summary_path.exists() and not args.force:
        print(f"[sweep] SKIP existing result: {summary_path}")
        return

    cmd: List[str] = [
        python_bin,
        "-u",
        "-m",
        "pace.experiments.train_pace_calibrator",
        "--model-path",
        args.model_path,
        "--logs-root",
        args.logs_root,
        "--prompt",
        str(prompt),
        "--fold",
        str(fold),
        "--cache-root",
        args.cache_root,
        "--out-dir",
        str(setting_out_dir),
        "--gen",
        str(args.gen),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(setting["lr"]),
        "--weight-decay",
        str(args.weight_decay),
        "--hidden-dim",
        str(setting["hidden_dim"]),
        "--dropout",
        str(setting["dropout"]),
        "--lambda-qwk",
        str(setting["lambda_qwk"]),
        "--decode-mode",
        str(args.decode_mode),
        "--blend-alpha",
        str(args.blend_alpha),
        "--max-delta-frac",
        str(args.max_delta_frac),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--seed",
        str(args.seed),
        "--checkpoint-every",
        str(args.checkpoint_every),
    ]
    if args.limit_train:
        cmd += ["--limit-train", str(args.limit_train)]
    if args.limit_val:
        cmd += ["--limit-val", str(args.limit_val)]
    if args.limit_test:
        cmd += ["--limit-test", str(args.limit_test)]
    if args.load_in_4bit:
        cmd.append("--load-in-4bit")
    if args.rebuild_evidence:
        cmd.append("--rebuild-evidence")
    if args.no_resume_evidence:
        cmd.append("--no-resume-evidence")

    print("[sweep] RUN", " ".join(cmd))
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_DATASETS_OFFLINE", "1")
    subprocess.run(cmd, cwd=str(_REPO_ROOT), env=env, check=True)


def _iter_existing_runs(sweep_root: Path) -> Iterable[tuple[Path, Path, Path]]:
    for setting_dir in sorted(p for p in sweep_root.iterdir() if p.is_dir()):
        setting_manifest = setting_dir / "sweep_setting.json"
        if not setting_manifest.exists():
            continue
        for run_dir in sorted(setting_dir.glob("p*_fold*")):
            summary_path = run_dir / "summary.json"
            per_essay_path = run_dir / "per_essay.csv"
            if summary_path.exists() and per_essay_path.exists():
                yield setting_dir, summary_path, per_essay_path


def _summarize_sweep(sweep_root: Path, logs_root: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for setting_dir, summary_path, per_essay_path in _iter_existing_runs(sweep_root):
        setting = json.loads((setting_dir / "sweep_setting.json").read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        df = pd.read_csv(per_essay_path)
        test = df[df["split"] == "test"].copy()
        if test.empty:
            continue
        prompt = int(summary["prompt"])
        fold = int(summary["fold"])
        old_qwk = _load_old_wise_qwk(logs_root, prompt, fold)
        raw_metrics = _compute_split_metrics(test, "y_raw")
        pace_metrics = _compute_split_metrics(test, "y_pred")
        improved = float(
            ((test["y_pred"] - test["y_true"]).abs() < (test["y_raw"] - test["y_true"]).abs()).mean()
        )
        worsened = float(
            ((test["y_pred"] - test["y_true"]).abs() > (test["y_raw"] - test["y_true"]).abs()).mean()
        )
        rows.append(
            {
                "setting_id": setting["id"],
                "prompt": prompt,
                "fold": fold,
                "lr": setting["lr"],
                "lambda_qwk": setting["lambda_qwk"],
                "hidden_dim": setting["hidden_dim"],
                "dropout": setting["dropout"],
                "best_epoch": summary["best_epoch"],
                "old_wise_qwk_ref": old_qwk,
                "local_wise_qwk": raw_metrics["qwk"],
                "pace_qwk": pace_metrics["qwk"],
                "delta_qwk_vs_local": pace_metrics["qwk"] - raw_metrics["qwk"],
                "delta_qwk_vs_old_ref": pace_metrics["qwk"] - old_qwk,
                "local_wise_mae": raw_metrics["mae"],
                "pace_mae": pace_metrics["mae"],
                "delta_mae": pace_metrics["mae"] - raw_metrics["mae"],
                "local_wise_acc": raw_metrics["acc"],
                "pace_acc": pace_metrics["acc"],
                "delta_acc": pace_metrics["acc"] - raw_metrics["acc"],
                "improved_share": improved,
                "worsened_share": worsened,
                "run_dir": str(summary_path.parent),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["prompt", "fold", "setting_id"]).reset_index(drop=True)
    out.to_csv(sweep_root / "summary_by_run.csv", index=False)

    best = (
        out.sort_values(
            ["prompt", "fold", "pace_qwk", "delta_mae", "delta_acc"],
            ascending=[True, True, False, True, False],
        )
        .groupby(["prompt", "fold"], as_index=False)
        .first()
    )
    best.to_csv(sweep_root / "best_by_prompt_fold.csv", index=False)

    setting_summary = (
        out.groupby("setting_id", as_index=False)
        .agg(
            n_runs=("setting_id", "size"),
            mean_old_wise_qwk_ref=("old_wise_qwk_ref", "mean"),
            mean_local_wise_qwk=("local_wise_qwk", "mean"),
            mean_pace_qwk=("pace_qwk", "mean"),
            mean_delta_qwk_vs_local=("delta_qwk_vs_local", "mean"),
            worst_delta_qwk_vs_local=("delta_qwk_vs_local", "min"),
            mean_delta_qwk_vs_old_ref=("delta_qwk_vs_old_ref", "mean"),
            mean_local_wise_mae=("local_wise_mae", "mean"),
            mean_pace_mae=("pace_mae", "mean"),
            mean_delta_mae=("delta_mae", "mean"),
            mean_local_wise_acc=("local_wise_acc", "mean"),
            mean_pace_acc=("pace_acc", "mean"),
            mean_delta_acc=("delta_acc", "mean"),
            mean_improved_share=("improved_share", "mean"),
            mean_worsened_share=("worsened_share", "mean"),
        )
        .sort_values(
            ["mean_delta_qwk_vs_local", "mean_delta_mae", "worst_delta_qwk_vs_local"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )
    setting_summary.to_csv(sweep_root / "summary_by_setting.csv", index=False)
    return out


def main() -> int:
    args = _parse_cli()
    logs_root = Path(args.logs_root)
    sweep_root = Path(args.base_out_dir) / args.sweep_name
    sweep_root.mkdir(parents=True, exist_ok=True)

    for setting in DEFAULT_GRID:
        _write_setting_manifest(sweep_root / setting["id"] / "sweep_setting.json", setting)

    if not args.aggregate_only:
        for setting in DEFAULT_GRID:
            setting_out_dir = sweep_root / setting["id"]
            for prompt in args.prompts:
                for fold in args.folds:
                    try:
                        _train_one(
                            python_bin=args.python_bin,
                            setting=setting,
                            prompt=prompt,
                            fold=fold,
                            args=args,
                            setting_out_dir=setting_out_dir,
                        )
                    except subprocess.CalledProcessError as exc:
                        print(
                            f"[sweep] ERROR setting={setting['id']} prompt={prompt} fold={fold}: "
                            f"returncode={exc.returncode}"
                        )
                        if not args.continue_on_error:
                            raise

    out = _summarize_sweep(sweep_root, logs_root)
    if out.empty:
        print(f"[sweep] No completed runs found under {sweep_root}")
    else:
        print(f"[sweep] Wrote {sweep_root / 'summary_by_run.csv'}")
        print(f"[sweep] Wrote {sweep_root / 'best_by_prompt_fold.csv'}")
        print(f"[sweep] Wrote {sweep_root / 'summary_by_setting.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
