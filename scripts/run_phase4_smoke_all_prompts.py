#!/usr/bin/env python
"""Run or preview Phase-4 smoke checks across ASAP prompts.

The launcher derives score_min/score_max from the ASAP TSV for each prompt and
writes per-prompt generated configs under configs/_generated/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def derive_score_range(asap_path: Path, prompt: int) -> Tuple[int, int]:
    if not asap_path.exists():
        raise FileNotFoundError(f"ASAP data file not found: {asap_path}")
    df = pd.read_csv(asap_path, sep="\t", encoding="latin1")
    subset = df[df["essay_set"].astype(int) == int(prompt)]
    if subset.empty:
        raise ValueError(f"No rows for essay_set={prompt} in {asap_path}")
    scores = subset["domain1_score"].astype(int)
    return int(scores.min()), int(scores.max())


def write_prompt_config(template_path: Path, prompt: int, score_range: Tuple[int, int]) -> Path:
    cfg = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    cfg.setdefault("data", {})
    cfg["data"]["essay_set"] = int(prompt)
    cfg["data"]["score_min"] = int(score_range[0])
    cfg["data"]["score_max"] = int(score_range[1])
    out_dir = REPO_ROOT / "configs" / "_generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{template_path.stem}_p{prompt}.yaml"
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_path


def newest_exp_dir(before: set[Path], logs_root: Path) -> Path | None:
    after = {p for p in logs_root.glob("exp_*") if p.is_dir()}
    created = sorted(after - before, key=lambda p: p.stat().st_mtime, reverse=True)
    return created[0] if created else None


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="+", type=int, default=list(range(1, 9)))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--config-template", default="configs/phase4_smoke.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    template_path = (REPO_ROOT / args.config_template).resolve()
    cfg_template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    asap_path = (REPO_ROOT / cfg_template["data"]["asap_path"]).resolve()
    logs_root = REPO_ROOT / "logs"

    for prompt in args.prompts:
        score_range = derive_score_range(asap_path, prompt)
        cfg_path = (
            REPO_ROOT / "configs" / "_generated" / f"{template_path.stem}_p{prompt}.yaml"
            if args.dry_run
            else write_prompt_config(template_path, prompt, score_range)
        )
        cmd = [
            sys.executable,
            "wise_aes.py",
            "--config",
            str(cfg_path.relative_to(REPO_ROOT)),
            "--fold",
            str(args.fold),
        ]
        print(f"[P{prompt}] score_range={score_range} command={' '.join(cmd)}")
        if args.dry_run:
            print(f"[P{prompt}] dry_run_config_would_be={cfg_path}")
            continue
        before = {p for p in logs_root.glob("exp_*") if p.is_dir()}
        proc = subprocess.run(cmd, cwd=REPO_ROOT)
        exp_dir = newest_exp_dir(before, logs_root)
        print(f"[P{prompt}] returncode={proc.returncode} exp_dir={exp_dir}")
        if proc.returncode != 0:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
