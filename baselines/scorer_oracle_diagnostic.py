"""Run oracle-ceiling diagnostics across scorer LLMs in separate processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase4_evidence_mutation.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--model-path", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/baselines/scorer_diagnostic"))
    args = parser.parse_args()

    script = Path(__file__).with_name("oracle_ceiling.py")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model_path in args.model_path:
        cmd = [
            sys.executable,
            str(script),
            "--config",
            args.config,
            "--fold",
            str(args.fold),
            "--model-path",
            model_path,
            "--output-dir",
            str(args.output_dir),
        ]
        print("RUN", " ".join(cmd), flush=True)
        completed = subprocess.run(cmd, check=False)
        if completed.returncode != 0:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
