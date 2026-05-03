"""Shared helpers for local WISE-PACE baseline scripts."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import yaml

import wise_aes
from pace.llm_backend import LocalLlamaBackend
from wise_aes import EvolutionOptimizer, PromptIndividual


class BaselineExpManager:
    def __init__(self, config: Dict):
        self.config = config
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0

    def log_llm_trace(self, _record: Dict) -> None:
        return

    def track_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.total_prompt_tokens += int(prompt_tokens)
        self.total_completion_tokens += int(completion_tokens)
        self.total_tokens += int(prompt_tokens) + int(completion_tokens)

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def load_config(path: str | Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_debug_split(config: Dict, fold: int = 0) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    df = pd.read_csv(config["data"]["asap_path"], sep="\t", encoding="latin-1")
    df = df[df["essay_set"] == config["data"]["essay_set"]]
    all_data = [
        {
            "essay_id": int(row["essay_id"]),
            "essay_text": row["essay"],
            "domain1_score": int(row["domain1_score"]),
        }
        for _, row in df.iterrows()
    ]
    debug = config.get("debug", {})
    seed = debug.get("seed", 42 + fold)
    n_train = debug.get("n_train", 240)
    n_val = debug.get("n_val", 32)
    n_test = debug.get("n_test", 32)
    if debug.get("stratified", True):
        try:
            return wise_aes._stratified_debug_split(
                all_data,
                n_train,
                n_val,
                n_test,
                seed,
                config["data"]["score_min"],
                config["data"]["score_max"],
            )
        except Exception as exc:
            print(f"[Baseline] Stratified split failed ({exc}); falling back to seeded shuffle.")
    random.Random(seed).shuffle(all_data)
    return (
        all_data[:n_train],
        all_data[n_train : n_train + n_val],
        all_data[n_train + n_val : n_train + n_val + n_test],
    )


def init_local_runtime(config: Dict) -> LocalLlamaBackend:
    wise_aes.EXP_MANAGER = BaselineExpManager(config)
    pace_cfg = config.get("pace", {})
    backend = LocalLlamaBackend(
        config=config,
        model_path=pace_cfg.get("model_path", "models/Meta-Llama-3.1-8B-Instruct"),
        dtype=pace_cfg.get("dtype", "bfloat16"),
        load_in_4bit=pace_cfg.get("load_in_4bit", False),
    )
    wise_aes.LOCAL_BACKEND = backend
    return backend


def fixed_anchors(train_set: List[Dict], val_set: List[Dict], config: Dict) -> List[Dict]:
    optimizer = EvolutionOptimizer(train_set, val_set, config)
    return optimizer.get_stratified_exemplars(config["evolution"].get("n_static_exemplars", 3))


def evaluate_instruction(
    *,
    instruction: str,
    anchors: List[Dict],
    val_set: List[Dict],
    config: Dict,
) -> float:
    individual = PromptIndividual(instruction, anchors, config=config)
    optimizer = EvolutionOptimizer([], val_set, config)
    return individual.evaluate(val_set, optimizer.vector_store, enable_rerank=False)


def generate_text(prompt: str, call_type: str) -> str:
    return wise_aes._call_local_generate(prompt, call_type=call_type)


def parse_numbered_candidates(text: str) -> List[str]:
    candidates = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        line = line.lstrip("-* ")
        if "." in line[:4]:
            line = line.split(".", 1)[1].strip()
        if len(line) >= 40:
            candidates.append(line)
    if candidates:
        return candidates
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(x).strip() for x in parsed if len(str(x).strip()) >= 40]
    except Exception:
        pass
    return [text.strip()] if len(text.strip()) >= 40 else []


def save_result(output_dir: Path, name: str, payload: Dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
