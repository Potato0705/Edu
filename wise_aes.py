"""
WISE-AES: Weakly-supervised Integrated Scoring Evolution
版本: v3.0 (Enhanced with CoT, JSON Parsing, Real Reflection, Stratified Sampling)
"""

import os
import sys
import re
import csv
import math
import random
import json
import time
import hashlib
import sqlite3
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

# 第三方库
from dotenv import load_dotenv
import requests
import numpy as np
import yaml
import pandas as pd
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer # [NEW] For precise token counting

# WISE-PACE 可选依赖（pace.enabled=false 时不加载，向后兼容）
try:
    from pace.llm_backend import LocalLlamaBackend, ScoringRequest
    from pace.pace_fitness import PaceFitnessConfig, PaceFitnessEvaluator
    _PACE_AVAILABLE = True
except ImportError:
    _PACE_AVAILABLE = False

load_dotenv(override=True)

# ============================================================================
# 0. 基础设施: 双向日志系统 & 实验管理
# ============================================================================

class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a", encoding='utf-8')
        self.lock = threading.Lock()
    def write(self, message):
        with self.lock:
            self.terminal.write(message)
            self.log_file.write(message)
            self.log_file.flush()
    def flush(self):
        with self.lock:
            self.terminal.flush()
            self.log_file.flush()

class ExperimentManager:
    def __init__(self, base_dir="logs", config_path="configs/default.yaml", fold=0, resume_path=None):
        if resume_path:
            # Resume 模式：复用旧的实验目录
            gen_file = Path(resume_path)
            # 假设结构是 logs/exp_xxx/generations/gen_xxx.json
            self.exp_dir = gen_file.parent.parent
            self.timestamp = self.exp_dir.name.split('_')[1] # 尝试提取，仅用于显示
            print(f"=== Experiment Resumed from {resume_path} ===")
        else:
            # 新实验模式
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_dir = Path(base_dir) / f"exp_{self.timestamp}_fold{fold}"
            print(f"=== Experiment Started: {self.timestamp} (Fold {fold}) ===")
        
        self.gens_dir = self.exp_dir / "generations"
        self.gens_dir.mkdir(parents=True, exist_ok=True)
        
        # 追加模式打开日志
        self.console_log_path = self.exp_dir / f"console.log"
        sys.stdout = TeeLogger(self.console_log_path)
        
        print(f"Config File: {config_path}")
        print(f"Result Directory: {self.exp_dir}")
        
        self.llm_trace_path = self.exp_dir / "llm_trace.jsonl"
        # 如果是 Resume，尽量复用原配置，但这里允许用新 Config 覆盖参数（为了灵活性）
        self.config = self._load_and_save_config(config_path)
        self.lock = threading.Lock()
        
        # [NEW] Performance Metrics Tracking
        self.exp_start_time = time.time()
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.gen_metrics = {}
        
        # [NEW] Initialize Tokenizer
        model_name_conf = self.config.get('model', {}).get('name', 'gpt2')
        print(f"[Tokenizer] Initializing tokenizer for: {model_name_conf}...")
        try:
            # 尝试处理 OpenRouter 格式 (vendor/model)
            if "/" in model_name_conf:
                # 尝试直接加载 (有些是 huggingface ID)
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name_conf, trust_remote_code=True)
                except:
                    # 尝试去掉 vendor 前缀
                    clean_name = model_name_conf.split("/", 1)[1]
                    print(f"[Tokenizer] Retry with: {clean_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(clean_name, trust_remote_code=True)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name_conf, trust_remote_code=True)
        except Exception as e:
            print(f"[Tokenizer] Warning: Failed to load specific tokenizer ({e}). Fallback to 'gpt2'.")
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        print(f"[Tokenizer] Loaded: {self.tokenizer.name_or_path}")

    def _load_and_save_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with open(self.exp_dir / "config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        return config

    def log_llm_trace(self, record: Dict):
        with self.lock:
            with open(self.llm_trace_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def track_usage(self, prompt_tokens, completion_tokens):
        with self.lock:
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_tokens += (prompt_tokens + completion_tokens)

    def count_tokens(self, text: str) -> int:
        # Tokenizer encode might not be thread-safe depending on backend? 
        # Usually it is, but to be safe we can lock if needed, though it slows down.
        # Transformers fast tokenizers are generally safe.
        try:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        except:
            return len(text) // 4 # Fallback

    def save_generation_snapshot(self, generation: int, population_data: List[Dict], metrics: Dict = None):
        filename = self.gens_dir / f"gen_{generation:03d}.json"
        
        snapshot = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics or {},
            "best_qwk": max(p['fitness'] for p in population_data),
            "population": population_data
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def save_training_curve(self, history: List[Dict]):
        if not history:
            return

        json_path = self.exp_dir / "training_curve.json"
        csv_path = self.exp_dir / "training_curve.csv"
        svg_path = self.exp_dir / "training_curve.svg"

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        preferred = [
            "gen",
            "best_raw_val",
            "best_raw_guarded",
            "best_pareto",
            "best_protocol_quality",
            "best_pace_guarded",
            "best_pace_qwk",
            "best_anchor_geometry",
            "max_overfit_gap",
            "max_overfit_penalty",
            "max_distribution_tv",
            "max_distribution_penalty",
            "mean_parent_child_raw_delta",
            "positive_parent_child_delta_rate",
            "raw_adjusted_best_idx",
            "pareto_best_idx",
            "pareto_feasible_count",
            "raw_guard_triggered_count",
            "pace_evaluated_count",
            "duration_sec",
            "tokens_total_all",
        ]
        all_keys = []
        for row in history:
            for key in row.keys():
                if key not in all_keys:
                    all_keys.append(key)
        fieldnames = [k for k in preferred if k in all_keys] + [
            k for k in all_keys if k not in preferred
        ]
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in history:
                writer.writerow(row)

        self._save_training_curve_svg(history, svg_path)

    def _save_training_curve_svg(self, history: List[Dict], svg_path: Path):
        series = [
            ("best_raw_val", "#2563eb"),
            ("best_raw_guarded", "#f97316"),
            ("best_pareto", "#059669"),
            ("best_pace_guarded", "#7c3aed"),
            ("mean_parent_child_raw_delta", "#dc2626"),
        ]
        rows = [
            row for row in history
            if any(isinstance(row.get(name), (int, float)) for name, _ in series)
        ]
        if not rows:
            return

        width, height = 760, 360
        left, right, top, bottom = 56, 24, 24, 54
        plot_w = width - left - right
        plot_h = height - top - bottom
        gens = [int(row.get("gen", i + 1)) for i, row in enumerate(rows)]
        x_min, x_max = min(gens), max(gens)
        if x_min == x_max:
            x_min -= 1
            x_max += 1

        values = []
        for row in rows:
            for name, _ in series:
                val = row.get(name)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    values.append(float(val))
        if not values:
            return
        y_min = min(-0.05, min(values) - 0.03)
        y_max = max(1.0, max(values) + 0.03)
        if y_min == y_max:
            y_min -= 0.1
            y_max += 0.1

        def x_pos(gen):
            return left + (float(gen) - x_min) / (x_max - x_min) * plot_w

        def y_pos(value):
            return top + (y_max - float(value)) / (y_max - y_min) * plot_h

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#94a3b8"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#94a3b8"/>',
            f'<text x="{left}" y="18" font-family="Arial" font-size="14" fill="#0f172a">WISE-PACE Training Curve</text>',
        ]
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            y = top + plot_h * frac
            val = y_max - (y_max - y_min) * frac
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e2e8f0"/>')
            svg.append(f'<text x="10" y="{y + 4:.1f}" font-family="Arial" font-size="11" fill="#475569">{val:.2f}</text>')
        for name, color in series:
            pts = []
            for gen, row in zip(gens, rows):
                val = row.get(name)
                if isinstance(val, (int, float)) and np.isfinite(val):
                    pts.append(f'{x_pos(gen):.1f},{y_pos(val):.1f}')
            if len(pts) >= 2:
                svg.append(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            elif len(pts) == 1:
                x, y = pts[0].split(",")
                svg.append(f'<circle cx="{x}" cy="{y}" r="3" fill="{color}"/>')
        legend_x = left
        legend_y = height - 24
        for name, color in series:
            svg.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
            svg.append(f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#334155">{name}</text>')
            legend_x += 180
        svg.append(f'<text x="{left + plot_w / 2 - 30:.1f}" y="{height - 8}" font-family="Arial" font-size="11" fill="#475569">generation</text>')
        svg.append('</svg>')
        with open(svg_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(svg))

    def save_anchor_mutations(self, generation: int, events: List[Dict]):
        filename = self.gens_dir / f"anchor_mutations_gen_{generation:03d}.json"
        payload = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "events": events,
        }
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def save_final_results(self, best_ind, history, test_results=None):
        if best_ind is None:
            best_payload = None
        else:
            best_payload = {
                "best_qwk_val": best_ind.fitness,
                "instruction": best_ind.instruction_text,
                "static_exemplars": [ex['essay_id'] for ex in best_ind.static_exemplars],
                "static_exemplar_scores": [ex['domain1_score'] for ex in best_ind.static_exemplars],
            }
        res = {
            **(best_payload or {}),
            "primary_candidate": test_results.get("primary_candidate") if test_results else None,
            "history": history,
            "test_results": test_results
        }
        with open(self.exp_dir / "final_result.json", 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nFinal results saved to {self.exp_dir}")

EXP_MANAGER = None
LOCAL_BACKEND = None  # type: Optional[LocalLlamaBackend]  # 由 main() 在 pace.enabled=true 时初始化

# ============================================================================
# 1. 基础设施: 向量数据库
# ============================================================================
class SimpleVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "cache"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.embeddings: Optional[np.ndarray] = None
        self.model = None 
        self.lock = threading.Lock()

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def add_documents(self, data: List[Dict[str, Any]]):
        self.documents = data
        if not data: return
        doc_ids = [str(d['essay_id']) for d in data]
        ids_fingerprint = hashlib.md5(str(sorted(doc_ids)).encode('utf-8')).hexdigest()[:10]
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}_{len(data)}_{ids_fingerprint}.pkl"
        
        if cache_file.exists():
            print(f"[VectorStore] Loading cached embeddings from {cache_file.name}")
            with open(cache_file, 'rb') as f:
                saved_ids, self.embeddings = pickle.load(f)
        else:
            self._compute_and_save(data, doc_ids, cache_file)
            
    def _compute_and_save(self, data, doc_ids, cache_file):
        self._load_model()
        texts = [d['essay_text'] for d in data]
        print(f"[VectorStore] Encoding {len(texts)} documents...")
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        with open(cache_file, 'wb') as f:
            pickle.dump((doc_ids, self.embeddings), f)
            
    def search(self, query: str, top_k: int = 10, exclude_ids: set = None) -> List[Dict]:
        with self.lock:
            self._load_model()
            if self.embeddings is None or len(self.documents) == 0: return []
            
            query_vec = self.model.encode([query], convert_to_numpy=True)
            sims = cosine_similarity(query_vec, self.embeddings)[0]
        
        sorted_indices = np.argsort(sims)[::-1]
        results = []
        for idx in sorted_indices:
            doc = self.documents[idx]
            if exclude_ids and doc['essay_id'] in exclude_ids: continue
            results.append(doc)
            if len(results) >= top_k: break
        return results

# ============================================================================
# 2. LLM 缓存与并发调用
# ============================================================================
class LLMCache:
    def __init__(self, db_path: str = "cache/llm_cache.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path, timeout=60) as conn:
            conn.execute("PRAGMA journal_mode=WAL;") 
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    hash_key TEXT PRIMARY KEY, 
                    prompt TEXT, response TEXT, timestamp TEXT
                )
            """)

    def _get_hash(self, prompt: str, model: str, temperature: float) -> str:
        content = f"{model}_{temperature}_{prompt}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def get(self, prompt: str, model: str, temperature: float) -> Optional[str]:
        key = self._get_hash(prompt, model, temperature)
        try:
            with self.lock:
                with sqlite3.connect(self.db_path, timeout=60) as conn:
                    cursor = conn.execute("SELECT response FROM cache WHERE hash_key=?", (key,))
                    row = cursor.fetchone()
                    if row: return row[0]
        except Exception: return None
        return None

    def set(self, prompt: str, model: str, temperature: float, response: str):
        key = self._get_hash(prompt, model, temperature)
        try:
            with self.lock:
                with sqlite3.connect(self.db_path, timeout=60) as conn:
                    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)", 
                                 (key, prompt, response, datetime.now().isoformat()))
        except Exception as e: print(f"[Cache Write Error] {e}")

CACHE = LLMCache()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def call_llm(prompt: str, temperature: float = 0.0, call_type: str = "unknown") -> str:
    # 1. 获取配置
    model_config = EXP_MANAGER.config.get('model', {})
    model_name = model_config.get('name', 'deepseek/deepseek-chat')
    provider_setting = model_config.get('provider', None) 
    
    cached_resp = CACHE.get(prompt, model_name, temperature)
    if cached_resp: return cached_resp

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}", 
        "Content-Type": "application/json", 
        "HTTP-Referer": "https://wise-aes.local"
    }
    
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    
    # [NEW] 注入 Provider 配置
    if provider_setting:
        payload["provider"] = {
            "order": [provider_setting],
            "allow_fallbacks": False
        }
    
    start_time = time.time()
    response_content = ""
    error_msg = ""
    last_status = None
    last_body_preview = ""
    
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            last_status = resp.status_code
            last_body_preview = resp.text[:500]
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            data = resp.json()
            if "choices" in data and len(data["choices"]) > 0:
                content = data["choices"][0]["message"].get("content", "")
                if isinstance(content, list):
                    content = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                if isinstance(content, str) and content.strip(): # Ensure content is not empty or just whitespace
                    response_content = content
                    CACHE.set(prompt, model_name, temperature, response_content)
                    break
                else:
                    error_msg = f"Empty content in response: {data}"
            else:
                error_msg = f"Empty response structure: {data}"
        except Exception as e:
            error_msg = (
                f"{type(e).__name__}: {e}"
                f" | status={last_status}"
                f" | body={last_body_preview}"
            )
            time.sleep(1)
    
    
    # [NEW] Track Metrics
    if EXP_MANAGER:
        p_tokens = EXP_MANAGER.count_tokens(prompt)
        r_tokens = EXP_MANAGER.count_tokens(response_content)
        EXP_MANAGER.track_usage(p_tokens, r_tokens)

    if EXP_MANAGER:
        EXP_MANAGER.log_llm_trace({
            "timestamp": datetime.now().isoformat(),
            "call_type": call_type,
            "provider": provider_setting,
            "duration": round(time.time() - start_time, 3),
            "len": len(response_content),
            "len": len(response_content),
            "error": error_msg,
            "prompt_preview": prompt[:100],
            "response_preview": response_content[:200]
        })
    if not response_content.strip():
        raise RuntimeError(
            f"LLM call failed for {call_type} after 3 attempts. "
            f"model={model_name} provider={provider_setting} error={error_msg}"
        )
    return response_content


def _call_local_generate(prompt: str, call_type: str = "unknown") -> str:
    """用本地 Llama 模型做纯文本生成（reflection / rewrite / induction），不需要 hidden state。

    LOCAL_BACKEND 为 None 时（pace.enabled=false）回退到 call_llm（向后兼容）。
    """
    if LOCAL_BACKEND is None:
        return call_llm(prompt, temperature=0.7, call_type=call_type)
    chat_prompt = LOCAL_BACKEND._apply_chat_template(prompt)
    with LOCAL_BACKEND._lock:
        text, _ = LOCAL_BACKEND._generate(chat_prompt)
    LOCAL_BACKEND.record_generation_usage(chat_prompt, text)
    if EXP_MANAGER:
        p_tokens = EXP_MANAGER.count_tokens(chat_prompt)
        r_tokens = EXP_MANAGER.count_tokens(text)
        EXP_MANAGER.track_usage(p_tokens, r_tokens)
    return text


def _set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _prepend_score_range_contract(text: str, config: Dict[str, Any]) -> str:
    score_min = config['data']['score_min']
    score_max = config['data']['score_max']
    contract = (
        "Score Range Contract: Use only integer final_score values from "
        f"{score_min} to {score_max}, inclusive. Do not use incompatible "
        "point totals, percentages, letter grades, or decimal scores."
    )
    head = text[:600].lower()
    if "score range contract" in head and str(score_min) in head and str(score_max) in head:
        return text.strip()
    return f"{contract}\n\n{text.strip()}"


def _sum_generation_metric(gens_dir: Path, key: str) -> int:
    total = 0
    for path in sorted(gens_dir.glob("gen_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                snap = json.load(f)
            total += int(snap.get("metrics", {}).get(key, 0) or 0)
        except Exception:
            continue
    return total


def _score_distribution(items: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in items:
        key = str(int(item["domain1_score"]))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: int(kv[0])))


def _score_band_label(score: int, score_min: int, score_max: int) -> str:
    low_max = math.floor(score_min + (score_max - score_min) / 3.0)
    high_min = math.ceil(score_min + 2.0 * (score_max - score_min) / 3.0)
    if score <= low_max:
        return "low"
    if score >= high_min:
        return "high"
    return "mid"


def _score_band_labels(items: List[Dict], score_min: int, score_max: int) -> List[str]:
    return [
        _score_band_label(int(item["domain1_score"]), score_min, score_max)
        for item in items
    ]


def _score_band_distribution(items: List[Dict], score_min: int, score_max: int) -> Dict[str, int]:
    counts = {"low": 0, "mid": 0, "high": 0}
    for label in _score_band_labels(items, score_min, score_max):
        counts[label] += 1
    return counts


def _stratified_debug_split(
    all_data: List[Dict],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
    score_min: int,
    score_max: int,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    labels = _score_band_labels(all_data, score_min, score_max)
    train_val, test_set = train_test_split(
        all_data,
        train_size=n_train + n_val,
        test_size=n_test,
        stratify=labels,
        random_state=seed,
    )
    train_val_labels = _score_band_labels(train_val, score_min, score_max)
    train_set, val_set = train_test_split(
        train_val,
        train_size=n_train,
        test_size=n_val,
        stratify=train_val_labels,
        random_state=seed + 1,
    )
    return list(train_set), list(val_set), list(test_set)


# ============================================================================
# 3. PromptIndividual (包含核心修改)
# ============================================================================

@dataclass
class PromptIndividual:
    instruction_text: str
    static_exemplars: List[Dict[str, Any]] = field(default_factory=list)
    fitness: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    evidence_feedback: Dict[str, Any] = field(default_factory=dict)
    
    last_pred_scores: List[int] = field(default_factory=list)

    # Templates
    QUERY_GEN_RUBRIC_TEMPLATE = """Based on the SCORING RUBRIC below, extract 3-5 keywords from the STUDENT ESSAY.\nSCORING RUBRIC:\n{rubric}\nSTUDENT ESSAY:\n{essay}\nOUTPUT JSON LIST:"""
    QUERY_GEN_GENERIC_TEMPLATE = """Extract 3-5 distinct keywords from the STUDENT ESSAY.\nSTUDENT ESSAY:\n{essay}\nOUTPUT JSON LIST:"""
    RERANK_TEMPLATE = """Select the {k} essays from CANDIDATES that best match the Rubric criteria to serve as references for grading the Target Essay.

RUBRIC:
{rubric}

TARGET ESSAY:
{essay}

CANDIDATES:
{candidates}

Output ONLY a JSON list of essay_ids (e.g., [101, 204])."""

    # CoT + JSON SCORING TEMPLATE
    SCORING_TEMPLATE = """You are an expert essay grader. 
Your goal is to evaluate the essay based on the SCORING RUBRIC and REFERENCE EXAMPLES.

Step 1: Analyze the essay's Coherence, Organization, and Grammar based on the rubric.
Step 2: Explicitly compare the essay to the Reference Examples provided (Anchors).
Step 3: Assign a final integer score between {score_min} and {score_max}.

## SCORE RANGE CONTRACT:
- The only valid final_score values are integers from {score_min} to {score_max}, inclusive.
- If the rubric text mentions an incompatible point scale, ignore that scale and map the evidence onto {score_min}-{score_max}.
- Do not output percentages, 60-point scores, 100-point scores, letter grades, or decimal scores.

## SCORING RUBRIC:
{instruction}

## REFERENCE EXAMPLES (Anchors):
{static_ex}

## RETRIEVED SIMILAR EXAMPLES (Local Context):
{dynamic_ex}

## ESSAY TO SCORE:
{essay}

Output your response in valid JSON format:
{{
  "reasoning": "The essay demonstrates...",
  "comparison": "Compared to the score {score_min} example, this essay is...",
  "final_score": <integer>
}}
"""

    INDUCTION_TEMPLATE = """You are an expert essay scoring system designer.
Your task is to create a refined, operational SCORING RUBRIC (I_0) based on the OFFICIAL CRITERIA and concrete STUDENT ESSAYS.

### SCORE RANGE CONTRACT:
The scoring system must output exactly one integer final_score in [{score_min}, {score_max}].
Every rule must be expressed for this {score_min}-{score_max} scale.
Do not invent or preserve incompatible point totals such as 60 points, 100 points, percentages, letter grades, or decimal scores.

### OFFICIAL CRITERIA:
{official_criteria}

### STUDENT ESSAY SAMPLES (Grounding Data):
{samples_text}

### INSTRUCTION:
1. Analyze the difference between High-Scoring and Low-Scoring essays.
2. Refine the OFFICIAL CRITERIA into a detailed, step-by-step scoring guide.
3. Highlight specific, observable discriminators (e.g., "use of transitions," "sentence variety").
4. Add boundary rules for adjacent score bands so the grader can distinguish near-miss essays.
5. The output must be ready to use as a prompt for an LLM grader.

Output ONLY the Rubric text. Do not output explanations."""

    def __post_init__(self):
        if not self.config: self.config = EXP_MANAGER.config

    def _score_range_contract(self) -> str:
        score_min = self.config['data']['score_min']
        score_max = self.config['data']['score_max']
        return (
            f"Use only integer final_score values in [{score_min}, {score_max}]. "
            f"All grading rules must be calibrated to the {score_min}-{score_max} scale. "
            "Remove incompatible point totals, percentages, letter grades, and decimal scores."
        )

    def get_signature(self):
        # Fingerprint includes ordered anchors because Phase 2 uses low/mid/high slots.
        ex_payload = [
            {"essay_id": str(ex["essay_id"]), "score": int(ex["domain1_score"])}
            for ex in self.static_exemplars
        ]
        content = json.dumps(
            {"instruction": self.instruction_text, "anchors": ex_payload},
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def generate_query(self, essay_text: str) -> str:
        # [NEW] 原文检索模式 (Raw Query Mode)
        if self.config['rag'].get('use_raw_query', False):
            # 直接返回前 512 个字符 (通常足够 SentenceTransformer 填满上下文)
            return essay_text[:512]

        rubric_driven = self.config['rag'].get('rubric_driven_retrieval', False)
        template = self.QUERY_GEN_RUBRIC_TEMPLATE if rubric_driven else self.QUERY_GEN_GENERIC_TEMPLATE
        prompt = template.format(rubric=self.instruction_text, essay=essay_text[:800])
        temp = self.config['llm'].get('temperature_query', 0.7)
        response = call_llm(prompt, temperature=temp, call_type="rag_query")
        try:
            keywords = json.loads(re.search(r'\[.*\]', response, re.DOTALL).group())
            return " ".join(keywords)
        except: return essay_text[:200]

    def rerank_exemplars(self, essay_text: str, candidates: List[Dict]) -> List[Dict]:
        k_select = self.config['rag']['n_selected']
        cand_text = "\n".join([f"ID {c['essay_id']} (Score {c['domain1_score']}): {c['essay_text'][:200]}..." for c in candidates])
        
        prompt = self.RERANK_TEMPLATE.format(
            k=k_select,
            rubric=self.instruction_text,
            essay=essay_text[:500], # 截取一部分以节省 Token
            candidates=cand_text
        )
        
        temp = self.config['llm'].get('temperature_rerank', 0.0)
        response = call_llm(prompt, temperature=temp, call_type="rag_rerank")
        try:
            ids = json.loads(re.search(r'\[.*\]', response, re.DOTALL).group())
            str_ids = set(str(x) for x in ids)
            selected = [c for c in candidates if str(c['essay_id']) in str_ids]
            if len(selected) < k_select:
                remain = [c for c in candidates if c not in selected]
                selected.extend(remain[:k_select - len(selected)])
            return selected[:k_select]
        except: 
            return candidates[:k_select]

    # [NEW] Robust JSON Extraction Helper
    def _extract_json_safe(self, text: str) -> Optional[Dict]:
        try:
            # 1. Try passing the whole text first
            return json.loads(text)
        except:
            pass
            
        # 2. Backward search for the last valid JSON block
        # Anchor to the last '}' and scan backwards for '{'
        last_brace = text.rfind('}')
        if last_brace == -1: return None
        
        # Iterate backwards to find the matching opening brace
        # We limit the search to avoid infinite loops in worst cases, though length check is naturally finite
        for i in range(last_brace, -1, -1):
            if text[i] == '{':
                candidate = text[i : last_brace+1]
                try:
                    obj = json.loads(candidate)
                    # Weak check: it should look like our expected output
                    if isinstance(obj, dict) and "final_score" in obj:
                        return obj
                except:
                    continue
        return None

    # [MODIFIED] 包含重试机制、配置读取和双向日志
    def predict_score(self, essay_text: str, vector_store: SimpleVectorStore, enable_rerank: bool = False, essay_id: int = 0) -> int:
        # 本地 PACE 模式：直接用 LocalLlamaBackend 评分，跳过 RAG 和 call_llm
        if LOCAL_BACKEND is not None:
            req = ScoringRequest(
                essay_id=essay_id,
                essay_text=essay_text,
                instruction=self.instruction_text,
                static_exemplars=self.static_exemplars,
                score_min=self.config['data']['score_min'],
                score_max=self.config['data']['score_max'],
                dynamic_ex="(None)",
            )
            result = LOCAL_BACKEND.score(req)
            if EXP_MANAGER:
                EXP_MANAGER.track_usage(
                    result.meta.get("prompt_tokens", 0),
                    result.meta.get("completion_tokens", 0),
                )
            return result.y_raw

        dynamic_exemplars = []
        if self.config['rag']['enabled']:
            query = self.generate_query(essay_text)
            exclude_ids = {ex['essay_id'] for ex in self.static_exemplars}
            candidates = vector_store.search(query, top_k=self.config['rag']['n_retrieved'], exclude_ids=exclude_ids)
            
            if enable_rerank:
                dynamic_exemplars = self.rerank_exemplars(essay_text, candidates)
            else:
                dynamic_exemplars = candidates[:self.config['rag']['n_selected']]
            
        base_prompt = self.SCORING_TEMPLATE.format(
            instruction=self.instruction_text,
            static_ex=self._format_list(self.static_exemplars),
            dynamic_ex=self._format_list(dynamic_exemplars) if dynamic_exemplars else "(None)",
            essay=essay_text,
            score_min=self.config['data']['score_min'],
            score_max=self.config['data']['score_max']
        )
        
        # [NEW] 从 Config 读取重试次数 (默认为 1)
        max_retries = self.config['llm'].get('max_retries', 1)
        current_prompt = base_prompt
        
        for attempt in range(max_retries + 1):
            # 重试时稍微提高温度 (0.3)，增加跳出局部错误的可能性
            temp = self.config['llm']['temperature_scoring'] if attempt == 0 else 0.3
            call_type = "scoring" if attempt == 0 else f"scoring_retry_{attempt}"
            
            response = call_llm(current_prompt, temperature=temp, call_type=call_type)
            
            try:
                # [MODIFIED] Use robust extraction instead of regex
                result = self._extract_json_safe(response)
                
                if result:
                    score = int(result.get('final_score', -1))
                    
                    if self.config['data']['score_min'] <= score <= self.config['data']['score_max']:
                        return score
                    else:
                        raise ValueError(f"Score {score} out of range")
                else:
                    raise ValueError("No valid JSON found")
            
            except Exception as e:
                # 如果这是最后一次尝试，跳出循环进入 Fallback
                if attempt == max_retries:
                    # [Log] 记录最终失败，将进入 Fallback
                    # print(f"    [Warning] JSON parsing failed after {attempt+1} attempts: {e}. Falling back to Regex.")
                    break
                

                # [Log] 打印重试日志 (TeeLogger 会同时输出到屏幕和文件)
                print(f"    [Retry] Attempt {attempt+1}/{max_retries} failed ({e}). Retrying...")
                if self.config.get('output', {}).get('verbose', False):
                    print(f"      > Debug: Response was: {response[:100].replace(chr(10), ' ')}...")
                
                # 简单重试 (通过添加无意义空格改变 Hash Key，避免命中缓存)
                current_prompt = base_prompt + " " * (attempt + 1)

        # --- Fallback 1: Regex (更稳健的倒序查找) ---
        try:
            nums = re.findall(r'-?\d+', response)
            if nums:
                for num_str in reversed(nums):
                    val = int(num_str)
                    if self.config['data']['score_min'] <= val <= self.config['data']['score_max']:
                        # print(f"    [Fallback] Recovered score {val} via Regex.")
                        return val
        except: pass
        
        # --- Fallback 2: Median ---
        return (self.config['data']['score_min'] + self.config['data']['score_max']) // 2

    def _format_list(self, exs):
        if not exs:
            return "(None)"
        roles = (
            ["Low anchor", "Middle anchor", "High anchor"]
            if len(exs) <= 3
            else ["Low anchor", "Lower boundary anchor", "Upper boundary anchor", "High anchor"]
        )
        blocks = []
        for idx, ex in enumerate(exs):
            role = roles[idx] if idx < len(roles) else f"Additional anchor {idx + 1}"
            blocks.append(
                f"### {role} (Known Score: {ex['domain1_score']})\n"
                "Use this as a global scoring-scale reference, not as a retrieved neighbor.\n"
                f"Essay: {ex['essay_text'][:400]}..."
            )
        return "\n\n".join(blocks)

    def evaluate(self, val_set: List[Dict], vector_store: SimpleVectorStore, enable_rerank: bool = False, fitness_cache=None) -> float:
        # [Optimization] 1. 检查缓存
        sig = self.get_signature()
        if fitness_cache is not None and sig in fitness_cache:
            print(f"    [Cache Hit] Skipping evaluation for {sig[:8]}...")
            cached = fitness_cache[sig]
            if isinstance(cached, dict):
                self.fitness = float(cached.get("qwk", 0.0))
                self.last_pred_scores = list(cached.get("pred_scores", []))
            else:
                self.fitness = float(cached)
            return self.fitness
        
        true_scores = [item['domain1_score'] for item in val_set]
        pred_scores = [0] * len(val_set)
        max_workers = self.config['evolution'].get('max_workers', 5)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.predict_score, item['essay_text'], vector_store, enable_rerank, item.get('essay_id', i)): i
                for i, item in enumerate(val_set)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try: 
                    pred = future.result()
                    pred_scores[idx] = pred
                    essay_id = val_set[idx].get('essay_id', 'unknown')
                    mode = "Rerank" if enable_rerank else "Vector"
                    if self.config.get('output', {}).get('verbose', False):
                        print(f"    [Eval-{mode}] ID {essay_id:<4} | Truth: {true_scores[idx]:<2} | Pred: {pred:<2}")
                except Exception as e: 
                    print(f"    [Error] {e}")
                    pred_scores[idx] = (self.config['data']['score_min'] + self.config['data']['score_max']) // 2
                    
        self.last_pred_scores = pred_scores
        qwk = cohen_kappa_score(true_scores, pred_scores, weights='quadratic')
        if not np.isfinite(qwk):
            qwk = 0.0
        qwk = float(qwk)
        self.fitness = qwk
        
        # [Optimization] 2. 写入缓存
        if fitness_cache is not None:
            fitness_cache[sig] = {
                "qwk": qwk,
                "pred_scores": list(pred_scores),
            }
            
        return qwk

    def clone(self):
        cloned = PromptIndividual(
            self.instruction_text,
            self.static_exemplars.copy(),
            self.fitness,
            self.config,
            dict(self.evidence_feedback),
        )
        cloned.last_pred_scores = list(self.last_pred_scores)
        if hasattr(self, "parent_trace"):
            cloned.parent_trace = dict(getattr(self, "parent_trace"))
        return cloned
    
    def generate_real_feedback(self, val_set: List[Dict]) -> str:
        if not self.last_pred_scores: return "Focus on improving general accuracy."
        true_scores = [x['domain1_score'] for x in val_set]
        score_min = self.config['data']['score_min']
        score_max = self.config['data']['score_max']

        def score_dist(vals):
            counts = {str(s): 0 for s in range(score_min, score_max + 1)}
            for val in vals:
                key = str(int(val))
                if key in counts:
                    counts[key] += 1
            return counts

        score_errors = [p - t for p, t in zip(self.last_pred_scores, true_scores)]
        mean_bias = float(np.mean(score_errors)) if score_errors else 0.0
        mae = float(np.mean([abs(x) for x in score_errors])) if score_errors else 0.0
        pred_dist = score_dist(self.last_pred_scores)
        true_dist = score_dist(true_scores)
        collapse_ratio = (
            max(pred_dist.values()) / max(1, len(self.last_pred_scores))
            if pred_dist else 0.0
        )
        distribution_feedback = {
            "mean_pred_minus_true": round(mean_bias, 4),
            "mae": round(mae, 4),
            "pred_distribution": pred_dist,
            "true_distribution": true_dist,
            "prediction_collapse_ratio": round(collapse_ratio, 4),
        }
        errors = []
        for i, (p, t) in enumerate(zip(self.last_pred_scores, true_scores)):
            if abs(p - t) >= 1: 
                errors.append((val_set[i], p, t))
        
        errors.sort(key=lambda item: abs(item[1] - item[2]), reverse=True)
        selected_errors = errors[: self.config.get('evolution', {}).get('n_error_cases', 3)]
        if not selected_errors: return "No significant errors found."

        error_desc = "\n".join([
            f"- Essay (ID {e['essay_id']}): Predicted {p}, but Truth was {t}. \nSnippet: {e['essay_text'][:200]}..."
            for e, p, t in selected_errors
        ])
        
        evidence_block = ""
        if self.evidence_feedback:
            evidence_block = "\nEVIDENCE DIAGNOSTICS:\n" + json.dumps(
                self.evidence_feedback,
                ensure_ascii=False,
                indent=2,
            )[:2500] + "\n"

        score_contract = self._score_range_contract()
        prompt = f"""Review the grading errors below and suggest improvements to the SCORING RUBRIC.

SCORE RANGE CONTRACT:
{score_contract}

CURRENT RUBRIC:
{self.instruction_text}
{evidence_block}
VALIDATION DISTRIBUTION DIAGNOSTICS:
{json.dumps(distribution_feedback, ensure_ascii=False, indent=2)}

ERRORS:
{error_desc}

Analyze WHY these errors occurred. Use the evidence diagnostics when present:
- If hidden evidence is closer to low anchors but the raw score is high, add stricter deduction rules.
- If hidden evidence is closer to high anchors but the raw score is low, add clearer high-score recognition rules.
- If anchor confusion or boundary ambiguity is dominant, sharpen adjacent-score boundary rules.
- If reasoning-score contradiction appears, require the final score to match the stated strengths and weaknesses.
- If mean_pred_minus_true is strongly negative, add explicit recognition rules for high-quality essays and reduce overly harsh deductions.
- If predictions collapse into one or two scores, add score-band diversity and boundary calibration rules.
- If pace_teacher_guidance is present, convert it into direct-scoring rubric edits; do not rely on a calibrator to fix raw scores.

Return actionable rewrite guidance only. Preserve the score range contract.
"""
        temp = self.config['llm'].get('temperature_induce', 0.8)
        if LOCAL_BACKEND is not None:
            return _call_local_generate(prompt, call_type="reflection")
        return call_llm(prompt, temperature=temp, call_type="reflection")

    def evolve_instruction(self, feedback):
        score_contract = self._score_range_contract()
        prompt = f"""Rewrite and improve the rubric based on the feedback.

SCORE RANGE CONTRACT:
{score_contract}

FEEDBACK:
{feedback}

OLD RUBRIC:
{self.instruction_text}

Rewrite requirements:
- Preserve the exact valid score range.
- Remove or rewrite any incompatible point totals, percentages, letter grades, or decimal scoring language.
- Add concrete boundary rules for adjacent score levels when the feedback mentions boundary ambiguity.
- Add hidden-evidence-aware corrections when the feedback mentions over-score, under-score, anchor confusion, or reasoning-score contradiction.
- Translate PACE teacher guidance into concrete score-band and boundary rules, but optimize direct raw final_score quality rather than post-hoc calibration.
- Keep the rubric concise enough to fit in an LLM grading prompt.

OUTPUT ONLY THE NEW RUBRIC TEXT."""
        temp = self.config['llm'].get('temperature_induce', 0.8)
        if LOCAL_BACKEND is not None:
            new_text = _call_local_generate(prompt, call_type="rewrite")
        else:
            new_text = call_llm(prompt, temperature=temp, call_type="rewrite")
        return _prepend_score_range_contract(new_text, self.config)
    
    def to_dict(self):
        return {
            "fitness": self.fitness,
            "instruction_preview": self.instruction_text[:200] + "...",
            "full_instruction": self.instruction_text,
            "static_exemplar_ids": [ex['essay_id'] for ex in self.static_exemplars],
            "static_exemplar_scores": [ex['domain1_score'] for ex in self.static_exemplars]
        }

# ============================================================================
# 4. 进化优化器 (EvolutionOptimizer)
# ============================================================================

class EvolutionOptimizer:
    def __init__(self, train_data, val_data, config,
                 pace_evaluator=None,   # Optional[PaceFitnessEvaluator]
                 calib_items=None,      # Optional[List[Dict]]
                 fitness_items=None):   # Optional[List[Dict]]
        self.config = config
        self.train_data = train_data
        self.val_data = val_data

        self.vector_store = SimpleVectorStore(model_name=config['rag']['model_name'])
        if config['rag'].get('enabled', False):
            print("[Optimizer] Building Vector Store (Train Set Only)...")
            self.vector_store.add_documents(self.train_data)
        else:
            self.vector_store.documents = self.train_data
            print("[Optimizer] RAG disabled. Skipping Vector Store build.")

        self.population = []
        self.history = []

        self.fitness_memo = {}  # [New] 全局 raw QWK 适应度缓存（仅缓存 raw，不缓存 combined）
        self.anchor_mutation_events = []

        # WISE-PACE
        self.pace_evaluator = pace_evaluator
        self.calib_items = calib_items
        self.fitness_items = fitness_items
        self.latest_pace_results = {}
        self.best_candidates = {
            "best_raw_val": None,
            "best_raw_guarded": None,
            "best_protocol_quality": None,
            "best_pace_guarded": None,
            "best_pareto": None,
        }
        self.best_candidate_scores = {
            "best_raw_val": float("-inf"),
            "best_raw_guarded": float("-inf"),
            "best_protocol_quality": float("-inf"),
            "best_pace_guarded": float("-inf"),
            "best_pareto": float("-inf"),
        }
        self._build_stratum_pools()

    def _build_stratum_pools(self):
        strategy = self._stratum_strategy()
        if strategy == 'percentile' and len(self.train_data) >= 3:
            sorted_data = sorted(self.train_data, key=lambda x: (x['domain1_score'], x['essay_id']))
            n = len(sorted_data)
            cut1 = max(1, n // 3)
            cut2 = max(cut1 + 1, (2 * n) // 3)
            cut2 = min(cut2, n - 1)
            self.low_pool = sorted_data[:cut1]
            self.mid_pool = sorted_data[cut1:cut2]
            self.high_pool = sorted_data[cut2:]
            lo = self.low_pool[-1]['domain1_score']
            hi = self.high_pool[0]['domain1_score']
        else:
            lo, hi = self._stratum_thresholds(self.train_data)
            self.low_pool = [x for x in self.train_data if x['domain1_score'] <= lo]
            self.mid_pool = [x for x in self.train_data if lo < x['domain1_score'] < hi]
            self.high_pool = [x for x in self.train_data if x['domain1_score'] >= hi]
            self._repair_empty_score_band_pools()
        self.stratum_by_id = {}
        for name, pool in (('low', self.low_pool), ('mid', self.mid_pool), ('high', self.high_pool)):
            for item in pool:
                self.stratum_by_id[item['essay_id']] = name
        print(
            "[Optimizer] Anchor strata: "
            f"strategy={strategy} "
            f"lo={float(lo):.2f} hi={float(hi):.2f} "
            f"low={len(self.low_pool)} mid={len(self.mid_pool)} high={len(self.high_pool)}"
        )
        self.anchor_slot_specs = self._build_anchor_slot_specs(
            int(self.config.get('evolution', {}).get('n_static_exemplars', 3))
        )
        self.anchor_slot_pools = self._build_anchor_slot_pools(self.anchor_slot_specs)
        if self.config.get('evolution', {}).get('anchor_profile') == 'boundary':
            slot_summary = ", ".join(
                f"{spec['name']}[{spec['min']}-{spec['max']}]={len(self.anchor_slot_pools.get(spec['name'], []))}"
                for spec in self.anchor_slot_specs
            )
            print(f"[Optimizer] Boundary anchor slots: {slot_summary}")

    def _get_stratum(self, essay: dict) -> str:
        mapped = getattr(self, 'stratum_by_id', {}).get(essay.get('essay_id'))
        if mapped:
            return mapped
        lo, hi = self._stratum_thresholds(self.train_data)
        score = essay['domain1_score']
        if score <= lo:
            return 'low'
        if score >= hi:
            return 'high'
        return 'mid'

    def _stratum_strategy(self) -> str:
        return self.config.get('evolution', {}).get('anchor_stratum_strategy', 'score_band')

    def _stratum_thresholds(self, data: List[Dict]) -> tuple[float, float]:
        strategy = self._stratum_strategy()
        if strategy == 'percentile' and data:
            scores = np.array([x['domain1_score'] for x in data], dtype=float)
            lo = float(np.quantile(scores, 1.0 / 3.0))
            hi = float(np.quantile(scores, 2.0 / 3.0))
            if lo < hi:
                return lo, hi
        s_min = self.config['data']['score_min']
        s_max = self.config['data']['score_max']
        evo_cfg = self.config.get('evolution', {})
        low_max = evo_cfg.get('anchor_low_max')
        high_min = evo_cfg.get('anchor_high_min')
        if low_max is None:
            low_max = math.floor(s_min + (s_max - s_min) / 3.0)
        if high_min is None:
            high_min = math.ceil(s_min + 2.0 * (s_max - s_min) / 3.0)
        return float(low_max), float(high_min)

    def _repair_empty_score_band_pools(self):
        if not self.train_data:
            return
        sorted_data = sorted(self.train_data, key=lambda x: (x['domain1_score'], x['essay_id']))
        if not self.low_pool:
            min_score = sorted_data[0]['domain1_score']
            self.low_pool = [x for x in sorted_data if x['domain1_score'] == min_score]
        if not self.high_pool:
            max_score = sorted_data[-1]['domain1_score']
            self.high_pool = [x for x in sorted_data if x['domain1_score'] == max_score]
        if not self.mid_pool:
            low_ids = {x['essay_id'] for x in self.low_pool}
            high_ids = {x['essay_id'] for x in self.high_pool}
            self.mid_pool = [
                x for x in sorted_data
                if x['essay_id'] not in low_ids and x['essay_id'] not in high_ids
            ] or sorted_data

    def _build_anchor_slot_specs(self, n_slots: int) -> List[Dict[str, Any]]:
        evo_cfg = self.config.get('evolution', {})
        profile = evo_cfg.get('anchor_profile', 'stratum')
        s_min = int(self.config['data']['score_min'])
        s_max = int(self.config['data']['score_max'])
        if profile != 'boundary' or n_slots < 4:
            lo, hi = self._stratum_thresholds(self.train_data)
            specs = [
                {"name": "low", "min": s_min, "max": int(math.floor(lo))},
                {"name": "mid", "min": int(math.floor(lo)) + 1, "max": int(math.ceil(hi)) - 1},
                {"name": "high", "min": int(math.ceil(hi)), "max": s_max},
            ]
        else:
            available_scores = sorted({int(x['domain1_score']) for x in self.train_data})
            if len(available_scores) >= 2:
                count_by_score = {
                    s: sum(1 for x in self.train_data if int(x['domain1_score']) == s)
                    for s in available_scores
                }
                adjacent_pairs = [
                    (a, b)
                    for a, b in zip(available_scores[:-1], available_scores[1:])
                    if b > a
                ]
                mid = (s_min + s_max) / 2.0
                ranked_pairs = sorted(
                    adjacent_pairs,
                    key=lambda p: (
                        min(count_by_score.get(p[0], 0), count_by_score.get(p[1], 0)),
                        -abs(((p[0] + p[1]) / 2.0) - mid),
                        -abs(p[1] - p[0]),
                    ),
                    reverse=True,
                )
                boundary_pairs = ranked_pairs[:2]
            else:
                boundary_pairs = []
            span = max(1, s_max - s_min)
            if len(boundary_pairs) < 2:
                b1 = int(round(s_min + 0.45 * span))
                b2 = int(round(s_min + 0.60 * span))
                boundary_pairs.extend([(b1, min(s_max, b1 + 1)), (b2, min(s_max, b2 + 1))])
            low_cut = max(s_min, int(math.floor(np.quantile(available_scores, 0.20))) if available_scores else s_min)
            high_cut = min(s_max, int(math.ceil(np.quantile(available_scores, 0.80))) if available_scores else s_max)
            specs = [
                {"name": "low", "min": s_min, "max": low_cut},
                {
                    "name": f"boundary_{boundary_pairs[0][0]}_{boundary_pairs[0][1]}",
                    "min": max(s_min, boundary_pairs[0][0]),
                    "max": min(s_max, boundary_pairs[0][1]),
                },
                {
                    "name": f"boundary_{boundary_pairs[1][0]}_{boundary_pairs[1][1]}",
                    "min": max(s_min, boundary_pairs[1][0]),
                    "max": min(s_max, boundary_pairs[1][1]),
                },
                {"name": "high", "min": high_cut, "max": s_max},
            ]
        while len(specs) < n_slots:
            specs.insert(-1, {"name": f"mid_extra_{len(specs)}", "min": s_min, "max": s_max})
        return specs[:n_slots]

    def _build_anchor_slot_pools(self, specs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        pools = {}
        for spec in specs:
            lo = int(spec["min"])
            hi = int(spec["max"])
            exact = [x for x in self.train_data if lo <= int(x['domain1_score']) <= hi]
            if exact:
                pools[spec["name"]] = exact
                continue
            if not self.train_data:
                pools[spec["name"]] = []
                continue
            def distance_to_range(item):
                score = int(item['domain1_score'])
                if score < lo:
                    return lo - score
                if score > hi:
                    return score - hi
                return 0
            min_dist = min(distance_to_range(x) for x in self.train_data)
            pools[spec["name"]] = [x for x in self.train_data if distance_to_range(x) == min_dist]
        return pools

    def _anchor_slot_name(self, slot: int) -> Optional[str]:
        specs = getattr(self, 'anchor_slot_specs', [])
        if 0 <= int(slot) < len(specs):
            return specs[int(slot)].get("name")
        return None

    def _sample_stratum(self, stratum: str, exclude_id=None) -> dict:
        pools = {
            'low': self.low_pool,
            'mid': self.mid_pool,
            'high': self.high_pool,
        }
        pools.update(getattr(self, 'anchor_slot_pools', {}))
        pool = pools.get(stratum) or self.train_data
        if exclude_id is not None and len(pool) > 1:
            filtered = [x for x in pool if x['essay_id'] != exclude_id]
            if filtered:
                pool = filtered
        return random.choice(pool)

    def _select_anchor_replacement(
        self,
        child: PromptIndividual,
        slot: int,
        stratum: str,
        exclude_ids: Optional[set] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        pools = {
            'low': self.low_pool,
            'mid': self.mid_pool,
            'high': self.high_pool,
        }
        pools.update(getattr(self, 'anchor_slot_pools', {}))
        pool = list(pools.get(stratum) or self.train_data)
        exclude_ids = set(exclude_ids or set())
        if len(pool) > 1:
            pool = [x for x in pool if x['essay_id'] not in exclude_ids] or pool

        evo_cfg = self.config.get('evolution', {})
        strategy = evo_cfg.get('anchor_mutation_strategy', 'evidence_representative')
        candidate_pool_size = int(evo_cfg.get('anchor_mutation_candidate_pool', 12))
        hidden_rerank = bool(evo_cfg.get('anchor_mutation_hidden_rerank', False))
        hidden_top_m = int(evo_cfg.get('anchor_mutation_hidden_top_m', 4))
        if strategy == 'random' or not pool:
            chosen = random.choice(pool or self.train_data)
            return chosen, {
                "anchor_mutation_strategy": "random",
                "candidate_pool_size": len(pool),
                "replacement_rank_score": None,
                "hidden_rerank": False,
            }

        sample_n = min(max(1, candidate_pool_size), len(pool))
        candidates = random.sample(pool, sample_n) if len(pool) > sample_n else list(pool)
        scores = np.array([float(x['domain1_score']) for x in pool], dtype=float)
        lengths = np.array([max(1, len(str(x.get('essay_text', '')).split())) for x in pool], dtype=float)
        median_score = float(np.median(scores)) if scores.size else 0.0
        median_len = float(np.median(lengths)) if lengths.size else 1.0
        score_span = max(1.0, float(scores.max() - scores.min())) if scores.size else 1.0
        len_span = max(1.0, float(lengths.max() - lengths.min())) if lengths.size else 1.0

        other_anchors = [
            ex for i, ex in enumerate(child.static_exemplars)
            if i != slot
        ]
        other_token_sets = [self._token_set(ex.get('essay_text', '')) for ex in other_anchors]
        slot_specs = getattr(self, 'anchor_slot_specs', [])
        slot_spec = slot_specs[slot] if 0 <= int(slot) < len(slot_specs) else {}
        boundary_priority_enabled = bool(evo_cfg.get('anchor_mutation_boundary_hard_priority', True))
        slot_midpoint = (
            (float(slot_spec.get("min", median_score)) + float(slot_spec.get("max", median_score))) / 2.0
            if slot_spec else median_score
        )

        ranked = []
        for cand in candidates:
            cand_score = float(cand['domain1_score'])
            cand_len = max(1, len(str(cand.get('essay_text', '')).split()))
            score_centrality = 1.0 - min(1.0, abs(cand_score - median_score) / score_span)
            length_centrality = 1.0 - min(1.0, abs(cand_len - median_len) / len_span)
            cand_tokens = self._token_set(cand.get('essay_text', ''))
            max_overlap = 0.0
            for token_set in other_token_sets:
                denom = max(1, len(cand_tokens | token_set))
                max_overlap = max(max_overlap, len(cand_tokens & token_set) / denom)
            diversity = 1.0 - max_overlap
            boundary_hardness = 0.0
            if boundary_priority_enabled and str(stratum).startswith("boundary_"):
                boundary_hardness = 1.0 - min(1.0, abs(cand_score - slot_midpoint) / max(1.0, score_span))
            rank_score = (
                0.48 * score_centrality
                + 0.22 * length_centrality
                + 0.20 * diversity
                + 0.10 * boundary_hardness
            )
            ranked.append((rank_score, cand))

        ranked.sort(key=lambda x: (x[0], x[1]['domain1_score']), reverse=True)
        hidden_meta = {
            "hidden_rerank": False,
            "hidden_rerank_candidates": 0,
            "hidden_anchor_separation_raw": None,
            "hidden_rerank_usage": {},
        }
        if hidden_rerank and self.pace_evaluator is not None:
            reranked = []
            usage_before = self.pace_evaluator.backend.usage_snapshot()
            for cheap_score, cand in ranked[: max(1, hidden_top_m)]:
                hidden_sep = None
                try:
                    trial_exemplars = list(child.static_exemplars)
                    trial_exemplars[slot] = cand
                    trial_hiddens = self.pace_evaluator.compute_anchor_hiddens(
                        trial_exemplars,
                        child.instruction_text,
                    )
                    hidden_sep = float(self.pace_evaluator._anchor_separation_raw(trial_hiddens))
                    hidden_bonus = max(0.0, min(1.0, hidden_sep / 2.0))
                    total_score = 0.65 * cheap_score + 0.35 * hidden_bonus
                except Exception as exc:
                    print(f"    [Anchor Hidden Rerank] skipped candidate {cand.get('essay_id')}: {exc}")
                    total_score = cheap_score
                reranked.append((total_score, cheap_score, hidden_sep, cand))
            if reranked:
                usage_delta = self.pace_evaluator.backend.usage_delta(usage_before)
                reranked.sort(key=lambda x: (x[0], x[1], x[3]['domain1_score']), reverse=True)
                total_score, cheap_score, hidden_sep, chosen = reranked[0]
                return chosen, {
                    "anchor_mutation_strategy": strategy,
                    "candidate_pool_size": len(candidates),
                    "replacement_rank_score": round(float(total_score), 6),
                    "replacement_cheap_rank_score": round(float(cheap_score), 6),
                    "hidden_rerank": True,
                    "hidden_rerank_candidates": len(reranked),
                    "hidden_anchor_separation_raw": (
                        round(float(hidden_sep), 6) if hidden_sep is not None else None
                    ),
                    "hidden_rerank_usage": usage_delta,
                }
        chosen_score, chosen = ranked[0]
        return chosen, {
            "anchor_mutation_strategy": strategy,
            "candidate_pool_size": len(candidates),
            "replacement_rank_score": round(float(chosen_score), 6),
            **hidden_meta,
        }

    def _token_set(self, text: str) -> set:
        return set(re.findall(r"\b\w+\b", str(text).lower()))

    def _pace_teacher_guidance(self, result: Dict[str, Any]) -> List[str]:
        guidance = []
        bias = float(result.get("pace_pred_bias", 0.0) or 0.0)
        dist_tv = float(result.get("pace_distribution_tv", 0.0) or 0.0)
        dist_penalty = float(result.get("distribution_penalty", 0.0) or 0.0)
        collapse = float(result.get("pace_pred_collapse_ratio", 0.0) or 0.0)
        max_over = float(result.get("pace_max_score_overprediction_ratio", 0.0) or 0.0)
        dominant_error = result.get("dominant_error_type")
        if bias < -1.0:
            guidance.append(
                "Raw/PACE evidence under-predicts: add concrete recognition rules for strong content, organization, and language quality in high-score bands."
            )
        elif bias > 1.0:
            guidance.append(
                "Raw/PACE evidence over-predicts: add stricter high-score requirements and explicit deductions for weak development or language control."
            )
        if dist_tv > float(self.config.get('pace', {}).get("distribution_tv_threshold", 0.25)) or dist_penalty > 0.0:
            guidance.append(
                "Prediction distribution is miscalibrated: sharpen score-band boundaries so the scorer uses the full valid range only when evidence supports it."
            )
        if collapse > float(self.config.get('pace', {}).get("score_collapse_threshold", 0.75)):
            guidance.append(
                "Predictions collapse into too few scores: add adjacent-score boundary tests and avoid defaulting to a central score."
            )
        if max_over > float(self.config.get('pace', {}).get("max_score_overprediction_threshold", 3.0)):
            guidance.append(
                "Maximum-score overuse detected: require near-complete fulfillment of all criteria before assigning the top score."
            )
        if dominant_error:
            guidance.append(f"Dominant hidden-evidence error type: {dominant_error}.")
        return guidance

    def _evidence_feedback_for(self, individual: PromptIndividual) -> Dict[str, Any]:
        if not self.config.get('evolution', {}).get('use_evidence_feedback', True):
            return {}
        if not self.latest_pace_results:
            return {}
        for idx, candidate in enumerate(self.population):
            if candidate is individual and idx in self.latest_pace_results:
                result = self.latest_pace_results[idx]
                return {
                    "pace_qwk": result.get("pace_qwk"),
                    "pace_raw_qwk": result.get("raw_qwk"),
                    "raw_val_qwk": result.get("raw_val_qwk"),
                    "calib_pace_qwk": result.get("calib_pace_qwk"),
                    "calib_raw_qwk": result.get("calib_raw_qwk"),
                    "overfit_gap": result.get("overfit_gap"),
                    "overfit_penalty": result.get("overfit_penalty"),
                    "distribution_penalty": result.get("distribution_penalty"),
                    "pace_distribution_tv": result.get("pace_distribution_tv"),
                    "pace_pred_collapse_ratio": result.get("pace_pred_collapse_ratio"),
                    "pace_max_score_overprediction_ratio": result.get("pace_max_score_overprediction_ratio"),
                    "pace_pred_bias": result.get("pace_pred_bias"),
                    "pace_distribution_metrics": result.get("pace_distribution_metrics"),
                    "pace_teacher_guidance": self._pace_teacher_guidance(result),
                    "protocol_quality_selection": result.get("protocol_quality_selection"),
                    "calibrator_probe_combined": result.get("calibrator_probe_combined"),
                    "anchor_geometry_score": result.get("anchor_geometry_score"),
                    "anchor_separation": result.get("anchor_separation"),
                    "anchor_ordinal_consistency": result.get("anchor_ordinal_consistency"),
                    "anchor_monotonicity": result.get("anchor_monotonicity"),
                    "evidence_dim": result.get("evidence_dim"),
                    "evidence_schema": result.get("evidence_schema", [])[:20],
                    "dominant_error_type": result.get("dominant_error_type"),
                    "diagnostic_summary": result.get("pace_diagnostic_summary", {}),
                    "diagnostics": result.get("pace_diagnostics", []),
                }
        return {}

    def _evidence_suggested_anchor_slot(self, individual: PromptIndividual) -> Optional[int]:
        if not self.config.get('evolution', {}).get('use_evidence_anchor_slot', True):
            return None
        if not self.latest_pace_results:
            return None
        for idx, candidate in enumerate(self.population):
            if candidate is individual and idx in self.latest_pace_results:
                slot = self.latest_pace_results[idx].get("suggested_anchor_mutation_slot")
                return int(slot) if slot is not None else None
        return None

    def _build_generation_metrics(
        self,
        gen_start_time: float,
        start_stats: Dict[str, int],
        mutation_events: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        gen_end_time = time.time()
        duration_sec = gen_end_time - gen_start_time

        curr_total = EXP_MANAGER.total_tokens if EXP_MANAGER else 0
        curr_prompt = EXP_MANAGER.total_prompt_tokens if EXP_MANAGER else 0
        curr_compl = EXP_MANAGER.total_completion_tokens if EXP_MANAGER else 0

        diff_total = curr_total - start_stats['total']
        diff_prompt = curr_prompt - start_stats['prompt']
        diff_compl = curr_compl - start_stats['completion']

        pace_prompt_tokens = sum(
            int(r.get("local_prompt_tokens", 0)) for r in self.latest_pace_results.values()
        )
        pace_completion_tokens = sum(
            int(r.get("local_completion_tokens", 0)) for r in self.latest_pace_results.values()
        )
        pace_representation_tokens = sum(
            int(r.get("local_representation_tokens", 0)) for r in self.latest_pace_results.values()
        )

        mutation_events = mutation_events or []
        mutation_hidden_prompt = 0
        mutation_hidden_completion = 0
        mutation_hidden_repr = 0
        for event in mutation_events:
            usage = event.get("hidden_rerank_usage") or {}
            mutation_hidden_prompt += int(usage.get("prompt_tokens", 0))
            mutation_hidden_completion += int(usage.get("completion_tokens", 0))
            mutation_hidden_repr += int(usage.get("representation_tokens", 0))

        pace_prompt_tokens += mutation_hidden_prompt
        pace_completion_tokens += mutation_hidden_completion
        pace_representation_tokens += mutation_hidden_repr
        pace_tokens_total = pace_prompt_tokens + pace_completion_tokens + pace_representation_tokens

        return {
            "duration_sec": round(duration_sec, 2),
            "tokens_total": diff_total,
            "tokens_prompt": diff_prompt,
            "tokens_completion": diff_compl,
            "pace_prompt_tokens": pace_prompt_tokens,
            "pace_completion_tokens": pace_completion_tokens,
            "pace_representation_tokens": pace_representation_tokens,
            "pace_tokens_total": pace_tokens_total,
            "tokens_total_all": diff_total + pace_tokens_total,
            "pace_evaluated_count": len(self.latest_pace_results),
            "pace_early_rejected_count": sum(
                1 for r in self.latest_pace_results.values() if r.get("_early_rejected")
            ),
            "pace_total_forward_passes": sum(
                int(r.get("total_local_forward_passes", 0)) for r in self.latest_pace_results.values()
            ),
            "pace_total_inference_sec": round(
                sum(float(r.get("total_pace_sec", 0.0)) for r in self.latest_pace_results.values()),
                2,
            ),
            "anchor_mutation_count": len(mutation_events),
            "anchor_hidden_rerank_count": sum(1 for e in mutation_events if e.get("hidden_rerank")),
            "anchor_hidden_rerank_prompt_tokens": mutation_hidden_prompt,
            "anchor_hidden_rerank_completion_tokens": mutation_hidden_completion,
            "anchor_hidden_rerank_representation_tokens": mutation_hidden_repr,
            "start_time": datetime.fromtimestamp(gen_start_time).isoformat(),
            "end_time": datetime.fromtimestamp(gen_end_time).isoformat(),
        }

    def _pace_selection_combined(self, pace_result: Dict[str, Any], raw_val_qwk: float) -> float:
        if self.pace_evaluator is None:
            return raw_val_qwk
        pace_cfg = self.config.get('pace', {})
        cfg = self.pace_evaluator.config
        raw_val_qwk = float(raw_val_qwk)
        anchor_guidance_weight = float(pace_cfg.get("anchor_guidance_weight", cfg.gamma))
        max_anchor_bonus = float(pace_cfg.get("max_anchor_geometry_bonus", 0.02))
        anchor_bonus_raw = anchor_guidance_weight * float(pace_result.get("anchor_geometry_score", 0.0))
        anchor_bonus = max(-max_anchor_bonus, min(max_anchor_bonus, anchor_bonus_raw))
        cost_penalty = float(pace_result.get("cost_penalty", 0.0))
        overfit_penalty = float(pace_result.get("overfit_penalty", 0.0))
        distribution_penalty = float(pace_result.get("distribution_penalty", 0.0))
        uncapped_guided = (
            raw_val_qwk
            + anchor_bonus
            - cost_penalty
            - overfit_penalty
            - distribution_penalty
        )
        max_lift = float(pace_cfg.get("max_protocol_quality_lift", 0.02))
        guided = min(uncapped_guided, raw_val_qwk + max_lift)
        pace_result["selection_objective"] = "raw_val_qwk + anchor_geometry_bonus - cost/overfit/distribution penalties"
        pace_result["pace_qwk_used_for_selection"] = False
        pace_result["selection_anchor_bonus_raw"] = anchor_bonus_raw
        pace_result["selection_anchor_bonus"] = anchor_bonus
        pace_result["selection_max_anchor_bonus"] = max_anchor_bonus
        pace_result["selection_uncapped_protocol_quality"] = uncapped_guided
        pace_result["selection_max_protocol_quality_lift"] = max_lift
        pace_result["selection_cost_penalty"] = cost_penalty
        pace_result["selection_overfit_penalty"] = overfit_penalty
        pace_result["selection_distribution_penalty"] = distribution_penalty
        return guided if np.isfinite(guided) else raw_val_qwk

    def _score_prediction_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, Any]:
        score_min = int(self.config['data']['score_min'])
        score_max = int(self.config['data']['score_max'])
        scores = list(range(score_min, score_max + 1))
        n = max(1, len(y_true))
        true_counts = {str(s): 0 for s in scores}
        pred_counts = {str(s): 0 for s in scores}
        for y in y_true:
            key = str(int(y))
            if key in true_counts:
                true_counts[key] += 1
        for y in y_pred:
            key = str(int(y))
            if key in pred_counts:
                pred_counts[key] += 1
        true_probs = np.array([true_counts[str(s)] / n for s in scores], dtype=float)
        pred_probs = np.array([pred_counts[str(s)] / n for s in scores], dtype=float)
        errors = [int(p) - int(t) for t, p in zip(y_true, y_pred)]
        true_mean = float(np.mean(y_true)) if y_true else 0.0
        pred_mean = float(np.mean(y_pred)) if y_pred else 0.0
        return {
            "true_counts": true_counts,
            "pred_counts": pred_counts,
            "tv_distance": float(0.5 * np.abs(pred_probs - true_probs).sum()),
            "pred_collapse_ratio": float(pred_probs.max()) if len(pred_probs) else 0.0,
            "true_mean": true_mean,
            "pred_mean": pred_mean,
            "pred_bias": pred_mean - true_mean,
            "mae": float(np.mean([abs(x) for x in errors])) if errors else 0.0,
            "pred_span": int(max(y_pred) - min(y_pred)) if y_pred else 0,
        }

    def _raw_distribution_penalty(self, metrics: Dict[str, Any]) -> float:
        evo_cfg = self.config.get('evolution', {})
        if not bool(evo_cfg.get("raw_distribution_guard_enabled", True)):
            return 0.0
        collapse_threshold = float(
            self.config.get('pace', {}).get(
                "score_collapse_threshold",
                evo_cfg.get("raw_collapse_threshold", 0.75),
            )
        )
        tv_excess = max(
            0.0,
            float(metrics.get("tv_distance", 0.0))
            - float(evo_cfg.get("raw_distribution_tv_threshold", 0.35)),
        )
        bias_excess = max(
            0.0,
            abs(float(metrics.get("pred_bias", 0.0)))
            - float(evo_cfg.get("raw_bias_threshold", 1.5)),
        )
        collapse_excess = max(
            0.0,
            float(metrics.get("pred_collapse_ratio", 0.0)) - collapse_threshold,
        )
        return (
            float(evo_cfg.get("raw_distribution_penalty_weight", 0.15)) * tv_excess
            + float(evo_cfg.get("raw_bias_penalty_weight", 0.05)) * bias_excess
            + float(evo_cfg.get("raw_collapse_penalty_weight", 0.10)) * collapse_excess
        )

    def _apply_raw_guard(
        self,
        raw_scores: List[float],
        protocol_quality_scores: List[float],
        pop_snapshot: List[Dict[str, Any]],
        raw_adjusted_scores: Optional[List[float]] = None,
        pace_results: Optional[Dict[int, Dict[str, Any]]] = None,
    ) -> Tuple[List[float], List[int], List[int]]:
        pace_cfg = self.config.get('pace', {})
        guard_enabled = bool(pace_cfg.get('raw_guard_enabled', True))
        raw_guard_margin = float(pace_cfg.get('raw_guard_margin', 0.03))
        raw_adjusted_margin = float(pace_cfg.get('raw_adjusted_guard_margin', 0.04))
        max_dist_penalty = float(pace_cfg.get('selection_max_distribution_penalty', 0.15))
        raw_best = float(max(raw_scores)) if raw_scores else 0.0
        raw_adjusted_scores = raw_adjusted_scores or list(raw_scores)
        raw_adjusted_best = float(max(raw_adjusted_scores)) if raw_adjusted_scores else raw_best
        threshold = raw_best - raw_guard_margin
        adjusted_threshold = raw_adjusted_best - raw_adjusted_margin
        selection_scores = [float(x) for x in protocol_quality_scores]
        triggered = []
        feasible_indices = []
        pace_results = pace_results or {}

        for idx, raw_score in enumerate(raw_scores):
            raw_score = float(raw_score)
            raw_adjusted = float(raw_adjusted_scores[idx])
            protocol_quality = float(protocol_quality_scores[idx])
            pace_result = pace_results.get(idx, {})
            dist_penalty = float(pace_result.get("distribution_penalty", 0.0) or 0.0)
            reasons = []
            if raw_score < threshold:
                reasons.append("raw_below_best_margin")
            if raw_adjusted < adjusted_threshold:
                reasons.append("raw_adjusted_below_best_margin")
            if dist_penalty > max_dist_penalty:
                reasons.append("pace_distribution_penalty_high")
            feasible = len(reasons) == 0
            if feasible:
                feasible_indices.append(idx)
            pop_snapshot[idx]['protocol_quality_before_guard'] = protocol_quality
            pop_snapshot[idx]['raw_guard_threshold'] = threshold
            pop_snapshot[idx]['raw_adjusted_guard_threshold'] = adjusted_threshold
            pop_snapshot[idx]['selection_max_distribution_penalty'] = max_dist_penalty
            pop_snapshot[idx]['raw_guard_triggered'] = False
            pop_snapshot[idx]['pareto_feasible'] = feasible
            pop_snapshot[idx]['constraint_reasons'] = reasons
            if guard_enabled and reasons and protocol_quality > raw_adjusted:
                selection_scores[idx] = raw_adjusted
                triggered.append(idx)
                pop_snapshot[idx]['raw_guard_triggered'] = True
                print(
                    f"    [Raw Guard] Ind {idx:02d} protocol_quality={protocol_quality:.4f} "
                    f"raw={raw_score:.4f} raw_adj={raw_adjusted:.4f} "
                    f"reasons={','.join(reasons)}; using raw-adjusted for selection."
                )
            pop_snapshot[idx]['protocol_quality'] = selection_scores[idx]
        return selection_scores, triggered, feasible_indices

    def _track_candidate(
        self,
        label: str,
        individual: PromptIndividual,
        score: float,
        gen: int,
        source_idx: int,
    ):
        score = float(score)
        if not np.isfinite(score):
            return
        if score <= self.best_candidate_scores.get(label, float("-inf")):
            return
        snapshot = individual.clone()
        snapshot.fitness = score
        snapshot.selection_label = label
        snapshot.source_generation = gen
        snapshot.source_index = int(source_idx)
        self.best_candidates[label] = snapshot
        self.best_candidate_scores[label] = score

    # [NEW] Rubric-Guided Induction
    def induce_instruction(self):
        print("[Induction] Starting Instruction Induction...")
        conf = self.config['induction']
        n_high = conf.get('n_high_samples', 3)
        n_low = conf.get('n_low_samples', 3)
        official_rubric = conf.get('official_criteria', "Evaluate based on general quality.")
        
        # Sort by score
        sorted_data = sorted(self.train_data, key=lambda x: x['domain1_score'])
        # 简单的首尾采样，如果数据太少注意处理
        if len(sorted_data) < n_low + n_high:
            low_samples = sorted_data[:len(sorted_data)//2]
            high_samples = sorted_data[len(sorted_data)//2:]
        else:
            low_samples = sorted_data[:n_low]
            high_samples = sorted_data[-n_high:]
        
        print(f"  Selected {len(low_samples)} Low and {len(high_samples)} High samples.")
        
        samples_text = ""
        for i, s in enumerate(high_samples):
            samples_text += f"\n[HIGH SCORE ({s['domain1_score']})] ID {s['essay_id']}:\n{s['essay_text'][:600]}...\n"
        for i, s in enumerate(low_samples):
            samples_text += f"\n[LOW SCORE ({s['domain1_score']})] ID {s['essay_id']}:\n{s['essay_text'][:600]}...\n"
            
        prompt = PromptIndividual.INDUCTION_TEMPLATE.format(
            score_min=self.config['data']['score_min'],
            score_max=self.config['data']['score_max'],
            official_criteria=official_rubric,
            samples_text=samples_text
        )
        
        print("  Generating induced rubric via LLM...")
        temp = self.config['llm'].get('temperature_induce', 0.8)
        if LOCAL_BACKEND is not None:
            induced_rubric = _call_local_generate(prompt, call_type="induction")
        else:
            induced_rubric = call_llm(prompt, temperature=temp, call_type="induction")
        print(f"  [Induction Done] Rubric Length {len(induced_rubric)}")
        return _prepend_score_range_contract(induced_rubric, self.config)

    # [NEW] 分层采样
    def get_stratified_exemplars(self, n=3):
        selection = []
        selected_ids = set()

        def add_from_pool(pool):
            candidates = [x for x in pool if x['essay_id'] not in selected_ids]
            if not candidates:
                return
            item = random.choice(candidates)
            selection.append(item)
            selected_ids.add(item['essay_id'])

        if self.config.get('evolution', {}).get('anchor_profile') == 'boundary':
            for spec in getattr(self, 'anchor_slot_specs', [])[:n]:
                if len(selection) >= n:
                    break
                add_from_pool(getattr(self, 'anchor_slot_pools', {}).get(spec['name'], []))

        if len(selection) < n and self.low_pool:
            add_from_pool(self.low_pool)
        if self.mid_pool and len(selection) < n:
            add_from_pool(self.mid_pool)
        if len(selection) < n and self.high_pool:
            add_from_pool(self.high_pool)
        
        while len(selection) < n:
            candidates = [x for x in self.train_data if x['essay_id'] not in selected_ids]
            if not candidates:
                break
            item = random.choice(candidates)
            selection.append(item)
            selected_ids.add(item['essay_id'])
            
        return selection

    def initialize_population(self, base_instruction):
        n_static = self.config['evolution']['n_static_exemplars']
        print(f"[Init] Pop size: {self.config['evolution']['population_size']}, Stratified Sampling Enabled.")
        for _ in range(self.config['evolution']['population_size']):
            static_ex = self.get_stratified_exemplars(n_static)
            self.population.append(PromptIndividual(base_instruction, static_ex, config=self.config))

    def evolve_one_generation(self, gen):
        gen_start_time = time.time()
        # Snapshot current totals
        start_stats = {
            "total": EXP_MANAGER.total_tokens if EXP_MANAGER else 0,
            "prompt": EXP_MANAGER.total_prompt_tokens if EXP_MANAGER else 0,
            "completion": EXP_MANAGER.total_completion_tokens if EXP_MANAGER else 0
        }
        
        print(f"\n{'='*20} Generation {gen} {'='*20}")
        print(f"  [Start Time] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # [NEW] 读取训练时的 Rerank 开关
        use_rerank_train = self.config['rag'].get('use_rerank_train', False)
        print(f"  [Config] Training Rerank: {'ENABLED' if use_rerank_train else 'DISABLED'}")

        scores = []         # raw QWK per individual
        raw_adjusted_scores = []
        pop_snapshot = []
        lineage_raw_deltas = []
        true_scores_for_raw = [item['domain1_score'] for item in self.val_data]

        for i, ind in enumerate(self.population):
            ex_ids = [ex['essay_id'] for ex in ind.static_exemplars]
            print(f"  [Gen {gen}] Ind {i:02d} | Exemplars: {ex_ids}")

            qwk = ind.evaluate(self.val_data, self.vector_store, enable_rerank=use_rerank_train, fitness_cache=self.fitness_memo)
            raw_metrics = self._score_prediction_metrics(true_scores_for_raw, ind.last_pred_scores)
            raw_penalty = self._raw_distribution_penalty(raw_metrics)
            raw_adjusted = float(qwk) - float(raw_penalty)
            scores.append(qwk)
            raw_adjusted_scores.append(raw_adjusted)
            print(
                f"  >> Ind {i:02d} Finished: Raw QWK = {qwk:.4f} "
                f"| raw_adj={raw_adjusted:.4f} "
                f"| tv={raw_metrics['tv_distance']:.3f} "
                f"| bias={raw_metrics['pred_bias']:.3f}\n"
            )

            ind_data = ind.to_dict()
            ind_data['individual_id'] = i
            ind_data['raw_fitness'] = qwk
            ind_data['raw_adjusted_fitness'] = raw_adjusted
            ind_data['raw_distribution_penalty'] = raw_penalty
            ind_data['raw_distribution_tv'] = raw_metrics.get("tv_distance")
            ind_data['raw_pred_collapse_ratio'] = raw_metrics.get("pred_collapse_ratio")
            ind_data['raw_pred_bias'] = raw_metrics.get("pred_bias")
            ind_data['raw_mae'] = raw_metrics.get("mae")
            ind_data['raw_prediction_metrics'] = raw_metrics
            ind_data['pace_fitness'] = None
            ind_data['pace_raw_fitness'] = None
            ind_data['protocol_quality'] = qwk
            parent_trace = getattr(ind, "parent_trace", None)
            if parent_trace:
                parent_raw = parent_trace.get("parent_raw_qwk")
                raw_delta = (
                    float(qwk) - float(parent_raw)
                    if isinstance(parent_raw, (int, float)) and np.isfinite(parent_raw)
                    else None
                )
                ind_data['parent_trace'] = parent_trace
                ind_data['parent_child_raw_delta'] = raw_delta
                if raw_delta is not None:
                    lineage_raw_deltas.append(raw_delta)
            ind_data['static_exemplar_strata'] = [
                self._anchor_slot_name(slot) or self._get_stratum(ex)
                for slot, ex in enumerate(ind.static_exemplars)
            ]
            pop_snapshot.append(ind_data)

        # WISE-PACE: top-K PACE fitness 评估
        protocol_quality_scores = list(scores)
        self.latest_pace_results = {}
        if self.pace_evaluator is not None and self.calib_items and self.fitness_items:
            top_k = self.pace_evaluator.config.top_k_pace
            elite_buffer = int(self.config.get('pace', {}).get('include_raw_elite_buffer', 0))
            pace_eval_count = min(len(scores), max(1, top_k + max(0, elite_buffer)))
            top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:pace_eval_count]
            print(
                f"  [PACE] Running PACE fitness for {pace_eval_count} individuals "
                f"(top_k={top_k}, elite_buffer={elite_buffer}): {top_k_indices}"
            )
            for idx in top_k_indices:
                ind = self.population[idx]
                pace_result = self.pace_evaluator.compute_pace_fitness(
                    ind, self.calib_items, self.fitness_items
                )
                self.latest_pace_results[idx] = pace_result
                calibrator_probe_combined = pace_result.get(
                    'calibrator_probe_combined',
                    0.0,
                )
                protocol_quality = self._pace_selection_combined(pace_result, scores[idx])
                pace_result['raw_val_qwk'] = scores[idx]
                pace_result['protocol_quality_selection'] = protocol_quality
                pace_result['calibrator_probe_combined'] = calibrator_probe_combined
                is_fallback = pace_result.get("_fallback", False)
                is_degenerate = (not is_fallback) and (not np.isfinite(protocol_quality))

                if is_fallback or is_degenerate:
                    reason = "fallback" if is_fallback else f"degenerate (protocol_quality={protocol_quality})"
                    print(f"    [PACE] Ind {idx:02d} {reason} -> using raw QWK={scores[idx]:.4f}")
                    pop_snapshot[idx]['pace_fitness'] = pace_result['pace_qwk']
                    pop_snapshot[idx]['pace_raw_fitness'] = pace_result.get('raw_qwk')
                    pop_snapshot[idx]['pace_raw_val_fitness'] = scores[idx]
                    pop_snapshot[idx]['calibrator_probe_combined'] = calibrator_probe_combined
                    pop_snapshot[idx]['protocol_quality'] = scores[idx]
                else:
                    protocol_quality_scores[idx] = protocol_quality
                    pop_snapshot[idx]['pace_fitness'] = pace_result['pace_qwk']
                    pop_snapshot[idx]['pace_raw_fitness'] = pace_result.get('raw_qwk')
                    pop_snapshot[idx]['pace_raw_val_fitness'] = scores[idx]
                    pop_snapshot[idx]['calibrator_probe_combined'] = calibrator_probe_combined
                    pop_snapshot[idx]['protocol_quality'] = protocol_quality
                for key in (
                    'anchor_geometry_score',
                    'anchor_separation',
                    'anchor_separation_raw',
                    'anchor_ordinal_consistency',
                    'anchor_monotonicity',
                    'anchor_min_pair',
                    'pace_diagnostic_summary',
                    'dominant_error_type',
                    'suggested_anchor_mutation_slot',
                    'early_rejection_metrics',
                    'cost_penalty',
                    'calib_pace_qwk',
                    'calib_raw_qwk',
                    'overfit_gap',
                    'overfit_penalty',
                    'distribution_penalty',
                    'pace_distribution_tv',
                    'pace_pred_collapse_ratio',
                    'pace_max_score_overprediction_ratio',
                    'pace_pred_bias',
                    'pace_pred_span',
                    'pace_distribution_metrics',
                    'selection_objective',
                    'pace_qwk_used_for_selection',
                    'selection_anchor_bonus_raw',
                    'selection_anchor_bonus',
                    'selection_max_anchor_bonus',
                    'selection_uncapped_protocol_quality',
                    'selection_max_protocol_quality_lift',
                    'selection_cost_penalty',
                    'selection_overfit_penalty',
                    'selection_distribution_penalty',
                    'evidence_dim',
                    'evidence_schema',
                ):
                    pop_snapshot[idx][key] = pace_result.get(key)
                pop_snapshot[idx]['pace_diagnostics'] = pace_result.get('pace_diagnostics', [])
                pop_snapshot[idx]['pace_cost_stats'] = {
                    k: v for k, v in pace_result.items()
                    if (
                        '_sec' in k
                        or k.endswith('_passes')
                        or k.endswith('_tokens')
                        or k in ('n_calib', 'n_fitness', 'local_usage')
                    )
                }
                print(
                    f"    [PACE] Ind {idx:02d} raw_val={scores[idx]:.4f} "
                    f"raw_pace={pace_result.get('raw_qwk', 0.0):.4f} "
                    f"pace={pace_result['pace_qwk']:.4f} "
                    f"anchor={pace_result.get('anchor_geometry_score', 0.0):.4f} "
                    f"overfit_gap={pace_result.get('overfit_gap', 0.0):.4f} "
                    f"dist_tv={pace_result.get('pace_distribution_tv', 0.0):.4f} "
                    f"protocol_quality={pop_snapshot[idx]['protocol_quality']:.4f} "
                    f"(calibrator_probe_combined={calibrator_probe_combined:.4f})"
                )

        selection_scores, raw_guard_triggered, pareto_feasible_indices = self._apply_raw_guard(
            scores,
            protocol_quality_scores,
            pop_snapshot,
            raw_adjusted_scores=raw_adjusted_scores,
            pace_results=self.latest_pace_results,
        )

        for i, ind in enumerate(self.population):
            ind.fitness = selection_scores[i]
            pop_snapshot[i]['fitness'] = selection_scores[i]
            pop_snapshot[i]['selection_fitness'] = selection_scores[i]

        raw_best_idx = int(np.argmax(scores))
        raw_adjusted_best_idx = int(np.argmax(raw_adjusted_scores))
        protocol_quality_best_idx = int(np.argmax(protocol_quality_scores))
        guarded_best_idx = int(np.argmax(selection_scores))
        pareto_pool = pareto_feasible_indices or list(range(len(selection_scores)))
        pareto_best_idx = int(max(pareto_pool, key=lambda i: selection_scores[i]))

        self._track_candidate("best_raw_val", self.population[raw_best_idx], scores[raw_best_idx], gen, raw_best_idx)
        self._track_candidate("best_raw_guarded", self.population[raw_adjusted_best_idx], raw_adjusted_scores[raw_adjusted_best_idx], gen, raw_adjusted_best_idx)
        self._track_candidate("best_protocol_quality", self.population[protocol_quality_best_idx], protocol_quality_scores[protocol_quality_best_idx], gen, protocol_quality_best_idx)
        self._track_candidate("best_pace_guarded", self.population[guarded_best_idx], selection_scores[guarded_best_idx], gen, guarded_best_idx)
        self._track_candidate("best_pareto", self.population[pareto_best_idx], selection_scores[pareto_best_idx], gen, pareto_best_idx)

        best_ind = self.population[pareto_best_idx]
        best_snapshot = best_ind.clone()
        print(
            f"  >> Gen {gen} Best Raw={scores[raw_best_idx]:.4f} (Ind {raw_best_idx}) | "
            f"Best RawAdj={raw_adjusted_scores[raw_adjusted_best_idx]:.4f} (Ind {raw_adjusted_best_idx}) | "
            f"Best ProtocolQuality={protocol_quality_scores[protocol_quality_best_idx]:.4f} (Ind {protocol_quality_best_idx}) | "
            f"Best Pareto={selection_scores[pareto_best_idx]:.4f} (Ind {pareto_best_idx})"
        )

        pace_qwks = [
            float(r.get("pace_qwk", 0.0))
            for r in self.latest_pace_results.values()
            if r.get("pace_qwk") is not None
        ]
        anchor_scores = [
            float(r.get("anchor_geometry_score", 0.0))
            for r in self.latest_pace_results.values()
            if r.get("anchor_geometry_score") is not None
        ]
        overfit_gaps = [
            float(r.get("overfit_gap", 0.0))
            for r in self.latest_pace_results.values()
            if r.get("overfit_gap") is not None
        ]
        overfit_penalties = [
            float(r.get("overfit_penalty", 0.0))
            for r in self.latest_pace_results.values()
            if r.get("overfit_penalty") is not None
        ]
        distribution_penalties = [
            float(r.get("distribution_penalty", 0.0))
            for r in self.latest_pace_results.values()
            if r.get("distribution_penalty") is not None
        ]
        distribution_tvs = [
            float(r.get("pace_distribution_tv", 0.0))
            for r in self.latest_pace_results.values()
            if r.get("pace_distribution_tv") is not None
        ]
        curve_row = {
            "gen": gen,
            "best_raw_val": float(scores[raw_best_idx]),
            "best_raw_guarded": float(raw_adjusted_scores[raw_adjusted_best_idx]),
            "best_protocol_quality": float(protocol_quality_scores[protocol_quality_best_idx]),
            "best_pace_guarded": float(selection_scores[guarded_best_idx]),
            "best_pareto": float(selection_scores[pareto_best_idx]),
            "best_qwk": float(scores[raw_best_idx]),
            "best_pace_qwk": max(pace_qwks) if pace_qwks else None,
            "best_anchor_geometry": max(anchor_scores) if anchor_scores else None,
            "max_overfit_gap": max(overfit_gaps) if overfit_gaps else None,
            "max_overfit_penalty": max(overfit_penalties) if overfit_penalties else None,
            "max_distribution_penalty": max(distribution_penalties) if distribution_penalties else None,
            "max_distribution_tv": max(distribution_tvs) if distribution_tvs else None,
            "raw_best_idx": raw_best_idx,
            "raw_adjusted_best_idx": raw_adjusted_best_idx,
            "protocol_quality_best_idx": protocol_quality_best_idx,
            "guarded_best_idx": guarded_best_idx,
            "pareto_best_idx": pareto_best_idx,
            "pareto_feasible_count": len(pareto_feasible_indices),
            "raw_guard_triggered_count": len(raw_guard_triggered),
            "pace_evaluated_count": len(self.latest_pace_results),
        }
        if lineage_raw_deltas:
            curve_row["mean_parent_child_raw_delta"] = float(np.mean(lineage_raw_deltas))
            curve_row["positive_parent_child_delta_rate"] = float(
                np.mean([1.0 if x > 0 else 0.0 for x in lineage_raw_deltas])
            )
        parent_eval_trace_by_object = {}
        for idx, ind in enumerate(self.population):
            pace_result = self.latest_pace_results.get(idx, {})
            parent_eval_trace_by_object[id(ind)] = {
                "parent_generation": gen,
                "parent_index": idx,
                "parent_raw_qwk": float(scores[idx]),
                "parent_raw_adjusted_qwk": float(raw_adjusted_scores[idx]),
                "parent_protocol_quality": float(protocol_quality_scores[idx]),
                "parent_pace_teacher_guidance": self._pace_teacher_guidance(pace_result) if pace_result else [],
                "parent_dominant_error_type": pace_result.get("dominant_error_type") if pace_result else None,
                "parent_suggested_anchor_slot": pace_result.get("suggested_anchor_mutation_slot") if pace_result else None,
                "pace_qwk_used_for_selection": bool(pace_result.get("pace_qwk_used_for_selection", False)) if pace_result else False,
            }
        
        # Metrics and generation snapshot are finalized after evolution so
        # reflection/rewrite and hidden-rerank costs are included.

        # 2. Evolve Rubrics（基于 raw-guarded selection_scores 选 elite）
        n_elite = self.config['evolution']['n_elite_evolve']
        elites = [self.population[i] for i in np.argsort(selection_scores)[-n_elite:]]

        # 精英保留策略：保留 protocol_quality 最高的个体直接晋级
        best_old_elite = elites[-1].clone()
        new_pop = [best_old_elite] # 精英保留策略：直接晋级下一代
        
        print("  Evolving Rubrics for Elites (Using Real Reflection)...")
        mutated_parents = []
        mutation_events = []
        
        # [NEW] Check config for instruction evolution
        evolve_instr = self.config['evolution'].get('evolve_instruction', True)
        if not evolve_instr:
             print("  [Config] Instruction evolution DISABLED. Skipping reflection/rewrite.")

        for elite in elites:
            elite.evidence_feedback = self._evidence_feedback_for(elite)
            # 这里是修改原始 elite 对象的引用
            if evolve_instr:
                real_feedback = elite.generate_real_feedback(self.val_data)
                print(f"    [Feedback] {real_feedback[:100]}...")
                new_text = elite.evolve_instruction(real_feedback)
                elite.instruction_text = new_text 
            mutated_parents.append(elite)
            
        # 补齐种群：从 mutated_parents 中克隆并变异 exemplars
        existing_fingerprints = {tuple(ex['essay_id'] for ex in ind.static_exemplars) for ind in new_pop}
        
        # [NEW] Check config for exemplar evolution
        evolve_exemplars = self.config['evolution'].get('evolve_static_exemplars', True)
        if not evolve_exemplars:
             print("  [Config] Static Exemplar evolution DISABLED. Skipping mutation.")

        # 剩下的位置用变异填充
        while len(new_pop) < len(self.population):
            parent = random.choice(mutated_parents) # 选新 Rubric 的个体做父母
            child = parent.clone()
            parent_trace = dict(parent_eval_trace_by_object.get(id(parent), {}))
            parent_trace.update({
                "instruction_mutated": bool(evolve_instr),
                "anchor_mutated": False,
                "anchor_mutation_source": None,
                "anchor_mutation_slot": None,
            })
            
            if evolve_exemplars:
                for _ in range(10): 
                    suggested_slot = self._evidence_suggested_anchor_slot(parent)
                    evidence_slot_prob = float(
                        self.config.get('evolution', {}).get('evidence_anchor_slot_prob', 1.0)
                    )
                    use_evidence_slot = (
                        suggested_slot is not None
                        and 0 <= suggested_slot < len(child.static_exemplars)
                        and random.random() < max(0.0, min(1.0, evidence_slot_prob))
                    )
                    if use_evidence_slot:
                        idx = suggested_slot
                        slot_selection_source = "evidence"
                    else:
                        idx = random.randint(0, len(child.static_exemplars)-1)
                        slot_selection_source = "random"
                    old_ex = child.static_exemplars[idx]
                    old_stratum = self._anchor_slot_name(idx) or self._get_stratum(old_ex)
                    exclude_ids = {ex['essay_id'] for ex in child.static_exemplars}
                    new_ex, replacement_meta = self._select_anchor_replacement(
                        child,
                        idx,
                        old_stratum,
                        exclude_ids=exclude_ids,
                    )
                    child.static_exemplars[idx] = new_ex
                    event = {
                        "generation": gen,
                        "slot": idx,
                        "stratum": old_stratum,
                        "old_essay_id": old_ex['essay_id'],
                        "old_score": old_ex['domain1_score'],
                        "new_essay_id": new_ex['essay_id'],
                        "new_score": new_ex['domain1_score'],
                        "mutation_source": slot_selection_source,
                        "slot_selection_source": slot_selection_source,
                        "anchor_slot_name": old_stratum,
                        "parent_trace": dict(parent_trace),
                    }
                    event.update(replacement_meta)
                    parent_trace.update({
                        "anchor_mutated": True,
                        "anchor_mutation_source": slot_selection_source,
                        "anchor_mutation_slot": idx,
                        "anchor_mutation_stratum": old_stratum,
                        "old_anchor_score": old_ex['domain1_score'],
                        "new_anchor_score": new_ex['domain1_score'],
                        "anchor_mutation_strategy": replacement_meta.get('anchor_mutation_strategy'),
                    })
                    mutation_events.append(event)
                    print(
                        f"    [Anchor Mutation] slot={idx} stratum={old_stratum} "
                        f"{old_ex['essay_id']} -> {new_ex['essay_id']} "
                        f"strategy={replacement_meta.get('anchor_mutation_strategy')}"
                    )
                    
                    fp = tuple(ex['essay_id'] for ex in child.static_exemplars)
                    if fp not in existing_fingerprints:
                        existing_fingerprints.add(fp)
                        break 
            
            child.parent_trace = parent_trace
            new_pop.append(child)
            
        self.population = new_pop
        self.anchor_mutation_events.extend(mutation_events)
        if EXP_MANAGER and mutation_events:
            EXP_MANAGER.save_anchor_mutations(gen, mutation_events)
        metrics = self._build_generation_metrics(gen_start_time, start_stats, mutation_events)
        curve_row.update({
            "duration_sec": metrics.get("duration_sec"),
            "tokens_total_all": metrics.get("tokens_total_all"),
            "tokens_total": metrics.get("tokens_total"),
            "pace_tokens_total": metrics.get("pace_tokens_total"),
        })
        metrics["selection_summary"] = {
            "best_raw_val": curve_row["best_raw_val"],
            "best_raw_guarded": curve_row["best_raw_guarded"],
            "best_protocol_quality": curve_row["best_protocol_quality"],
            "best_pace_guarded": curve_row["best_pace_guarded"],
            "best_pareto": curve_row["best_pareto"],
            "raw_best_idx": raw_best_idx,
            "raw_adjusted_best_idx": raw_adjusted_best_idx,
            "protocol_quality_best_idx": protocol_quality_best_idx,
            "guarded_best_idx": guarded_best_idx,
            "pareto_best_idx": pareto_best_idx,
            "pareto_feasible_indices": pareto_feasible_indices,
            "raw_guard_triggered_indices": raw_guard_triggered,
        }
        self.history.append(curve_row)
        print(f"  [Gen {gen} Stats] Duration: {metrics['duration_sec']:.1f}s")
        print(
            f"    Tokens: Total {metrics['tokens_total']} "
            f"(Prompt {metrics['tokens_prompt']} + Compl {metrics['tokens_completion']})"
        )
        if metrics["pace_tokens_total"]:
            print(
                f"    PACE Local Tokens: {metrics['pace_tokens_total']} "
                f"(Prompt {metrics['pace_prompt_tokens']} + Compl {metrics['pace_completion_tokens']} "
                f"+ Repr {metrics['pace_representation_tokens']})"
            )
        if EXP_MANAGER:
            EXP_MANAGER.save_generation_snapshot(gen, pop_snapshot, metrics=metrics)
            EXP_MANAGER.save_training_curve(self.history)
        return best_snapshot

    # [NEW] 断点续传加载
    def load_population(self, gen_file_path):
        with open(gen_file_path, 'r', encoding='utf-8') as f:
            snapshot = json.load(f)
            
        start_gen = snapshot['generation'] + 1
        pop_data = snapshot['population']
        
        print(f"[Resume] Loading population from Generation {snapshot['generation']} (Next: {start_gen})")
        
        # 构建 essay_id -> essay_content 的映射，用于恢复 Exemplar
        id_to_doc = {d['essay_id']: d for d in self.train_data}
        
        self.population = []
        for p in pop_data:
            rubric = p['full_instruction']
            fitness = p.get('fitness', 0.0)
            
            # 恢复 Exemplars
            exemplars = []
            static_ids = p['static_exemplar_ids']
            for eid in static_ids:
                if eid in id_to_doc:
                    exemplars.append(id_to_doc[eid])
                else:
                    print(f"  [Warning] Exemplar ID {eid} not found in Train Set! Using random replacement.")
                    exemplars.append(random.choice(self.train_data))
            
            ind = PromptIndividual(rubric, exemplars, fitness, self.config)
            # 恢复 last_pred_scores (如果在 json 里没存，那第一代 reflection 可能会受影响，但这是可接受的损失)
            # 目前 gen.json 确实没存 last_pred_scores，所以第一代变异可能效果打折，但之后会恢复正常。
            self.population.append(ind)
            
        # 恢复 History (简单起见，只恢复 best_qwk 记录，或者你可以去读之前的 gen files)
        # 这里我们暂且留空，或者只记录当前加载的这一代
        self.history = [{"gen": snapshot['generation'], "best_qwk": snapshot['best_qwk']}]
        
        return start_gen

    def run(self, start_gen=1):
        
        # 如果是从中间开始，先评估一下当前的 best_global
        if self.population and start_gen > 1:
             curr_best = max(self.population, key=lambda x: x.fitness)
             self._track_candidate("best_raw_val", curr_best, curr_best.fitness, start_gen - 1, -1)
             self._track_candidate("best_raw_guarded", curr_best, curr_best.fitness, start_gen - 1, -1)
             self._track_candidate("best_protocol_quality", curr_best, curr_best.fitness, start_gen - 1, -1)
             self._track_candidate("best_pace_guarded", curr_best, curr_best.fitness, start_gen - 1, -1)
             self._track_candidate("best_pareto", curr_best, curr_best.fitness, start_gen - 1, -1)

        for g in range(start_gen, self.config['evolution']['n_generations']+1):
            self.evolve_one_generation(g)
        if EXP_MANAGER:
            EXP_MANAGER.save_training_curve(self.history)
        return {
            "best_raw_val": self.best_candidates.get("best_raw_val"),
            "best_raw_guarded": self.best_candidates.get("best_raw_guarded"),
            "best_protocol_quality": self.best_candidates.get("best_protocol_quality"),
            "best_pace_guarded": self.best_candidates.get("best_pace_guarded"),
            "best_pareto": self.best_candidates.get("best_pareto"),
            "scores": dict(self.best_candidate_scores),
            "history": self.history,
        }

# ============================================================================
# 5. 主程序 (Main)
# ============================================================================

def main(config_path="configs/default.yaml", fold=0, resume_path=None):
    global EXP_MANAGER, LOCAL_BACKEND
    EXP_MANAGER = ExperimentManager(config_path=config_path, fold=fold, resume_path=resume_path)
    config = EXP_MANAGER.config
    seed = config.get('debug', {}).get('seed', config.get('experiment', {}).get('seed', 42 + fold))
    _set_global_seed(seed)
    print(f"[Main] Global seed set to {seed}")

    # WISE-PACE: 本地后端初始化（pace.enabled=false 时跳过）
    pace_cfg = config.get('pace', {})
    pace_enabled = pace_cfg.get('enabled', False)
    pace_evaluator = None
    calib_items = None
    fitness_items = None
    if not pace_enabled:
        LOCAL_BACKEND = None

    if pace_enabled:
        if not _PACE_AVAILABLE:
            raise ImportError(
                "pace/ 模块不可用，请检查 pace/llm_backend.py 和 pace/pace_fitness.py 是否存在。"
            )
        model_path = pace_cfg.get('model_path', 'models/Meta-Llama-3.1-8B-Instruct')
        print(f"[PACE] Loading LocalLlamaBackend from {model_path} ...")
        LOCAL_BACKEND = LocalLlamaBackend(
            config=config,
            model_path=model_path,
            dtype=pace_cfg.get('dtype', 'bfloat16'),
            load_in_4bit=pace_cfg.get('load_in_4bit', False),
        )
        pace_fit_cfg = PaceFitnessConfig.from_config(config)
        pace_evaluator = PaceFitnessEvaluator(
            local_backend=LOCAL_BACKEND,
            config=pace_fit_cfg,
            score_min=config['data']['score_min'],
            score_max=config['data']['score_max'],
        )
        print(
            f"[PACE] PaceFitnessEvaluator ready. "
            f"top_k={pace_fit_cfg.top_k_pace} alpha={pace_fit_cfg.alpha} "
            f"beta={pace_fit_cfg.beta} gamma={pace_fit_cfg.gamma}"
        )
    
    # 1. 加载数据
    data_path = config['data']['asap_path']
    essay_set = config['data']['essay_set']
    print(f"[Main] Loading ASAP Data Set {essay_set} from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    else:
        df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
        df = df[df['essay_set'] == essay_set]
        all_data = []
        for _, row in df.iterrows():
            all_data.append({
                "essay_id": row['essay_id'],
                "essay_text": row['essay'],
                "domain1_score": int(row['domain1_score'])
            })
    
    # 2. 数据切分
    if config.get('debug', {}).get('enabled', False):
        print("  [Debug Mode] Using small subset.")
        seed = config['debug'].get('seed', 42 + fold)
        n_train = config['debug'].get('n_train', 10)
        n_val = config['debug'].get('n_val', 5)
        n_test = config['debug'].get('n_test', 5)
        use_stratified = config['debug'].get('stratified', True)
        if use_stratified:
            try:
                train_set, val_set, test_set = _stratified_debug_split(
                    all_data,
                    n_train,
                    n_val,
                    n_test,
                    seed,
                    config['data']['score_min'],
                    config['data']['score_max'],
                )
                print("  [Debug Mode] Stratified score-band split enabled.")
            except Exception as exc:
                print(f"  [Debug Mode] Stratified split failed ({exc}); falling back to seeded shuffle.")
                random.Random(seed).shuffle(all_data)
                train_set = all_data[:n_train]
                val_set = all_data[n_train:n_train+n_val]
                test_set = all_data[n_train+n_val:n_train+n_val+n_test]
        else:
            random.Random(seed).shuffle(all_data)
            train_set = all_data[:n_train]
            val_set = all_data[n_train:n_train+n_val]
            test_set = all_data[n_train+n_val:n_train+n_val+n_test]
    else:
        labels = _score_band_labels(
            all_data,
            config['data']['score_min'],
            config['data']['score_max'],
        )
        try:
            kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kf.split(all_data, labels))
            print("  [Split] StratifiedKFold enabled.")
        except Exception as exc:
            print(f"  [Split] StratifiedKFold failed ({exc}); falling back to KFold.")
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            folds = list(kf.split(all_data))
        train_val_idx, test_idx = folds[fold]

        train_val_set = [all_data[i] for i in train_val_idx]
        test_set = [all_data[i] for i in test_idx]
        train_val_labels = _score_band_labels(
            train_val_set,
            config['data']['score_min'],
            config['data']['score_max'],
        )
        try:
            train_set, val_set = train_test_split(
                train_val_set,
                test_size=0.2,
                stratify=train_val_labels,
                random_state=42 + fold,
            )
            train_set = list(train_set)
            val_set = list(val_set)
        except Exception as exc:
            print(f"  [Split] Stratified train/val split failed ({exc}); falling back to sequential split.")
            split = int(len(train_val_set) * 0.8)
            train_set = train_val_set[:split]
            val_set = train_val_set[split:]
    
    print(f"  Split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    print(f"  Score Dist Train: {_score_distribution(train_set)}")
    print(f"  Score Dist Val:   {_score_distribution(val_set)}")
    print(f"  Score Dist Test:  {_score_distribution(test_set)}")
    print(
        "  Score Band Train/Val/Test: "
        f"{_score_band_distribution(train_set, config['data']['score_min'], config['data']['score_max'])} / "
        f"{_score_band_distribution(val_set, config['data']['score_min'], config['data']['score_max'])} / "
        f"{_score_band_distribution(test_set, config['data']['score_min'], config['data']['score_max'])}"
    )

    # WISE-PACE: 将 val_set 进一步切分为 calib + fitness（仅 pace 启用时）
    if pace_enabled:
        desired_calib = int(len(val_set) * pace_cfg.get('calib_split_ratio', 0.5))
        n_calib = max(4, desired_calib)
        if len(val_set) >= 8:
            n_calib = min(n_calib, len(val_set) - 4)
        else:
            n_calib = max(0, len(val_set) // 2)
        calib_items = val_set[:n_calib]
        fitness_items = val_set[n_calib:]
        print(f"  [PACE] Val split: calib={len(calib_items)}, fitness={len(fitness_items)}")
        # val_set 整体仍传给 EvolutionOptimizer 做 raw QWK（旧逻辑不变）

    # 3. 运行优化
    print("[Main] Initializing Optimizer...")
    optimizer = EvolutionOptimizer(
        train_set, val_set, config,
        pace_evaluator=pace_evaluator,
        calib_items=calib_items,
        fitness_items=fitness_items,
    )
    
    start_gen = 1
    if resume_path:
        # Resume 模式：跳过 Induction/Init，直接加载种群
        start_gen = optimizer.load_population(resume_path)
    else:
        # 正常模式：执行 Init
        if config.get('induction', {}).get('enabled', False):
            base_rubric = optimizer.induce_instruction()
        else:
            print("[Main] Induction DISABLED. Using Official Criteria as Base Rubric.")
            base_rubric = config.get('induction', {}).get('official_criteria', 
                "Evaluate the essay based on: 1. Content and Ideas, 2. Organization, 3. Vocabulary and Grammar.")
        
        optimizer.initialize_population(base_rubric)
    
    best_bundle = optimizer.run(start_gen=start_gen)
    candidates = {
        "best_raw_val": best_bundle.get("best_raw_val"),
        "best_raw_guarded": best_bundle.get("best_raw_guarded"),
        "best_pareto": best_bundle.get("best_pareto"),
        "best_protocol_quality": best_bundle.get("best_protocol_quality"),
        "best_pace_guarded": best_bundle.get("best_pace_guarded"),
    }
    primary_label = "best_pareto"
    raw_primary_label = "best_raw_guarded" if candidates.get("best_raw_guarded") is not None else "best_raw_val"
    best = (
        candidates.get(primary_label)
        or candidates.get(raw_primary_label)
        or candidates.get("best_raw_val")
        or candidates.get("best_protocol_quality")
    )
    val_best_fitness = best.fitness if best is not None else 0.0
    
    # 4. 最终测试
    print(f"\n{'='*20} Final Test Evaluation {'='*20}")
    
    # [NEW] 读取推理时的 Rerank 开关
    use_rerank_test = config['rag'].get('use_rerank_test', True)
    print(f"  [Config] Inference Rerank: {'ENABLED' if use_rerank_test else 'DISABLED'}")
    
    candidate_results = {}
    seen_signatures = {}
    final_pace_tokens = 0
    final_pace_enabled = bool(pace_enabled and pace_evaluator is not None and pace_cfg.get('final_pace_calibrated', True))

    for label, candidate in candidates.items():
        if candidate is None:
            continue
        signature = candidate.get_signature()
        if signature in seen_signatures:
            candidate_results[label] = {
                **candidate_results[seen_signatures[signature]],
                "label": label,
                "duplicate_of": seen_signatures[signature],
            }
            continue

        val_score = float(candidate.fitness)
        print(f"\n  [Final] Evaluating {label} | val_score={val_score:.4f}")
        raw_test_qwk = candidate.evaluate(test_set, optimizer.vector_store, enable_rerank=use_rerank_test)
        raw_test_metrics = optimizer._score_prediction_metrics(
            [x['domain1_score'] for x in test_set],
            candidate.last_pred_scores,
        )
        candidate.fitness = val_score

        pace_calibrated = None
        guarded_calibrated = None
        if final_pace_enabled:
            pace_calibrated = pace_evaluator.score_with_pace_calibrator(candidate, calib_items, test_set)
            final_pace_tokens += (
                int(pace_calibrated.get("local_prompt_tokens", 0))
                + int(pace_calibrated.get("local_completion_tokens", 0))
                + int(pace_calibrated.get("local_representation_tokens", 0))
            )
            final_dist_penalty = float(pace_calibrated.get("distribution_penalty", 0.0) or 0.0)
            final_max_dist_penalty = float(pace_cfg.get("selection_max_distribution_penalty", 0.15))
            guarded_source = "pace_calibrated"
            guarded_qwk = float(pace_calibrated.get("pace_qwk", 0.0) or 0.0)
            if final_dist_penalty > final_max_dist_penalty:
                guarded_source = "raw_test_due_to_distribution_guard"
                guarded_qwk = float(raw_test_qwk)
            guarded_calibrated = {
                "qwk": guarded_qwk,
                "source": guarded_source,
                "distribution_penalty": final_dist_penalty,
                "selection_max_distribution_penalty": final_max_dist_penalty,
            }
            print(
                f"    raw_test={raw_test_qwk:.4f} | "
                f"pace_calibrated={pace_calibrated.get('pace_qwk', 0.0):.4f} "
                f"| guarded_calibrated={guarded_calibrated['qwk']:.4f} "
                f"({guarded_calibrated['source']}) "
                f"(same_pass_raw={pace_calibrated.get('raw_qwk', 0.0):.4f}) "
                f"dist_tv={pace_calibrated.get('pace_distribution_tv', 0.0):.4f} "
                f"max_score_ratio={pace_calibrated.get('pace_max_score_overprediction_ratio', 0.0):.2f}"
            )
        else:
            print(f"    raw_test={raw_test_qwk:.4f}")

        candidate_results[label] = {
            "label": label,
            "validation_score": val_score,
            "source_generation": getattr(candidate, "source_generation", None),
            "source_index": getattr(candidate, "source_index", None),
            "raw_test_qwk": float(raw_test_qwk),
            "raw_test_mae": raw_test_metrics.get("mae"),
            "raw_test_distribution_metrics": raw_test_metrics,
            "pace_calibrated_test": pace_calibrated,
            "guarded_calibrated_test": guarded_calibrated,
            "instruction": candidate.instruction_text,
            "static_exemplar_ids": [ex['essay_id'] for ex in candidate.static_exemplars],
            "static_exemplar_scores": [ex['domain1_score'] for ex in candidate.static_exemplars],
        }
        seen_signatures[signature] = label

    test_qwk = candidate_results.get(primary_label, {}).get("raw_test_qwk")
    if test_qwk is None and candidate_results:
        test_qwk = next(iter(candidate_results.values())).get("raw_test_qwk", 0.0)
    test_qwk = float(test_qwk or 0.0)
    raw_primary_test_qwk = float(
        candidate_results.get(raw_primary_label, {}).get("raw_test_qwk", 0.0) or 0.0
    )
    raw_primary_test_mae = float(
        candidate_results.get(raw_primary_label, {}).get("raw_test_mae", 0.0) or 0.0
    )
    raw_primary_val_qwk = float(
        candidate_results.get(raw_primary_label, {}).get("validation_score", 0.0) or 0.0
    )

    print(f"\nValidation Primary QWK: {val_best_fitness:.4f} ({primary_label})")
    print(f"Primary Raw Test QWK:   {test_qwk:.4f}")
    if raw_primary_label in candidate_results:
        print(
            f"Raw-Scoring Primary:    {raw_primary_label} "
            f"val={raw_primary_val_qwk:.4f} raw_test={raw_primary_test_qwk:.4f} "
            f"raw_mae={raw_primary_test_mae:.4f}"
        )
    
    # [NEW] Final Global Stats
    total_duration = time.time() - EXP_MANAGER.exp_start_time
    total_pace_tokens = _sum_generation_metric(EXP_MANAGER.gens_dir, "pace_tokens_total")
    total_tokens_all = EXP_MANAGER.total_tokens + total_pace_tokens + final_pace_tokens
    print(f"\n{'='*20} Experiment Summary {'='*20}")
    print(f"Total Duration:      {total_duration / 60:.1f} minutes")
    print(f"Total Tokens:        {EXP_MANAGER.total_tokens}")
    print(f"  - Prompt:          {EXP_MANAGER.total_prompt_tokens}")
    print(f"  - Completion:      {EXP_MANAGER.total_completion_tokens}")
    print(f"PACE Tokens:         {total_pace_tokens}")
    print(f"Final PACE Tokens:   {final_pace_tokens}")
    print(f"Total Tokens (All):  {total_tokens_all}")
    
    EXP_MANAGER.save_final_results(best, optimizer.history, test_results={
        "primary_candidate": primary_label,
        "raw_primary_candidate": raw_primary_label,
        "selection_policy": (
            "Primary candidate is selected before test evaluation by raw-first constrained/Pareto validation. "
            "PACE-calibrated test scores are diagnostics only; raw direct test QWK is the success metric."
        ),
        "test_qwk": test_qwk,
        "primary_raw_test_qwk": test_qwk,
        "raw_primary_test_qwk": raw_primary_test_qwk,
        "raw_primary_test_mae": raw_primary_test_mae,
        "raw_primary_score_distribution": (
            candidate_results.get(raw_primary_label, {})
            .get("raw_test_distribution_metrics", {})
            .get("pred_counts", {})
        ),
        "raw_primary_val_qwk": raw_primary_val_qwk,
        "candidate_results": candidate_results,
        "best_candidate_scores": best_bundle.get("scores", {}),
        "total_duration_sec": total_duration,
        "total_tokens": EXP_MANAGER.total_tokens,
        "total_prompt_tokens": EXP_MANAGER.total_prompt_tokens,
        "total_completion_tokens": EXP_MANAGER.total_completion_tokens,
        "total_pace_tokens": total_pace_tokens,
        "final_pace_tokens": final_pace_tokens,
        "total_tokens_all": total_tokens_all,
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to generation json file (e.g., logs/exp.../generations/gen_010.json) to resume from.")
    args = parser.parse_args()
    main(args.config, args.fold, args.resume)
