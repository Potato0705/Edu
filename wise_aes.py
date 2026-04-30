"""
WISE-AES: Weakly-supervised Integrated Scoring Evolution
版本: v3.0 (Enhanced with CoT, JSON Parsing, Real Reflection, Stratified Sampling)
"""

import os
import sys
import re
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
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from transformers import AutoTokenizer # [NEW] For precise token counting

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

    def save_final_results(self, best_ind, history, test_results=None):
        res = {
            "best_qwk_val": best_ind.fitness,
            "instruction": best_ind.instruction_text,
            "static_exemplars": [ex['essay_id'] for ex in best_ind.static_exemplars],
            "history": history,
            "test_results": test_results
        }
        with open(self.exp_dir / "final_result.json", 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nFinal results saved to {self.exp_dir}")

EXP_MANAGER = None

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

# ============================================================================
# 3. PromptIndividual (包含核心修改)
# ============================================================================

@dataclass
class PromptIndividual:
    instruction_text: str
    static_exemplars: List[Dict[str, Any]] = field(default_factory=list)
    fitness: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    
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

### OFFICIAL CRITERIA:
{official_criteria}

### STUDENT ESSAY SAMPLES (Grounding Data):
{samples_text}

### INSTRUCTION:
1. Analyze the difference between High-Scoring and Low-Scoring essays.
2. Refine the OFFICIAL CRITERIA into a detailed, step-by-step scoring guide.
3. Highlight specific, observable discriminators (e.g., "use of transitions," "sentence variety").
4. The output must be ready to use as a prompt for an LLM grader.

Output ONLY the Rubric text. Do not output explanations."""

    def __post_init__(self):
        if not self.config: self.config = EXP_MANAGER.config

    def get_signature(self):
        # 指纹 = Rubric文本的Hash + 排序后的范文ID
        ex_ids = sorted([str(ex['essay_id']) for ex in self.static_exemplars])
        content = self.instruction_text + "_" + "_".join(ex_ids)
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
    def predict_score(self, essay_text: str, vector_store: SimpleVectorStore, enable_rerank: bool = False) -> int:
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
        return "\n\n".join([f"### Essay (Score: {ex['domain1_score']})\n{ex['essay_text'][:400]}..." for ex in exs])

    def evaluate(self, val_set: List[Dict], vector_store: SimpleVectorStore, enable_rerank: bool = False, fitness_cache=None) -> float:
        # [Optimization] 1. 检查缓存
        sig = self.get_signature()
        if fitness_cache is not None and sig in fitness_cache:
            print(f"    [Cache Hit] Skipping evaluation for {sig[:8]}...")
            self.fitness = fitness_cache[sig]
            return self.fitness
        
        true_scores = [item['domain1_score'] for item in val_set]
        pred_scores = [0] * len(val_set)
        max_workers = self.config['evolution'].get('max_workers', 5)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.predict_score, item['essay_text'], vector_store, enable_rerank): i 
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
        self.fitness = qwk
        
        # [Optimization] 2. 写入缓存
        if fitness_cache is not None:
            fitness_cache[sig] = qwk
            
        return qwk

    def clone(self):
        return PromptIndividual(self.instruction_text, self.static_exemplars.copy(), self.fitness, self.config)
    
    def generate_real_feedback(self, val_set: List[Dict]) -> str:
        if not self.last_pred_scores: return "Focus on improving general accuracy."
        true_scores = [x['domain1_score'] for x in val_set]
        errors = []
        for i, (p, t) in enumerate(zip(self.last_pred_scores, true_scores)):
            if abs(p - t) >= 1: 
                errors.append((val_set[i], p, t))
        
        selected_errors = errors[:3]
        if not selected_errors: return "No significant errors found."

        error_desc = "\n".join([
            f"- Essay (ID {e['essay_id']}): Predicted {p}, but Truth was {t}. \nSnippet: {e['essay_text'][:200]}..."
            for e, p, t in selected_errors
        ])
        
        prompt = f"""Review the grading errors below and suggest improvements to the SCORING RUBRIC.
CURRENT RUBRIC:\n{self.instruction_text}
ERRORS:\n{error_desc}
Analyze WHY these errors occurred. Provide specific, actionable instructions to update the rubric.
"""
        temp = self.config['llm'].get('temperature_induce', 0.8)
        return call_llm(prompt, temperature=temp, call_type="reflection")
        
    def evolve_instruction(self, feedback):
        prompt = f"Rewrite and improve the rubric based on the feedback.\nFEEDBACK:\n{feedback}\n\nOLD RUBRIC:\n{self.instruction_text}\n\nOUTPUT ONLY THE NEW RUBRIC TEXT."
        temp = self.config['llm'].get('temperature_induce', 0.8)
        return call_llm(prompt, temperature=temp, call_type="rewrite")
    
    def to_dict(self):
        return {
            "fitness": self.fitness,
            "instruction_preview": self.instruction_text[:200] + "...",
            "full_instruction": self.instruction_text,
            "static_exemplar_ids": [ex['essay_id'] for ex in self.static_exemplars]
        }

# ============================================================================
# 4. 进化优化器 (EvolutionOptimizer)
# ============================================================================

class EvolutionOptimizer:
    def __init__(self, train_data, val_data, config):
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

        self.fitness_memo = {} # [New] 全局适应度缓存

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
            official_criteria=official_rubric,
            samples_text=samples_text
        )
        
        print("  Generating induced rubric via LLM...")
        temp = self.config['llm'].get('temperature_induce', 0.8)
        induced_rubric = call_llm(prompt, temperature=temp, call_type="induction")
        print(f"  [Induction Done] Rubric Length {len(induced_rubric)}")
        return induced_rubric

    # [NEW] 分层采样
    def get_stratified_exemplars(self, n=3):
        score_min = self.config['data']['score_min']
        score_max = self.config['data']['score_max']
        
        threshold_low = score_min + (score_max - score_min) * 0.33
        threshold_high = score_min + (score_max - score_min) * 0.66
        
        low_pool = [x for x in self.train_data if x['domain1_score'] <= threshold_low]
        mid_pool = [x for x in self.train_data if threshold_low < x['domain1_score'] < threshold_high]
        high_pool = [x for x in self.train_data if x['domain1_score'] >= threshold_high]
        
        selection = []
        if n > 0 and low_pool: selection.append(random.choice(low_pool))
        if len(selection) < n and high_pool: selection.append(random.choice(high_pool))
        if mid_pool and len(selection) < n: selection.append(random.choice(mid_pool))
        
        while len(selection) < n:
            selection.append(random.choice(self.train_data))
            
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

        scores = []
        pop_snapshot = [] 
        
        for i, ind in enumerate(self.population):
            ex_ids = [ex['essay_id'] for ex in ind.static_exemplars]
            print(f"  [Gen {gen}] Ind {i:02d} | Exemplars: {ex_ids}")
            
            # [MODIFIED] 透传 Rerank 开关
            qwk = ind.evaluate(self.val_data, self.vector_store, enable_rerank=use_rerank_train, fitness_cache=self.fitness_memo)
            scores.append(qwk)
            print(f"  >> Ind {i:02d} Finished: QWK = {qwk:.4f}\n")
            
            ind_data = ind.to_dict()
            ind_data['individual_id'] = i
            pop_snapshot.append(ind_data)
            
        best_idx = np.argmax(scores)
        best_ind = self.population[best_idx]
        print(f"  >> Gen {gen} Best QWK: {best_ind.fitness:.4f}")
        
        self.history.append({"gen": gen, "best_qwk": best_ind.fitness})
        
        # [NEW] Calculate Metrics
        gen_end_time = time.time()
        duration_sec = gen_end_time - gen_start_time
        
        curr_total = EXP_MANAGER.total_tokens if EXP_MANAGER else 0
        curr_prompt = EXP_MANAGER.total_prompt_tokens if EXP_MANAGER else 0
        curr_compl = EXP_MANAGER.total_completion_tokens if EXP_MANAGER else 0
        
        diff_total = curr_total - start_stats['total']
        diff_prompt = curr_prompt - start_stats['prompt']
        diff_compl = curr_compl - start_stats['completion']
        
        metrics = {
            "duration_sec": round(duration_sec, 2),
            "tokens_total": diff_total,
            "tokens_prompt": diff_prompt,
            "tokens_completion": diff_compl,
            "start_time": datetime.fromtimestamp(gen_start_time).isoformat(),
            "end_time": datetime.fromtimestamp(gen_end_time).isoformat()
        }
        
        print(f"  [Gen {gen} Stats] Duration: {duration_sec:.1f}s")
        print(f"    Tokens: Total {diff_total} (Prompt {diff_prompt} + Compl {diff_compl})")
        
        if EXP_MANAGER: 
            EXP_MANAGER.save_generation_snapshot(gen, pop_snapshot, metrics=metrics)
        
        # 2. Evolve Rubrics
        n_elite = self.config['evolution']['n_elite_evolve']
        elites = [self.population[i] for i in np.argsort(scores)[-n_elite:]]
        
        # [FIX] 关键修复：在修改 Rubric 之前，先保留上一代最好的个体（Elite Preservation）
        # 如果不保留，万一改坏了，种群分数会退化
        best_old_elite = elites[-1].clone() 
        new_pop = [best_old_elite] # 精英保留策略：直接晋级下一代
        
        print("  Evolving Rubrics for Elites (Using Real Reflection)...")
        mutated_parents = []
        
        # [NEW] Check config for instruction evolution
        evolve_instr = self.config['evolution'].get('evolve_instruction', True)
        if not evolve_instr:
             print("  [Config] Instruction evolution DISABLED. Skipping reflection/rewrite.")

        for elite in elites:
            # 这里是修改原始 elite 对象的引用
            if evolve_instr:
                real_feedback = elite.generate_real_feedback(self.val_data)
                print(f"    [Feedback] {real_feedback[:100]}...")
                new_text = elite.evolve_instruction(real_feedback)
                elite.instruction_text = new_text 
            mutated_parents.append(elite)
            
        # 补齐种群：从 mutated_parents 中克隆并变异 exemplars
        existing_fingerprints = {frozenset(ex['essay_id'] for ex in ind.static_exemplars) for ind in new_pop}
        
        # [NEW] Check config for exemplar evolution
        evolve_exemplars = self.config['evolution'].get('evolve_static_exemplars', True)
        if not evolve_exemplars:
             print("  [Config] Static Exemplar evolution DISABLED. Skipping mutation.")

        # 剩下的位置用变异填充
        while len(new_pop) < len(self.population):
            parent = random.choice(mutated_parents) # 选新 Rubric 的个体做父母
            child = parent.clone()
            
            if evolve_exemplars:
                for _ in range(10): 
                    idx = random.randint(0, len(child.static_exemplars)-1)
                    new_ex = random.choice(self.train_data)
                    child.static_exemplars[idx] = new_ex
                    
                    fp = frozenset(ex['essay_id'] for ex in child.static_exemplars)
                    if fp not in existing_fingerprints:
                        existing_fingerprints.add(fp)
                        break 
            
            new_pop.append(child)
            
        self.population = new_pop
        return best_ind

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
        best_global = None
        
        # 如果是从中间开始，先评估一下当前的 best_global
        if self.population:
             curr_best = max(self.population, key=lambda x: x.fitness)
             best_global = curr_best.clone()

        for g in range(start_gen, self.config['evolution']['n_generations']+1):
            best = self.evolve_one_generation(g)
            if not best_global or best.fitness > best_global.fitness:
                best_global = best.clone()
        return best_global

# ============================================================================
# 5. 主程序 (Main)
# ============================================================================

def main(config_path="configs/default.yaml", fold=0, resume_path=None):
    global EXP_MANAGER
    EXP_MANAGER = ExperimentManager(config_path=config_path, fold=fold, resume_path=resume_path)
    config = EXP_MANAGER.config
    
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
        n_train = config['debug'].get('n_train', 10)
        n_val = config['debug'].get('n_val', 5)
        random.shuffle(all_data)
        train_set = all_data[:n_train]
        val_set = all_data[n_train:n_train+n_val]
        test_set = all_data[n_train+n_val:n_train+n_val+5]
    else:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds = list(kf.split(all_data))
        train_val_idx, test_idx = folds[fold]
        
        split = int(len(train_val_idx) * 0.8)
        train_idx = train_val_idx[:split]
        val_idx = train_val_idx[split:]
        
        train_set = [all_data[i] for i in train_idx]
        val_set = [all_data[i] for i in val_idx]
        test_set = [all_data[i] for i in test_idx]
    
    print(f"  Split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    # 3. 运行优化
    print("[Main] Initializing Optimizer...")
    optimizer = EvolutionOptimizer(train_set, val_set, config)
    
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
    
    best = optimizer.run(start_gen=start_gen)
    
    # 4. 最终测试
    print(f"\n{'='*20} Final Test Evaluation {'='*20}")
    
    # [NEW] 读取推理时的 Rerank 开关
    use_rerank_test = config['rag'].get('use_rerank_test', True)
    print(f"  [Config] Inference Rerank: {'ENABLED' if use_rerank_test else 'DISABLED'}")
    
    test_qwk = best.evaluate(test_set, optimizer.vector_store, enable_rerank=use_rerank_test)
    
    print(f"Validation Best QWK: {best.fitness:.4f}")
    print(f"Test Set QWK:        {test_qwk:.4f}")
    
    # [NEW] Final Global Stats
    total_duration = time.time() - EXP_MANAGER.exp_start_time
    print(f"\n{'='*20} Experiment Summary {'='*20}")
    print(f"Total Duration:      {total_duration / 60:.1f} minutes")
    print(f"Total Tokens:        {EXP_MANAGER.total_tokens}")
    print(f"  - Prompt:          {EXP_MANAGER.total_prompt_tokens}")
    print(f"  - Completion:      {EXP_MANAGER.total_completion_tokens}")
    
    EXP_MANAGER.save_final_results(best, optimizer.history, test_results={
        "test_qwk": test_qwk,
        "total_duration_sec": total_duration,
        "total_tokens": EXP_MANAGER.total_tokens,
        "total_prompt_tokens": EXP_MANAGER.total_prompt_tokens,
        "total_completion_tokens": EXP_MANAGER.total_completion_tokens
    })

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to generation json file (e.g., logs/exp.../generations/gen_010.json) to resume from.")
    args = parser.parse_args()
    main(args.config, args.fold, args.resume)
