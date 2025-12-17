"""
WISE-AES: Weakly-supervised Integrated Scoring Evolution
版本: v2.3 (With Item-level Scoring Logs)
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
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(override=True)

# ============================================================================
# 0. 基础设施: 双向日志系统 & 实验管理
# ============================================================================

class TeeLogger(object):
    """双向日志：同时写控制台和文件"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

class ExperimentManager:
    """管理实验目录、配置、日志和结果保存"""
    def __init__(self, base_dir="logs", config_path="configs/default.yaml"):
        # 1. 创建目录结构
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = Path(base_dir) / f"exp_{self.timestamp}"
        
        # 创建子目录: generations 用于存放每一代的详细 JSON
        self.gens_dir = self.exp_dir / "generations"
        self.gens_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. 设置日志
        self.console_log_path = self.exp_dir / f"console_{self.timestamp}.log"
        sys.stdout = TeeLogger(self.console_log_path)
        
        print(f"=== Experiment Started: {self.timestamp} ===")
        print(f"Result Directory: {self.exp_dir}")
        
        self.llm_trace_path = self.exp_dir / "llm_trace.jsonl"
        self.config = self._load_and_save_config(config_path)
        
    def _load_and_save_config(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with open(self.exp_dir / "config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        return config

    def log_llm_trace(self, record: Dict):
        with open(self.llm_trace_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def save_generation_snapshot(self, generation: int, population_data: List[Dict]):
        """保存当前代所有个体的信息"""
        filename = self.gens_dir / f"gen_{generation:03d}.json"
        
        snapshot = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
            "best_qwk": max(p['fitness'] for p in population_data),
            "population": population_data
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, ensure_ascii=False, indent=2)

    def save_final_results(self, best_ind, history):
        res = {
            "best_qwk": best_ind.fitness,
            "instruction": best_ind.instruction_text,
            "static_exemplars": [ex['essay_id'] for ex in best_ind.static_exemplars],
            "history": history
        }
        with open(self.exp_dir / "final_result.json", 'w', encoding='utf-8') as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"\nFinal results saved to {self.exp_dir}")

EXP_MANAGER = None

# ============================================================================
# 1. 基础设施: 向量数据库 (Simple Vector Store)
# ============================================================================
class SimpleVectorStore:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "cache"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model = None 

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def add_documents(self, data: List[Dict[str, Any]]):
        self.documents = data
        doc_ids = [str(d['essay_id']) for d in data]
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}_{len(data)}.pkl"
        
        if cache_file.exists():
            print(f"[VectorStore] Loading cached embeddings from {cache_file}")
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
        self._load_model()
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
        with sqlite3.connect(self.db_path) as conn:
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
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("SELECT response FROM cache WHERE hash_key=?", (key,))
                    row = cursor.fetchone()
                    if row: return row[0]
        except Exception: return None
        return None

    def set(self, prompt: str, model: str, temperature: float, response: str):
        key = self._get_hash(prompt, model, temperature)
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("INSERT OR REPLACE INTO cache VALUES (?, ?, ?, ?)", 
                                 (key, prompt, response, datetime.now().isoformat()))
        except Exception as e: print(f"[Cache Write Error] {e}")

CACHE = LLMCache()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

def call_llm(prompt: str, temperature: float = 0.0, call_type: str = "unknown") -> str:
    model_name = EXP_MANAGER.config['model']['name']
    
    # 1. 缓存读取
    cached_resp = CACHE.get(prompt, model_name, temperature)
    if cached_resp: return cached_resp

    # 2. API 调用
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json", "HTTP-Referer": "https://wise-aes.local"}
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "provider": {"order": ["cerebras", "deepinfra"]}
    }
    
    start_time = time.time()
    response_content = ""
    error_msg = ""
    
    for attempt in range(3):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            response_content = resp.json()["choices"][0]["message"]["content"]
            CACHE.set(prompt, model_name, temperature, response_content)
            break
        except Exception as e:
            error_msg = str(e)
            time.sleep(1)
    
    # 3. 日志记录
    if EXP_MANAGER:
        EXP_MANAGER.log_llm_trace({
            "timestamp": datetime.now().isoformat(),
            "call_type": call_type,
            "duration": round(time.time() - start_time, 3),
            "prompt_len": len(prompt),
            "response_len": len(response_content),
            "error": error_msg,
            "prompt_preview": prompt[:200],
            "response_preview": response_content[:200]
        })
            
    return response_content

# ============================================================================
# 3. PromptIndividual
# ============================================================================

@dataclass
class PromptIndividual:
    instruction_text: str
    static_exemplars: List[Dict[str, Any]] = field(default_factory=list)
    fitness: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    
    QUERY_GEN_RUBRIC_TEMPLATE = """Based on the SCORING RUBRIC below, extract 3-5 keywords from the STUDENT ESSAY that relate to critical grading features.
SCORING RUBRIC:\n{rubric}\nSTUDENT ESSAY:\n{essay}\nOUTPUT JSON LIST:"""

    QUERY_GEN_GENERIC_TEMPLATE = """Extract 3-5 distinct keywords from the STUDENT ESSAY that capture its main topic and writing style.\nSTUDENT ESSAY:\n{essay}\nOUTPUT JSON LIST:"""

    RERANK_RUBRIC_TEMPLATE = """Select the {k} essays from CANDIDATES that best demonstrate the specific grading criteria in the SCORING RUBRIC.\nSCORING RUBRIC:\n{rubric}\nTARGET ESSAY SUMMARY:\n{essay_summary}\nCANDIDATES:\n{candidates}\nOUTPUT JSON LIST OF IDs:"""

    RERANK_GENERIC_TEMPLATE = """Select the {k} essays from CANDIDATES that are most semantically similar to the TARGET ESSAY.\nTARGET ESSAY SUMMARY:\n{essay_summary}\nCANDIDATES:\n{candidates}\nOUTPUT JSON LIST OF IDs:"""

    SCORING_TEMPLATE = """You are an expert essay grader. Score the essay (0-60).
## SCORING RUBRIC:\n{instruction}
## STATIC REFERENCE EXAMPLES (Global Anchors):\n{static_ex}
## RETRIEVED SIMILAR EXAMPLES (Local Context):\n{dynamic_ex}
## ESSAY TO SCORE:\n{essay}
Output ONLY the numeric score.\nSCORE:"""

    def __post_init__(self):
        if not self.config: self.config = EXP_MANAGER.config

    def generate_query(self, essay_text: str) -> str:
        rubric_driven = self.config['rag']['rubric_driven_retrieval']
        template = self.QUERY_GEN_RUBRIC_TEMPLATE if rubric_driven else self.QUERY_GEN_GENERIC_TEMPLATE
        prompt = template.format(rubric=self.instruction_text, essay=essay_text[:800])
        response = call_llm(prompt, temperature=0.7, call_type="rag_query")
        try:
            keywords = json.loads(re.search(r'\[.*\]', response, re.DOTALL).group())
            return " ".join(keywords)
        except: return essay_text[:200]

    def rerank_exemplars(self, essay_text: str, candidates: List[Dict]) -> List[Dict]:
        if not self.config['rag']['use_rerank']: return candidates[:self.config['rag']['n_selected']]
        
        rubric_driven = self.config['rag']['rubric_driven_retrieval']
        k_select = self.config['rag']['n_selected']
        cand_text = "\n".join([f"ID {c['essay_id']} (Score {c['domain1_score']}): {c['essay_text'][:200]}..." for c in candidates])
        
        if rubric_driven:
            prompt = self.RERANK_RUBRIC_TEMPLATE.format(rubric=self.instruction_text, k=k_select, essay_summary=essay_text[:500], candidates=cand_text)
        else:
            prompt = self.RERANK_GENERIC_TEMPLATE.format(k=k_select, essay_summary=essay_text[:500], candidates=cand_text)
            
        response = call_llm(prompt, temperature=0.0, call_type="rag_rerank")
        try:
            ids = json.loads(re.search(r'\[.*\]', response, re.DOTALL).group())
            selected = [c for c in candidates if c['essay_id'] in ids]
            if len(selected) < k_select:
                remain = [c for c in candidates if c not in selected]
                selected.extend(remain[:k_select - len(selected)])
            return selected[:k_select]
        except: return candidates[:k_select]

    def predict_score(self, essay_text: str, vector_store: SimpleVectorStore) -> int:
        dynamic_exemplars = []
        if self.config['rag']['enabled']:
            query = self.generate_query(essay_text)
            exclude_ids = {ex['essay_id'] for ex in self.static_exemplars}
            candidates = vector_store.search(query, top_k=self.config['rag']['n_retrieved'], exclude_ids=exclude_ids)
            dynamic_exemplars = self.rerank_exemplars(essay_text, candidates)
            
        prompt = self.SCORING_TEMPLATE.format(
            instruction=self.instruction_text,
            static_ex=self._format_list(self.static_exemplars),
            dynamic_ex=self._format_list(dynamic_exemplars) if dynamic_exemplars else "(None)",
            essay=essay_text
        )
        response = call_llm(prompt, temperature=self.config['llm']['temperature_scoring'], call_type="scoring")
        try:
            nums = re.findall(r'\b(60|[1-5]?[0-9])\b', response)
            if nums: return int(nums[-1])
        except: pass
        return 30

    def _format_list(self, exs):
        return "\n\n".join([f"### Essay (Score: {ex['domain1_score']})\n{ex['essay_text'][:300]}..." for ex in exs])

    # ---------------- [UPDATED] 带逐项输出的评估 ----------------
    def evaluate(self, val_set: List[Dict], vector_store: SimpleVectorStore) -> float:
        true_scores = [item['domain1_score'] for item in val_set]
        pred_scores = [0] * len(val_set)
        max_workers = self.config['evolution'].get('max_workers', 5)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.predict_score, item['essay_text'], vector_store): i for i, item in enumerate(val_set)}
            
            for future in tqdm(as_completed(future_to_idx), total=len(val_set), desc="  Evaluating", leave=False):
                idx = future_to_idx[future]
                try: 
                    pred = future.result()
                    pred_scores[idx] = pred
                    
                    # [NEW] 打印每个个体的打分结果 (使用 tqdm.write 避免破坏进度条)
                    true_score = true_scores[idx]
                    essay_id = val_set[idx].get('essay_id', 'unknown')
                    tqdm.write(f"    [Eval] ID {essay_id}: Truth={true_score} vs Pred={pred}")
                    
                except: 
                    pred_scores[idx] = 30
                    
        qwk = self._calc_qwk(true_scores, pred_scores)
        self.fitness = qwk
        return qwk

    def _calc_qwk(self, t, p):
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(t, p, weights='quadratic')

    def clone(self):
        return PromptIndividual(self.instruction_text, self.static_exemplars.copy(), self.fitness, self.config)
    
    def get_feedback(self, val_set, preds):
        prompt = f"Analyze errors based on Rubric:\n{self.instruction_text}\n..." 
        return call_llm(prompt, temperature=0.7, call_type="feedback")
        
    def evolve_instruction(self, feedback):
        prompt = f"Rewrite rubric based on feedback:\n{feedback}\nOld Rubric:\n{self.instruction_text}"
        return call_llm(prompt, temperature=0.7, call_type="rewrite")
    
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
        print("[Optimizer] Building Vector Store...")
        self.vector_store.add_documents(self.train_data)
        
        self.population = []
        self.history = []

    def initialize_population(self, base_instruction):
        n_static = self.config['evolution']['n_static_exemplars']
        print(f"[Init] Pop size: {self.config['evolution']['population_size']}, Static Ex: {n_static}")
        for _ in range(self.config['evolution']['population_size']):
            static_ex = random.sample(self.train_data, n_static)
            self.population.append(PromptIndividual(base_instruction, static_ex, config=self.config))

    def evolve_one_generation(self, gen):
        print(f"\n{'='*20} Generation {gen} {'='*20}")
        
        scores = []
        pop_snapshot = [] 
        
        for i, ind in enumerate(self.population):
            # 这里的 print 会同时进入 log
            print(f"  [Gen {gen}] Evaluating Individual {i:02d}...")
            qwk = ind.evaluate(self.val_data, self.vector_store)
            scores.append(qwk)
            
            print(f"  >> Ind {i:02d} Finished: QWK = {qwk:.4f}\n")
            
            ind_data = ind.to_dict()
            ind_data['individual_id'] = i
            pop_snapshot.append(ind_data)
            
        best_idx = np.argmax(scores)
        best_ind = self.population[best_idx]
        print(f"  >> Gen {gen} Best QWK: {best_ind.fitness:.4f}")
        
        self.history.append({"gen": gen, "best_qwk": best_ind.fitness})
        
        if EXP_MANAGER:
            EXP_MANAGER.save_generation_snapshot(gen, pop_snapshot)
        
        # 2. Evolve Rubrics (Elite Only)
        n_elite = self.config['evolution']['n_elite_evolve']
        elites = [self.population[i] for i in np.argsort(scores)[-n_elite:]]
        
        print("  Evolving Rubrics for Elites...")
        for elite in elites:
            dummy_feedback = "Please make the rubric more specific about grammar."
            new_text = elite.evolve_instruction(dummy_feedback)
            elite.instruction_text = new_text 
            
        # 3. GA for Exemplars
        new_pop = [e.clone() for e in elites]
        while len(new_pop) < len(self.population):
            parent = random.choice(elites)
            child = parent.clone()
            if random.random() < self.config['evolution']['mutation_rate']:
                idx = random.randint(0, len(child.static_exemplars)-1)
                child.static_exemplars[idx] = random.choice(self.train_data)
            new_pop.append(child)
            
        self.population = new_pop
        return best_ind

    def run(self):
        best_global = None
        for g in range(1, self.config['evolution']['n_generations']+1):
            best = self.evolve_one_generation(g)
            if not best_global or best.fitness > best_global.fitness:
                best_global = best.clone()
        return best_global

# ============================================================================
# 5. 主程序 (Main)
# ============================================================================

def main():
    global EXP_MANAGER
    EXP_MANAGER = ExperimentManager(config_path="configs/default.yaml")
    
    print("[Main] Loading Data (Simulated)...")
    raw_data = [{"essay_id": i, "essay_text": f"Simulated essay {i} content...", "domain1_score": random.randint(10, 60)} for i in range(50)]
    
    train = raw_data[:30]
    val = raw_data[30:]
    
    # 1. 诱导
    print("[Main] Inducing Initial Rubric...")
    base_rubric = "Score based on details and emotion."
    
    # 2. 进化
    optimizer = EvolutionOptimizer(train, val, EXP_MANAGER.config)
    optimizer.initialize_population(base_rubric)
    best = optimizer.run()
    
    print(f"\nFinal Best QWK: {best.fitness}")
    
    EXP_MANAGER.save_final_results(best, optimizer.history)

if __name__ == "__main__":
    main()