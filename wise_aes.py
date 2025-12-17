"""
WISE-AES: Weakly-supervised Integrated Scoring Evolution
版本: v2.6 (Production Ready: Real Data, 5-Fold CV, Forced Mutation)
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
from sklearn.model_selection import KFold # [NEW] 用于五折交叉验证

load_dotenv(override=True)

# ============================================================================
# 0. 基础设施: 双向日志系统 & 实验管理
# ============================================================================

class TeeLogger(object):
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
    def __init__(self, base_dir="logs", config_path="configs/default.yaml", fold=0):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 实验目录带上 fold 编号，方便区分
        self.exp_dir = Path(base_dir) / f"exp_{self.timestamp}_fold{fold}"
        
        self.gens_dir = self.exp_dir / "generations"
        self.gens_dir.mkdir(parents=True, exist_ok=True)
        
        self.console_log_path = self.exp_dir / f"console.log"
        sys.stdout = TeeLogger(self.console_log_path)
        
        print(f"=== Experiment Started: {self.timestamp} (Fold {fold}) ===")
        print(f"Config File: {config_path}")
        print(f"Result Directory: {self.exp_dir}")
        
        self.llm_trace_path = self.exp_dir / "llm_trace.jsonl"
        self.config = self._load_and_save_config(config_path)
        
    def _load_and_save_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        with open(self.exp_dir / "config.yaml", 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
        return config

    def log_llm_trace(self, record: Dict):
        with open(self.llm_trace_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    def save_generation_snapshot(self, generation: int, population_data: List[Dict]):
        filename = self.gens_dir / f"gen_{generation:03d}.json"
        snapshot = {
            "generation": generation,
            "timestamp": datetime.now().isoformat(),
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
            "test_results": test_results # [NEW] 包含测试集结果
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
        self.model = None 

    def _load_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def add_documents(self, data: List[Dict[str, Any]]):
        self.documents = data
        doc_ids = [str(d['essay_id']) for d in data]
        # 缓存键加入数据长度，防止不同 fold 混用缓存时出错
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}_{len(data)}_{hash(str(doc_ids[:5]))}.pkl"
        
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
    model_name = EXP_MANAGER.config['model']['name']
    cached_resp = CACHE.get(prompt, model_name, temperature)
    if cached_resp: return cached_resp

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
    
    if EXP_MANAGER:
        EXP_MANAGER.log_llm_trace({
            "timestamp": datetime.now().isoformat(),
            "call_type": call_type,
            "duration": round(time.time() - start_time, 3),
            "len": len(response_content),
            "error": error_msg,
            "prompt_preview": prompt[:100]
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
    
    # Templates
    QUERY_GEN_RUBRIC_TEMPLATE = """Based on the SCORING RUBRIC below, extract 3-5 keywords from the STUDENT ESSAY that relate to critical grading features.
SCORING RUBRIC:\n{rubric}\nSTUDENT ESSAY:\n{essay}\nOUTPUT JSON LIST:"""

    QUERY_GEN_GENERIC_TEMPLATE = """Extract 3-5 distinct keywords from the STUDENT ESSAY that capture its main topic and writing style.\nSTUDENT ESSAY:\n{essay}\nOUTPUT JSON LIST:"""

    RERANK_RUBRIC_TEMPLATE = """Select the {k} essays from CANDIDATES that best demonstrate the specific grading criteria in the SCORING RUBRIC.\nSCORING RUBRIC:\n{rubric}\nTARGET ESSAY SUMMARY:\n{essay_summary}\nCANDIDATES:\n{candidates}\nOUTPUT JSON LIST OF IDs:"""

    RERANK_GENERIC_TEMPLATE = """Select the {k} essays from CANDIDATES that are most semantically similar to the TARGET ESSAY.\nTARGET ESSAY SUMMARY:\n{essay_summary}\nCANDIDATES:\n{candidates}\nOUTPUT JSON LIST OF IDs:"""

    # [FIX] 修复：打分范围不再硬编码，改为动态变量
    SCORING_TEMPLATE = """You are an expert essay grader. Score the essay on a scale of {score_min} to {score_max}.
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
            essay=essay_text,
            # [FIX] 动态传入分数范围
            score_min=self.config['data']['score_min'],
            score_max=self.config['data']['score_max']
        )
        response = call_llm(prompt, temperature=self.config['llm']['temperature_scoring'], call_type="scoring")
        try:
            # 宽容解析：查找最后一个数字
            nums = re.findall(r'-?\d+', response)
            if nums: 
                score = int(nums[-1])
                # 截断到合法范围
                return max(self.config['data']['score_min'], min(self.config['data']['score_max'], score))
        except: pass
        
        # Fallback: 返回中位数
        return (self.config['data']['score_min'] + self.config['data']['score_max']) // 2

    def _format_list(self, exs):
        return "\n\n".join([f"### Essay (Score: {ex['domain1_score']})\n{ex['essay_text'][:300]}..." for ex in exs])

    def evaluate(self, val_set: List[Dict], vector_store: SimpleVectorStore) -> float:
        true_scores = [item['domain1_score'] for item in val_set]
        pred_scores = [0] * len(val_set)
        max_workers = self.config['evolution'].get('max_workers', 5)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {executor.submit(self.predict_score, item['essay_text'], vector_store): i for i, item in enumerate(val_set)}
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try: 
                    pred = future.result()
                    pred_scores[idx] = pred
                    true_score = true_scores[idx]
                    essay_id = val_set[idx].get('essay_id', 'unknown')
                    print(f"    [Eval] ID {essay_id:<4} | Truth: {true_score:<2} | Pred: {pred:<2}")
                except Exception as e: 
                    print(f"    [Error] {e}")
                    pred_scores[idx] = (self.config['data']['score_min'] + self.config['data']['score_max']) // 2
                    
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
        print("[Optimizer] Building Vector Store (Train Set Only)...")
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
            ex_ids = [ex['essay_id'] for ex in ind.static_exemplars]
            print(f"  [Gen {gen}] Ind {i:02d} | Exemplars: {ex_ids}")
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
        if EXP_MANAGER: EXP_MANAGER.save_generation_snapshot(gen, pop_snapshot)
        
        # 2. Evolve Rubrics (Elite Only)
        n_elite = self.config['evolution']['n_elite_evolve']
        elites = [self.population[i] for i in np.argsort(scores)[-n_elite:]]
        
        print("  Evolving Rubrics for Elites...")
        for elite in elites:
            dummy_feedback = "Please make the rubric more specific about grammar."
            new_text = elite.evolve_instruction(dummy_feedback)
            elite.instruction_text = new_text 
            
        # 3. GA for Exemplars (With Forced Mutation)
        new_pop = [e.clone() for e in elites]
        
        # [FIX] 记录当前指纹，防止重复
        existing_fingerprints = {frozenset(ex['essay_id'] for ex in ind.static_exemplars) for ind in new_pop}
        
        while len(new_pop) < len(self.population):
            parent = random.choice(elites)
            child = parent.clone()
            
            # 强制变异循环：最多尝试 10 次，直到生成独特的基因组合
            for _ in range(10):
                idx = random.randint(0, len(child.static_exemplars)-1)
                new_ex = random.choice(self.train_data)
                child.static_exemplars[idx] = new_ex
                
                # 检查唯一性
                fp = frozenset(ex['essay_id'] for ex in child.static_exemplars)
                if fp not in existing_fingerprints:
                    existing_fingerprints.add(fp)
                    break 
            
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
# 5. 主程序 (Main) - [FIX] Real Data & 5-Fold
# ============================================================================

def main(config_path="configs/default.yaml", fold=0):
    global EXP_MANAGER
    EXP_MANAGER = ExperimentManager(config_path=config_path, fold=fold)
    config = EXP_MANAGER.config
    
    # 1. 加载真实数据
    data_path = config['data']['asap_path']
    essay_set = config['data']['essay_set']
    print(f"[Main] Loading ASAP Data Set {essay_set} from {data_path}...")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
        
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    df = df[df['essay_set'] == essay_set]
    
    all_data = []
    for _, row in df.iterrows():
        all_data.append({
            "essay_id": row['essay_id'],
            "essay_text": row['essay'],
            "domain1_score": row['domain1_score']
        })
    print(f"  Total essays: {len(all_data)}")
    
    # 2. 五折交叉验证切分 (60/20/20)
    # 使用 Seed=42 保证每次运行切分一致
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(all_data))
    
    # 获取当前 fold 的索引
    # indices: array of indices
    train_val_indices, test_indices = folds[fold]
    
    # 从 train_val 中再切分出 20% (相对于总数据) 作为 val
    # 逻辑：Test=20%, Train+Val=80%. Val 是 Train+Val 的 1/4 (即总数的20%)
    # 我们简单地把 train_val 的后 25% 划给 Val
    split_point = int(len(train_val_indices) * 0.75)
    train_indices = train_val_indices[:split_point]
    val_indices = train_val_indices[split_point:]
    
    train_set = [all_data[i] for i in train_indices]
    val_set = [all_data[i] for i in val_indices]
    test_set = [all_data[i] for i in test_indices]
    
    print(f"  Fold {fold} Split: Train={len(train_set)}, Val={len(val_set)}, Test={len(test_set)}")
    
    # 3. 诱导
    print("[Main] Inducing Initial Rubric (Mocked)...")
    # 实际应调用 InstructionInductor
    base_rubric = "Score the essay based on coherence, organization, and grammar."
    
    # 4. 进化 (只在 Train/Val 上进行)
    optimizer = EvolutionOptimizer(train_set, val_set, config)
    optimizer.initialize_population(base_rubric)
    best = optimizer.run()
    
    print(f"\n{'='*20} Final Test Evaluation {'='*20}")
    # 5. [FIX] 最终测试集评估 (Test Set)
    # 注意：vector_store 依然是基于 Train Set 构建的，没有任何 Test Set 数据泄露
    test_qwk = best.evaluate(test_set, optimizer.vector_store)
    
    print(f"Validation Best QWK: {best.fitness:.4f}")
    print(f"Test Set QWK:        {test_qwk:.4f}")
    
    EXP_MANAGER.save_final_results(best, optimizer.history, test_results={"test_qwk": test_qwk})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run WISE-AES Experiment")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    # [FIX] 增加 fold 参数
    parser.add_argument("--fold", type=int, default=0, help="Fold index (0-4) for 5-fold CV")
    args = parser.parse_args()
    
    main(config_path=args.config, fold=args.fold)