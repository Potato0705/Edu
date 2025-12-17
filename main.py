"""
WISE-AES: Weakly-supervised Integrated Scoring Evolution
实现策略: One Rubric to rule them all (Rubric 驱动的 RAG)
"""

import os
import re
import random
import json
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
import numpy as np
import yaml
import pandas as pd
import pickle

# 新增依赖
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv(override=True)

# ============================================================================
# 0. 基础设施: 日志与配置 (保持原样，略微精简以突出重点)
# ============================================================================

class ExperimentLogger:
    """(保持原有的 Logger 代码不变，此处省略以节省篇幅，实际运行时请保留原代码)"""
    def __init__(self, result_dir: str = "result", resume_dir: str = None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.result_dir / f"exp_{self.timestamp}" if not resume_dir else Path(resume_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.exp_dir / "experiment.log"
        self.logs = []

    def _log(self, message: str, also_print: bool = True):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"[{timestamp}] {message}"
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(entry + "\n")
        if also_print: print(message)

    def log_config(self, config, name="config.yaml"):
        with open(self.exp_dir / name, 'w', encoding='utf-8') as f:
            yaml.dump(config, f)
            
    def save_final_results(self, best_ind, history, config):
        res = {
            "best_qwk": best_ind.fitness,
            "instruction": best_ind.instruction_text,
            "static_exemplars": [ex['essay_id'] for ex in best_ind.static_exemplars],
            "history": history
        }
        with open(self.exp_dir / "final_result.json", 'w') as f:
            json.dump(res, f, indent=2)

LOGGER = None

def load_config(path):
    with open(path, 'r', encoding='utf-8') as f: return yaml.safe_load(f)

# ============================================================================
# 1. 基础设施: 向量数据库 (Simple Vector Store)
# ============================================================================

class SimpleVectorStore:
    """基于内存的简单向量检索库"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: str = "cache"):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.documents: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None
        self.model = None # 延迟加载
        
    def _load_model(self):
        if self.model is None:
            print(f"[VectorStore] Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)

    def add_documents(self, data: List[Dict[str, Any]]):
        """添加文档并计算/加载 Embeddings"""
        self.documents = data
        doc_ids = [str(d['essay_id']) for d in data]
        cache_file = self.cache_dir / f"embeddings_{self.model_name.replace('/', '_')}_{len(data)}.pkl"
        
        if cache_file.exists():
            print(f"[VectorStore] Loading cached embeddings from {cache_file}")
            with open(cache_file, 'rb') as f:
                saved_ids, self.embeddings = pickle.load(f)
            # 简单校验 ID 是否一致
            if saved_ids != doc_ids:
                print("[VectorStore] Cache mismatch! Recomputing...")
                self._compute_and_save(data, doc_ids, cache_file)
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
        """检索 Top-K"""
        self._load_model()
        query_vec = self.model.encode([query], convert_to_numpy=True)
        
        # 计算余弦相似度
        sims = cosine_similarity(query_vec, self.embeddings)[0]
        
        # 排序
        sorted_indices = np.argsort(sims)[::-1]
        
        results = []
        for idx in sorted_indices:
            doc = self.documents[idx]
            if exclude_ids and doc['essay_id'] in exclude_ids:
                continue
            
            # 避免返回自己 (如果是训练集内检索)
            # 实际应用中可能不需要这步，视情况而定
            
            results.append(doc)
            if len(results) >= top_k:
                break
                
        return results

# ============================================================================
# 2. LLM 调用 (OpenRouter)
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CONFIG = load_config("configs/default.yaml") # 默认加载，main中会覆盖
OPENROUTER_MODEL = CONFIG['model']['name']

def call_llm(prompt: str, temperature: float = 0.0, call_type: str = "unknown") -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://wise-aes.local",
        "X-Title": "WISE-AES"
    }
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "provider": {"order": ["cerebras", "deepinfra"]} # 偏好速度快的
    }
    
    for _ in range(3): # 重试 3 次
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if LOGGER: LOGGER.log_llm_call(call_type, prompt, content, temperature)
            return content
        except Exception as e:
            print(f"[LLM Error] {e}, retrying...")
            time.sleep(1)
    return ""

# ============================================================================
# 3. 核心个体类: PromptIndividual (One Rubric Implementation)
# ============================================================================

@dataclass
class PromptIndividual:
    """
    进化个体: 
    - 核心基因: instruction_text (评分标准)
    - 静态组件: static_exemplars (Few-shot)
    - 动态能力: 通过 Rubric 驱动 Query 生成和 Rerank
    """
    
    instruction_text: str
    static_exemplars: List[Dict[str, Any]] = field(default_factory=list)
    fitness: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    
    # ---------------- Meta-Prompts (One Rubric Strategy) ----------------
    
    # 用于生成检索词 (Rubric-Driven)
    QUERY_GEN_RUBRIC_TEMPLATE = """
Based on the SCORING RUBRIC below, identify 3-5 specific linguistic or structural features that are most critical for grading.
Then, extract keywords from the STUDENT ESSAY that relate to these features.
Output ONLY a JSON list of keywords.

## SCORING RUBRIC:
{rubric}

## STUDENT ESSAY:
{essay}

OUTPUT JSON:"""

    # 用于生成检索词 (Standard / Generic)
    QUERY_GEN_GENERIC_TEMPLATE = """
Extract 3-5 distinct keywords from the STUDENT ESSAY that capture its main topic and writing style.
Output ONLY a JSON list of keywords.

## STUDENT ESSAY:
{essay}

OUTPUT JSON:"""

    # 用于重排序 (Rubric-Driven)
    RERANK_RUBRIC_TEMPLATE = """
You are selecting reference examples to help score a target essay.
Based on the SCORING RUBRIC, select the {k} essays from the CANDIDATES list that best demonstrate the specific grading criteria (e.g., similar errors, similar proficiency level).
Output ONLY a JSON list of the selected essay IDs.

## SCORING RUBRIC:
{rubric}

## TARGET ESSAY SUMMARY:
{essay_summary}

## CANDIDATES:
{candidates}

OUTPUT JSON:"""

    # 用于重排序 (Standard / Generic)
    RERANK_GENERIC_TEMPLATE = """
Select the {k} essays from the CANDIDATES list that are most semantically similar to the TARGET ESSAY.
Output ONLY a JSON list of the selected essay IDs.

## TARGET ESSAY SUMMARY:
{essay_summary}

## CANDIDATES:
{candidates}

OUTPUT JSON:"""

    # 最终打分 Prompt
    SCORING_TEMPLATE = """You are an expert essay grader. Score the essay (0-60).

## SCORING RUBRIC:
{instruction}

## STATIC REFERENCE EXAMPLES (Global Anchors):
{static_ex}

## RETRIEVED SIMILAR EXAMPLES (Local Context):
{dynamic_ex}

## ESSAY TO SCORE:
{essay}

Output ONLY the numeric score.
SCORE:"""

    def __post_init__(self):
        if not self.config: self.config = CONFIG

    # ---------------- 核心功能: Rubric 驱动的 RAG 流程 ----------------

    def generate_query(self, essay_text: str) -> str:
        """生成检索 Query"""
        # [配置检查] 是否使用进化 Rubric 驱动检索
        rubric_driven = self.config['rag']['rubric_driven_retrieval']
        
        if rubric_driven:
            prompt = self.QUERY_GEN_RUBRIC_TEMPLATE.format(
                rubric=self.instruction_text,
                essay=essay_text[:800]
            )
        else:
            prompt = self.QUERY_GEN_GENERIC_TEMPLATE.format(
                essay=essay_text[:800]
            )
            
        response = call_llm(prompt, temperature=0.7, call_type="rag_query")
        try:
            keywords = json.loads(re.search(r'\[.*\]', response, re.DOTALL).group())
            return " ".join(keywords)
        except:
            return essay_text[:200] # Fallback: 使用原文开头

    def rerank_exemplars(self, essay_text: str, candidates: List[Dict]) -> List[Dict]:
        """对检索结果进行重排序"""
        # [配置检查] 是否启用 Rerank
        if not self.config['rag']['use_rerank']:
            return candidates[:self.config['rag']['n_selected']]
            
        # [配置检查] 是否使用进化 Rubric 驱动 Rerank
        rubric_driven = self.config['rag']['rubric_driven_retrieval']
        k_select = self.config['rag']['n_selected']
        
        # 格式化候选项供 LLM 阅读
        cand_text = "\n".join([
            f"ID {c['essay_id']} (Score {c['domain1_score']}): {c['essay_text'][:200]}..."
            for c in candidates
        ])
        
        if rubric_driven:
            prompt = self.RERANK_RUBRIC_TEMPLATE.format(
                rubric=self.instruction_text,
                k=k_select,
                essay_summary=essay_text[:500],
                candidates=cand_text
            )
        else:
            prompt = self.RERANK_GENERIC_TEMPLATE.format(
                k=k_select,
                essay_summary=essay_text[:500],
                candidates=cand_text
            )
            
        response = call_llm(prompt, temperature=0.0, call_type="rag_rerank")
        
        selected = []
        try:
            ids = json.loads(re.search(r'\[.*\]', response, re.DOTALL).group())
            # 保持原始顺序或 LLM 推荐顺序? 这里简单起见，按 ID 匹配回原始对象
            selected = [c for c in candidates if c['essay_id'] in ids]
        except:
            pass
            
        # Fallback: 如果解析失败或选少了，补齐 Top-k
        if len(selected) < k_select:
            remain = [c for c in candidates if c not in selected]
            selected.extend(remain[:k_select - len(selected)])
            
        return selected[:k_select]

    # ---------------- 主流程: 预测分数 ----------------

    def predict_score(self, essay_text: str, vector_store: SimpleVectorStore) -> int:
        dynamic_exemplars = []
        
        # [配置检查] 是否启用 RAG
        if self.config['rag']['enabled']:
            # 1. 生成 Query
            query = self.generate_query(essay_text)
            
            # 2. 向量检索 (Retrieve)
            # 排除静态样例，避免重复
            exclude_ids = {ex['essay_id'] for ex in self.static_exemplars}
            candidates = vector_store.search(
                query, 
                top_k=self.config['rag']['n_retrieved'], 
                exclude_ids=exclude_ids
            )
            
            # 3. 重排序 (Rerank / Select)
            dynamic_exemplars = self.rerank_exemplars(essay_text, candidates)
            
        # 4. 构建最终 Prompt
        prompt = self.SCORING_TEMPLATE.format(
            instruction=self.instruction_text,
            static_ex=self._format_list(self.static_exemplars),
            dynamic_ex=self._format_list(dynamic_exemplars) if dynamic_exemplars else "(None)",
            essay=essay_text
        )
        
        # 5. 打分
        response = call_llm(prompt, temperature=self.config['llm']['temperature_scoring'], call_type="scoring")
        
        # 解析分数
        try:
            nums = re.findall(r'\b(60|[1-5]?[0-9])\b', response)
            if nums: return int(nums[-1])
        except: pass
        return 30 # Fallback mean score

    def _format_list(self, exs):
        return "\n\n".join([
            f"### Essay (Score: {ex['domain1_score']})\n{ex['essay_text'][:self.config['llm']['max_exemplar_length']]}..."
            for ex in exs
        ])

    def evaluate(self, val_set, vector_store):
        true_s, pred_s = [], []
        print(f"  Scoring {len(val_set)} validation essays...")
        for item in val_set:
            true_s.append(item['domain1_score'])
            # 传入 vector_store
            pred = self.predict_score(item['essay_text'], vector_store)
            pred_s.append(pred)
        
        qwk = self._calc_qwk(true_s, pred_s)
        self.fitness = qwk
        return qwk, true_s, pred_s
        
    def _calc_qwk(self, t, p):
        # 简化版 QWK 计算
        from sklearn.metrics import cohen_kappa_score
        return cohen_kappa_score(t, p, weights='quadratic')

    # ... clone, get_feedback, evolve_instruction 方法同之前 ...
    def clone(self):
        return PromptIndividual(self.instruction_text, self.static_exemplars.copy(), self.fitness, self.config)
    
    def get_feedback(self, val_set, preds):
        # ... (保持原样，省略) ...
        prompt = f"Analyze errors based on Rubric:\n{self.instruction_text}\n..."
        return call_llm(prompt, call_type="feedback")

    def evolve_instruction(self, feedback):
        # ... (保持原样，省略) ...
        prompt = f"Rewrite rubric based on feedback:\n{feedback}\nOld Rubric:\n{self.instruction_text}"
        return call_llm(prompt, call_type="rewrite")


# ============================================================================
# 4. 进化优化器 (EvolutionOptimizer)
# ============================================================================

class EvolutionOptimizer:
    def __init__(self, train_data, val_data, config):
        self.config = config
        self.train_data = train_data
        self.val_data = val_data
        
        # 初始化向量库
        self.vector_store = SimpleVectorStore(
            model_name=config['rag']['model_name'],
            cache_dir="cache"
        )
        # 将训练集作为 RAG 的知识库
        print("[Optimizer] Building Vector Store from Training Data...")
        self.vector_store.add_documents(self.train_data)
        
        self.population = []
        self.history = []

    def initialize_population(self, base_instruction):
        n_static = self.config['evolution']['n_static_exemplars']
        print(f"[Init] Creating population. Static exemplars per ind: {n_static}")
        
        for _ in range(self.config['evolution']['population_size']):
            # 随机采样静态范文
            static_ex = random.sample(self.train_data, n_static)
            ind = PromptIndividual(base_instruction, static_ex, config=self.config)
            self.population.append(ind)

    def evolve_one_generation(self, gen):
        print(f"\n=== Generation {gen} ===")
        
        # 1. 评估
        scores = []
        for i, ind in enumerate(self.population):
            qwk, _, _ = ind.evaluate(self.val_data, self.vector_store)
            scores.append(qwk)
            print(f"  Ind {i}: QWK={qwk:.3f}")
        
        best_idx = np.argmax(scores)
        best_ind = self.population[best_idx]
        print(f"  Best QWK: {best_ind.fitness:.4f}")
        
        self.history.append({"gen": gen, "best_qwk": best_ind.fitness})
        
        # 2. 进化 Rubric (Text Evolution)
        # 仅对精英进行文本进化
        n_elite = self.config['evolution']['n_elite_evolve']
        elites = [self.population[i] for i in np.argsort(scores)[-n_elite:]]
        
        print("  Evolving Rubrics for Elites...")
        for elite in elites:
            # 获取反馈 (需重新跑一遍拿 pred scores)
            _, _, preds = elite.evaluate(self.val_data[:5], self.vector_store) # 少量样本生成反馈
            fb = elite.get_feedback(self.val_data[:5], preds)
            new_instr = elite.evolve_instruction(fb)
            elite.instruction_text = new_instr # 原地更新
            
        # 3. 进化 Exemplars (GA)
        # 简单的精英保留 + 随机变异
        new_pop = [ind.clone() for ind in elites] # 保留精英
        
        while len(new_pop) < len(self.population):
            parent = random.choice(elites)
            child = parent.clone()
            
            # 变异静态范文
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
# 5. 主程序
# ============================================================================

def main():
    # 简化的数据加载 (实际需用 pd.read_csv)
    # 模拟数据
    print("[Main] Loading Data...")
    raw_data = pd.read_csv(CONFIG['data']['asap_path'], sep='\t', encoding='latin-1')
    raw_data = raw_data[raw_data['essay_set'] == CONFIG['data']['essay_set']]
    
    all_data = []
    for _, r in raw_data.iterrows():
        all_data.append({
            "essay_id": r['essay_id'],
            "essay_text": r['essay'],
            "domain1_score": r['domain1_score']
        })
    
    random.shuffle(all_data)
    split = int(len(all_data) * CONFIG['data']['train_ratio'])
    train = all_data[:split]
    val = all_data[split:][:CONFIG['data']['val_size']] # 截断验证集以加速
    
    LOGGER = ExperimentLogger()
    LOGGER.log_config(CONFIG)
    
    # 1. 指令诱导
    print("[Main] Inducing Initial Rubric...")
    # (此处应调用 InstructionInductor, 为简化直接模拟)
    base_rubric = "Score based on coherence, grammar, and topic relevance." 
    
    # 2. 进化
    optimizer = EvolutionOptimizer(train, val, CONFIG)
    optimizer.initialize_population(base_rubric)
    best = optimizer.run()
    
    print(f"\nFinal Best QWK: {best.fitness}")
    print(f"Rubric:\n{best.instruction_text}")

if __name__ == "__main__":
    main()