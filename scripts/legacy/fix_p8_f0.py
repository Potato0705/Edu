
import os
import sys
import json
import csv
import yaml
import copy
import argparse
import threading
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
import wise_aes

# Constants
TARGET_EXP_DIR = Path("result_final/icl_on_rag_off_instruction_on_exemplar_on/exp_20251229_102912_fold0")
CSV_OUTPUT = "rag_substitution_results.csv"

# Configuration for RAG substitution (Copied from test_rag_substitution.py)
RAG_CONFIG_OVERRIDE = {
    "enabled": True,
    "model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "n_retrieved": 6,
    "n_selected": 6,
    "use_rerank_train": False,
    "use_rerank_test": False,
    "rubric_driven_retrieval": False,
    "use_raw_query": True
}

MODEL_INIT_LOCK = threading.Lock()

class AnalysisExperimentManager:
    def __init__(self, config):
        self.config = config
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
             
    def count_tokens(self, text):
        return len(text) // 4

    def track_usage(self, p, c):
        self.total_prompt_tokens += p
        self.total_completion_tokens += c
        self.total_tokens += (p + c)

    def log_llm_trace(self, record):
        pass 

def load_data_split(config, fold):
    data_path = config['data']['asap_path']
    essay_set = config['data']['essay_set']
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    df = df[df['essay_set'] == essay_set]
    all_data = []
    for _, row in df.iterrows():
        all_data.append({
            "essay_id": row['essay_id'],
            "essay_text": row['essay'],
            "domain1_score": int(row['domain1_score'])
        })
        
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(all_data))
    train_idx, test_idx = folds[fold]
    
    train_set = [all_data[i] for i in train_idx]
    test_set = [all_data[i] for i in test_idx]
    
    return train_set, test_set

def find_champion_individual(exp_dir):
    generations_dir = exp_dir / "generations"
    if not generations_dir.exists():
        return None
        
    best_ind = None
    max_fitness = -float('inf')
    best_gen = -1
    
    gen_files = sorted(generations_dir.glob("gen_*.json"))
    for gf in gen_files:
        try:
            with open(gf, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            gen_max_ind = max(data['population'], key=lambda x: x.get('fitness', -float('inf')))
            fitness = gen_max_ind.get('fitness', -float('inf'))
            
            if fitness > max_fitness:
                max_fitness = fitness
                best_ind = gen_max_ind
                try:
                    best_gen = int(gf.stem.split('_')[1])
                except:
                    pass
        except:
            continue
            
    if best_ind:
        best_ind['source_gen'] = best_gen
        
    return best_ind

def evaluate_task(task_data):
    prompt_id = task_data['prompt']
    fold = task_data['fold']
    config = copy.deepcopy(task_data['config'])
    best_ind = task_data['best_ind']
    
    # 1. Apply RAG Configuration Override
    if 'rag' not in config: config['rag'] = {}
    config['rag'].update(RAG_CONFIG_OVERRIDE)
    
    if 'evolution' not in config: config['evolution'] = {}
    config['evolution']['max_workers'] = 200
    
    train_set = task_data['train_set']
    test_set = task_data['test_set']
    
    # 2. Build Vector Store
    print("Building Vector Store...")
    with MODEL_INIT_LOCK:
        vector_store = wise_aes.SimpleVectorStore(model_name=config['rag']['model_name'])
        vector_store.add_documents(train_set)
        vector_store._load_model()
    
    # 3. Initialize Independent
    ind = wise_aes.PromptIndividual(
        instruction_text=best_ind['full_instruction'],
        static_exemplars=[], # Discard exemplars
        fitness=0.0,
        config=config
    )
    
    # 4. Evaluate
    enable_rerank = RAG_CONFIG_OVERRIDE['use_rerank_test']
    print(f"Evaluating P{prompt_id} F{fold} with {len(test_set)} items...")
    qwk = ind.evaluate(test_set, vector_store=vector_store, enable_rerank=enable_rerank)
    
    return {
        "prompt": prompt_id,
        "fold": fold,
        "source_gen": best_ind.get('source_gen'),
        "val_fitness": best_ind.get('fitness'),
        "qwk": qwk
    }

def main():
    if not TARGET_EXP_DIR.exists():
        print(f"Error: {TARGET_EXP_DIR} does not exist.")
        return

    config_path = TARGET_EXP_DIR / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    pid = config['data']['essay_set']
    fold = int(str(TARGET_EXP_DIR.name).split('fold')[-1])
    
    print(f"Target: Prompt {pid}, Fold {fold}")
    
    # Initialize Manager
    wise_aes.EXP_MANAGER = AnalysisExperimentManager(config)
    
    # Find Champion
    print("Finding Champion Individual...")
    best_ind = find_champion_individual(TARGET_EXP_DIR)
    if not best_ind:
        print("Error: No champion found.")
        return
    print(f"Champion from Gen {best_ind.get('source_gen')} (Fitness: {best_ind.get('fitness')})")

    # Load Data
    print("Loading Data...")
    train_set, test_set = load_data_split(config, fold)
    
    task = {
        "prompt": pid,
        "fold": fold,
        "config": config,
        "best_ind": best_ind,
        "train_set": train_set,
        "test_set": test_set
    }
    
    # Run
    res = evaluate_task(task)
    print(f"Result: QWK = {res['qwk']}")
    
    # Append to CSV
    file_exists = os.path.exists(CSV_OUTPUT)
    with open(CSV_OUTPUT, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["prompt", "fold", "source_gen", "val_fitness", "qwk"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(res)
        
    print(f"Appended to {CSV_OUTPUT}")

if __name__ == "__main__":
    main()
