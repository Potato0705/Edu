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
from concurrent.futures import ThreadPoolExecutor, as_completed

import wise_aes

# Constants
BASE_LOGS_DIR = Path("result_final/icl_on_rag_off_instruction_on_exemplar_on")
CSV_OUTPUT = "rag_substitution_results.csv"

MODEL_INIT_LOCK = threading.Lock()

# Configuration for RAG substitution
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

class AnalysisExperimentManager:
    def __init__(self, config):
        self.config = config
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # No tokenizer necessary for analysis runs
             
    def count_tokens(self, text):
        return len(text) // 4

    def track_usage(self, p, c):
        self.total_prompt_tokens += p
        self.total_completion_tokens += c
        self.total_tokens += (p + c)

    def log_llm_trace(self, record):
        pass 

def load_data_split(config, fold):
    """
    Returns (train_set, test_set) for the given fold.
    """
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
    train_idx, test_idx = folds[fold] # Note: folds[fold] gives (train_idx, test_idx)
    
    train_set = [all_data[i] for i in train_idx]
    test_set = [all_data[i] for i in test_idx]
    
    return train_set, test_set

def find_champion_individual(exp_dir):
    """
    Finds the individual with the highest validation fitness across all generations.
    """
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
                # Extract generation number from filename
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
    # Unpack task
    prompt_id = task_data['prompt']
    fold = task_data['fold']
    config = copy.deepcopy(task_data['config'])
    best_ind = task_data['best_ind']
    
    # 1. Apply RAG Configuration Override
    if 'rag' not in config: config['rag'] = {}
    config['rag'].update(RAG_CONFIG_OVERRIDE)
    
    # Force single worker for inner loop
    if 'evolution' not in config: config['evolution'] = {}
    config['evolution']['max_workers'] = 1
    
    # 2. Build Vector Store from Train Set
    train_set = task_data['train_set']
    test_set = task_data['test_set']
    
    # Initialize Vector Store if RAG is enabled
    # Wise-AES SimpleVectorStore needs config and data
    # Found issue: SimpleVectorStore init takes (model_name, cache_dir), NOT (config, data).
    # Correct usage:
    # [FIX] Use global lock to prevent concurrent SentenceTransformer initialization crash
    with MODEL_INIT_LOCK:
        vector_store = wise_aes.SimpleVectorStore(model_name=config['rag']['model_name'])
        vector_store.add_documents(train_set)
        # Force load model sequentially to avoid "meta tensor" errors during concurrent inference init
        vector_store._load_model()
    
    # 3. Initialize Independent (Instruction Only, RAG On)
    # We deliberately IGNORE static_exemplars from best_ind as per user request
    ind = wise_aes.PromptIndividual(
        instruction_text=best_ind['full_instruction'],
        static_exemplars=[], # Discard exemplars
        fitness=0.0,
        config=config
    )
    
    # 4. Evaluate
    # Set global manager for this thread's context? 
    # wise_aes.call_llm uses global EXP_MANAGER.
    # We initialized it globally in main. Since config['model'] is likely same, it's fine.
    # BUT if config changes per prompt (e.g. model params), we might have issues.
    # Assuming standard config across all experiments for Model.
    
    # Enable RAG (Vector Store passed)
    # disable_rerank=True? User said "use_rerank_test: false".
    # wise_aes.evaluate signature: evaluate(self, val_set, vector_store, enable_rerank=False)
    # We should pass enable_rerank according to config 'use_rerank_test'
    enable_rerank = RAG_CONFIG_OVERRIDE['use_rerank_test']
    
    qwk = ind.evaluate(test_set, vector_store=vector_store, enable_rerank=enable_rerank)
    
    return {
        "prompt": prompt_id,
        "fold": fold,
        "source_gen": best_ind.get('source_gen'),
        "val_fitness": best_ind.get('fitness'),
        "qwk": qwk
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10, help="Concurrency")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    if not BASE_LOGS_DIR.exists():
        print(f"Directory {BASE_LOGS_DIR} not found.")
        return

    # 1. Scan Experiments
    exp_dirs = [d for d in BASE_LOGS_DIR.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    experiment_map = {} 
    
    print(f"Scanning {len(exp_dirs)} directories...")
    for d in exp_dirs:
        config_path = d / "config.yaml"
        if not config_path.exists(): continue
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        pid = config['data']['essay_set']
        fold = int(str(d.name).split('fold')[-1])
        if pid not in experiment_map: experiment_map[pid] = {}
        experiment_map[pid][fold] = {"path": d, "config": config}

    prompts = sorted(experiment_map.keys())
    if args.limit: prompts = prompts[:args.limit]
    
    # Initialize Global Manager
    if prompts and 0 in experiment_map[prompts[0]]:
        first_config = experiment_map[prompts[0]][0]['config']
        wise_aes.EXP_MANAGER = AnalysisExperimentManager(first_config)
    
    # 2. Check Resume
    completed_keys = set()
    if os.path.exists(CSV_OUTPUT):
        try:
            df = pd.read_csv(CSV_OUTPUT)
            for _, row in df.iterrows():
                completed_keys.add((row['prompt'], row['fold']))
            print(f"Resuming: Found {len(completed_keys)} completed records.")
        except: pass
        
    # 3. Prepare Tasks
    tasks = []
    print("Preparing tasks (Scanning for champions and loading data)...")
    
    for pid in prompts:
        folds_data = experiment_map[pid]
        for fold in range(5):
            if fold not in folds_data: continue
            if (pid, fold) in completed_keys: continue
            
            info = folds_data[fold]
            
            # Find Champion
            best_ind = find_champion_individual(info['path'])
            if not best_ind:
                print(f"[Warn] No champion found for P{pid} F{fold}")
                continue
                
            # Load Data (Train + Test)
            try:
                train_set, test_set = load_data_split(info['config'], fold)
                
                tasks.append({
                    "prompt": pid,
                    "fold": fold,
                    "config": info['config'],
                    "best_ind": best_ind,
                    "train_set": train_set,
                    "test_set": test_set
                })
            except Exception as e:
                print(f"[Error] Data load failed P{pid} F{fold}: {e}")

    print(f"Total Tasks: {len(tasks)}")
    
    # 4. Execute
    if not tasks:
        print("Done.")
        return
        
    file_exists = os.path.exists(CSV_OUTPUT)
    csv_file = open(CSV_OUTPUT, 'a', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=["prompt", "fold", "source_gen", "val_fitness", "qwk"])
    if not file_exists:
        csv_writer.writeheader()
        csv_file.flush()
        
    job_counter = 0
    total = len(tasks)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_task, t) for t in tasks]
        for future in as_completed(futures):
            job_counter += 1
            try:
                res = future.result()
                csv_writer.writerow(res)
                csv_file.flush()
                print(f"[{job_counter}/{total}] P{res['prompt']} F{res['fold']} -> QWK: {res['qwk']:.4f}")
            except Exception as e:
                print(f"[{job_counter}/{total}] Failed: {e}")
                
    csv_file.close()

if __name__ == "__main__":
    main()
