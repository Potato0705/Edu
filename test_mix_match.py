
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
DIR_INSTRUCTION = Path("result_final/icl_on_rag_off_instruction_on_exemplar_off")
DIR_EXEMPLAR = Path("result_final/icl_on_rag_off_instruction_off_exemplar_on")
CSV_OUTPUT = "mix_match_results.csv"

# Shared Lock
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

def get_best_individual(exp_dir, gen=25):
    """
    Reads directly from gen_025.json (or specified gen).
    """
    gen_file = exp_dir / "generations" / f"gen_{gen:03d}.json"
    if not gen_file.exists():
        # Fallback to finding max if exact gen missing? 
        # Requirement says "25th generation best individual".
        # If gen 25 missing, we should probably error or try latest.
        # Let's try to find highest available if 25 missing?
        # Or strict. Let's be strict first, or check if file exists.
        return None
        
    try:
        with open(gen_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Find best in this generation
        best_ind = max(data['population'], key=lambda x: x.get('fitness', -float('inf')))
        best_ind['source_gen'] = gen
        return best_ind
    except Exception as e:
        print(f"Error reading {gen_file}: {e}")
        return None

def find_experiment_dir(base_dir, prompt_id, fold_id):
    """
    Finds the directory matching prompt and fold in the given base path.
    """
    if not base_dir.exists(): return None
    
    # We scan all dirs once? Or just assume structure?
    # Structure is usually exp_{timestamp}_fold{fold}
    # But we need to check config inside to match prompt_id.
    # This is slow if done repeatedly.
    # Optimization: Helper scans all and returns map.
    pass # Implemented inline in main for efficiency

def evaluate_mix_match(task):
    prompt_id = task['prompt']
    fold = task['fold']
    
    # Use Config from Instruction source as base (arbitrary choice, but likely compatible)
    config = copy.deepcopy(task['config_instr'])
    
    # Force single worker
    if 'evolution' not in config: config['evolution'] = {}
    config['evolution']['max_workers'] = 1
    
    # Construct Mixed Individual
    instr_source = task['best_instr']
    exemplar_source = task['best_exemplar']
    
    # 1. Instruction
    instruction_text = instr_source['full_instruction']
    
    # 2. Exemplars
    # We need to make sure we have the full exemplar objects (text + score + id)
    # The JSON usually contains full objects in 'static_exemplars' OR just IDs in 'static_exemplar_ids'
    # depending on how it was saved.
    # Check what we have.
    
    mixed_exemplars = []
    
    if 'static_exemplars' in exemplar_source and len(exemplar_source['static_exemplars']) > 0:
        # Check if it has text.
        first_ex = exemplar_source['static_exemplars'][0]
        if 'essay_text' in first_ex:
            mixed_exemplars = exemplar_source['static_exemplars']
        else:
            # Need hydration
            pass # Implement hydration if needed
    elif 'static_exemplar_ids' in exemplar_source:
        # Need hydration from train set
        ids = set(exemplar_source['static_exemplar_ids'])
        train_set = task['train_set']
        mixed_exemplars = [ex for ex in train_set if ex['essay_id'] in ids]
    
    # Safety Check: If hydration failed or empty
    if not mixed_exemplars:
        # Fallback: Try to use IDs from source dict if present and hydrate
        possible_ids = []
        if 'static_exemplars' in exemplar_source:
             possible_ids = [x.get('essay_id') for x in exemplar_source['static_exemplars']]
        
        if possible_ids:
            ids = set(possible_ids)
            train_set = task['train_set']
            mixed_exemplars = [ex for ex in train_set if ex['essay_id'] in ids]
            
    # Create Individual
    # RAG Off -> config['rag']['enabled'] should be False (default in these directories)
    # Confirm config
    if 'rag' in config: config['rag']['enabled'] = False
            
    ind = wise_aes.PromptIndividual(
        instruction_text=instruction_text,
        static_exemplars=mixed_exemplars,
        fitness=0.0,
        config=config
    )
    
    test_set = task['test_set']
    
    # Evaluate
    qwk = ind.evaluate(test_set, vector_store=None, enable_rerank=False)
    
    return {
        "prompt": prompt_id,
        "fold": fold,
        "instr_gen": instr_source['source_gen'],
        "ex_gen": exemplar_source['source_gen'],
        "qwk": qwk
    }

def scan_experiments(base_dir):
    """
    Returns map: prompt -> fold -> {path, config}
    """
    experiment_map = {}
    if not base_dir.exists(): return {}
    
    dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    for d in dirs:
        config_path = d / "config.yaml"
        if not config_path.exists(): continue
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            pid = config['data']['essay_set']
            fold = int(str(d.name).split('fold')[-1])
            
            if pid not in experiment_map: experiment_map[pid] = {}
            experiment_map[pid][fold] = {"path": d, "config": config}
        except: continue
    return experiment_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    
    # 1. Scan Both Directories
    print("Scanning Instruction Source Directories...")
    map_instr = scan_experiments(DIR_INSTRUCTION)
    print("Scanning Exemplar Source Directories...")
    map_ex = scan_experiments(DIR_EXEMPLAR)
    
    prompts = sorted(list(set(map_instr.keys()) & set(map_ex.keys())))
    if args.limit: prompts = prompts[:args.limit]
    
    # Global Init
    if prompts:
        first_pid = prompts[0]
        if 0 in map_instr[first_pid]:
             wise_aes.EXP_MANAGER = AnalysisExperimentManager(map_instr[first_pid][0]['config'])

    # 2. Prepare Tasks
    tasks = []
    print("Preparing tasks...")
    
    # Check Resume
    completed = set()
    if os.path.exists(CSV_OUTPUT):
        try:
            df = pd.read_csv(CSV_OUTPUT)
            for _, row in df.iterrows():
                completed.add((row['prompt'], row['fold']))
        except: pass
        
    for pid in prompts:
        for fold in range(5):
            if (pid, fold) in completed: continue
            
            # Must exist in both
            if fold not in map_instr[pid] or fold not in map_ex[pid]:
                print(f"[Warn] Missing data for P{pid} F{fold} in one of the sources.")
                continue
                
            info_instr = map_instr[pid][fold]
            info_ex = map_ex[pid][fold]
            
            # Get Individuals (Gen 25)
            best_instr = get_best_individual(info_instr['path'], gen=25)
            best_ex = get_best_individual(info_ex['path'], gen=25)
            
            if not best_instr or not best_ex:
                print(f"[Warn] Gen 25 data missing for P{pid} F{fold}.")
                continue
            
            # Load Data (using config from instr)
            try:
                train_set, test_set = load_data_split(info_instr['config'], fold)
                
                tasks.append({
                    "prompt": pid,
                    "fold": fold,
                    "config_instr": info_instr['config'],
                    "best_instr": best_instr,
                    "best_exemplar": best_ex,
                    "train_set": train_set,
                    "test_set": test_set
                })
            except Exception as e:
                print(f"[Error] Data load failed: {e}")

    print(f"Total Tasks: {len(tasks)}")
    if not tasks: return

    # 3. Execute with ThreadPool
    file_exists = os.path.exists(CSV_OUTPUT)
    csv_file = open(CSV_OUTPUT, 'a', newline='', encoding='utf-8')
    csv_writer = csv.DictWriter(csv_file, fieldnames=["prompt", "fold", "instr_gen", "ex_gen", "qwk"])
    if not file_exists:
        csv_writer.writeheader()
        csv_file.flush()
        
    job_count = 0 
    total = len(tasks)
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(evaluate_mix_match, t) for t in tasks]
        for future in as_completed(futures):
            job_count += 1
            try:
                res = future.result()
                csv_writer.writerow(res)
                csv_file.flush()
                print(f"[{job_count}/{total}] P{res['prompt']} F{res['fold']} QWK: {res['qwk']:.4f}")
            except Exception as e:
                print(f"[{job_count}/{total}] Error: {e}")

    csv_file.close()
    print("Execution complete.")

if __name__ == "__main__":
    main()
