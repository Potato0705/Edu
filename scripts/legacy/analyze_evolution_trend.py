import os
import sys
import json
import csv
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import KFold
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor

# Import Wise-AES components
# We assume wise_aes.py is in the same directory
import wise_aes

# Constants
BASE_LOGS_DIR = Path("result_final/icl_off_rag_off_instruction_on_exemplar_off")
CSV_OUTPUT = "evolution_trend_data.csv"
PLOT_OUTPUT = "evolution_trend.png"

def load_asap_data(config, fold):
    """
    Replicates wise_aes.main data loading logic to get the exact Test Set.
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
        
    # Replicate KFold logic
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(all_data))
    train_val_idx, test_idx = folds[fold]
    
    test_set = [all_data[i] for i in test_idx]
    return test_set

def get_best_individual_from_gen(gen_file):
    with open(gen_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    pop = data['population']
    # If fitness is available, pick max. If not (early gens might vary), rely on logic.
    # Usually 'fitness' is present.
    best_ind = max(pop, key=lambda x: x.get('fitness', -float('inf')))
    return best_ind

class AnalysisExperimentManager:
    def __init__(self, config):
        self.config = config
        self.total_tokens = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        # No tokenizer necessary for analysis runs
             
    def count_tokens(self, text):
        # fast approximation
        return len(text) // 4

    def track_usage(self, p, c):
        self.total_prompt_tokens += p
        self.total_completion_tokens += c
        self.total_tokens += (p + c)

    def log_llm_trace(self, record):
        pass # Muted

def evaluate_generation(prompt_id, fold, gen, exp_dir, test_set, config, vector_store=None):
    gen_file = exp_dir / "generations" / f"gen_{gen:03d}.json"
    if not gen_file.exists():
        print(f"  [Warn] {gen_file} missing. Skipping.")
        return None
        
    best_ind_data = get_best_individual_from_gen(gen_file)
    
    # Reconstruct PromptIndividual
    # We need to set the global EXP_MANAGER to use this config so wise_aes.call_llm works correctly
    wise_aes.EXP_MANAGER = AnalysisExperimentManager(config)
    
    # Reconstruct Exemplars object list from IDs if possible, or just use what's in JSON if it has full text
    # But gen.json usually has `static_exemplar_ids`.
    # AND `get_best_individual_from_gen` returns a dict.
    # We need to hydrate exemplars if we want to run it.
    # HOWEVER, in this specific experiment (icl_off), static_exemplars should be empty or ignored.
    # Let's check `static_exemplar_ids`.
    
    static_ids = best_ind_data.get('static_exemplar_ids', [])
    exemplars = [] 
    # If we really needed them, we'd need to fetch from `train_set`. 
    # For now, if icl_off, len is 0.
    
    ind = wise_aes.PromptIndividual(
        instruction_text=best_ind_data['full_instruction'],
        static_exemplars=exemplars, # Empty for icl_off
        fitness=0.0,
        config=config
    )
    
    # Run Evaluation
    # Note: evaluate() expects a validation set format. test_set matches that.
    # We disable rerank as per user config (rag_off).
    qwk = ind.evaluate(test_set, vector_store=vector_store, enable_rerank=False)
    
    return qwk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of prompts/gens for testing")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent evaluation threads")
    args = parser.parse_args()
    
    # 1. Map Prompts to Exp Dirs
    # We need to find all fold dirs for each prompt.
    # Structure: result_final/.../exp_..._foldX
    
    if not BASE_LOGS_DIR.exists():
        print(f"Directory {BASE_LOGS_DIR} not found.")
        return

    exp_dirs = [d for d in BASE_LOGS_DIR.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    
    # Build Map: prompt_id -> fold_id -> exp_dir
    experiment_map = {} 
    
    print(f"Scanning {len(exp_dirs)} directories...")
    
    for d in exp_dirs:
        config_path = d / "config.yaml"
        if not config_path.exists(): continue
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        pid = config['data']['essay_set'] # prompt_id
        fold = int(str(d.name).split('fold')[-1])
        
        if pid not in experiment_map: experiment_map[pid] = {}
        experiment_map[pid][fold] = {
            "path": d,
            "config": config
        }
        
    print(f"Found {len(experiment_map)} prompts.")
    
    # 2. Main Eval Loop
    # Flatten tasks
    tasks = []
    
    # Initialize Global EXP_MANAGER once with a representative config
    # We assume 'model' config is consistent across the batch.
    
    prompts = sorted(experiment_map.keys())
    if args.limit: prompts = prompts[:args.limit]

    # We take the first available config from the first prompt.
    if prompts and 0 in experiment_map[prompts[0]]:
        first_config = experiment_map[prompts[0]][0]['config']
        wise_aes.EXP_MANAGER = AnalysisExperimentManager(first_config)
        print(f"Initialized Global Experiment Manager (Model: {first_config.get('model', {}).get('name')})")
    
    generations = list(range(1, 26)) # 1 to 25
    if args.limit: generations = generations[:args.limit]
    
    # Load existing results for resume
    completed_keys = set()
    if os.path.exists(CSV_OUTPUT):
        try:
            existing_df = pd.read_csv(CSV_OUTPUT)
            for _, row in existing_df.iterrows():
                completed_keys.add((row['prompt'], row['fold'], row['generation']))
            print(f"Resuming from {len(completed_keys)} existing records in {CSV_OUTPUT}.")
        except Exception as e:
            print(f"[Warn] Could not read existing CSV: {e}")

    print(f"Preparing tasks for {len(prompts)} Prompts x {len(generations)} Gens x 5 Folds...")
    
    for pid in prompts:
        folds_data = experiment_map[pid]
        for fold in range(5):
            if fold not in folds_data:
                continue
                
            info = folds_data[fold]
            
            # Check if all gens for this fold are done?? No, check individually.
            # But loading Test Set is expensive. 
            # We should check if we have ANY pending tasks for this fold.
            pending_gens = []
            for gen in generations:
                if (pid, fold, gen) not in completed_keys:
                    pending_gens.append(gen)
            
            if not pending_gens:
                continue

            # Load Test Set (Thread-safe? Reading file is fine)
            try:
                # Optimization: Load once per Fold.
                test_set = load_asap_data(info['config'], fold)
                
                for gen in pending_gens:
                    tasks.append({
                        "prompt": pid,
                        "fold": fold,
                        "gen": gen,
                        "exp_path": info['path'],
                        "config": info['config'],
                        "test_set": test_set
                    })
            except Exception as e:
                print(f"[Error] loading data for P{pid} F{fold}: {e}")

    print(f"Total Tasks to run: {len(tasks)}")
    print(f"Concurrency: {args.workers} workers")
    
    if not tasks:
        print("All tasks completed. Proceeding to plotting.")
    else:
        # Prepare CSV for appending if it doesn't exist, write header
        file_exists = os.path.exists(CSV_OUTPUT)
        csv_file = open(CSV_OUTPUT, 'a', newline='', encoding='utf-8')
        csv_writer = csv.DictWriter(csv_file, fieldnames=["prompt", "fold", "generation", "qwk"])
        if not file_exists:
            csv_writer.writeheader()
            csv_file.flush()
        
        job_counter = 0
        total_jobs = len(tasks)
        
        def process_task(task):
            # Local config override for inner threads
            # Must deep copy to avoid race conditions on shared dict
            t_config = copy.deepcopy(task['config'])
            
            # Reduce inner parallelism to avoid thread explosion
            if 'evolution' not in t_config: t_config['evolution'] = {}
            t_config['evolution']['max_workers'] = 1 
            
            qwk = evaluate_generation(
                task['prompt'], 
                task['fold'], 
                task['gen'], 
                task['exp_path'], 
                task['test_set'], 
                t_config
            )
            return {
                "prompt": task['prompt'],
                "fold": task['fold'],
                "generation": task['gen'],
                "qwk": qwk
            }

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(process_task, t) for t in tasks]
            
            from concurrent.futures import as_completed
            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res['qwk'] is not None:
                        # Append to CSV immediately
                        csv_writer.writerow(res)
                        csv_file.flush()
                    else:
                        pass
                except Exception as e:
                    print(f"  [Job Error] {e}")
                    
                # Simple progress
                job_counter += 1
                if job_counter % 10 == 0:
                    print(f"Progress: {job_counter}/{total_jobs} ({job_counter/total_jobs*100:.1f}%)", end='\r')
        
        csv_file.close()
        print(f"\nCompleted new evaluations.")
                    
    # 3. Reload Full Data for Plotting
    if os.path.exists(CSV_OUTPUT):
        df = pd.read_csv(CSV_OUTPUT)
    else:
        print("No data found.")
        return
    
    print(f"Total records in CSV: {len(df)}")
    
    if df.empty:
        print("No results generated.")
        return

    # 4. Aggregate & Plot
    # Group by [generation, prompt] -> mean(qwk) over folds
    prompt_gen_means = df.groupby(['generation', 'prompt'])['qwk'].mean().reset_index()
    
    # Group by [generation] -> mean(mean_qwk) over prompts (Grand Mean)
    grand_means = prompt_gen_means.groupby('generation')['qwk'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    plt.plot(grand_means['generation'], grand_means['qwk'], marker='o', linestyle='-', linewidth=2, label='Average QWK (All Prompts)')
    
    plt.title('Evolution of Test QWK (Average of 8 Prompts)')
    plt.xlabel('Generation')
    plt.ylabel('Test QWK')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(PLOT_OUTPUT)
    print(f"Plot saved to {PLOT_OUTPUT}")
    
    # Also print the table of Grand Means
    print("\nGeneration | Grand Mean QWK")
    print("-" * 30)
    # Sort by gen just in case
    grand_means = grand_means.sort_values('generation')
    for _, row in grand_means.iterrows():
        print(f"{int(row['generation']):<10} | {row['qwk']:.4f}")

if __name__ == "__main__":
    main()
