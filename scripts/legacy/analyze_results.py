import os
import re
import yaml
import numpy as np
from pathlib import Path

BASE_DIR = "/Users/yuqinshu174/project/wise_aes_profiles/1/wise-aes/result_final/icl_on_rag_off_instruction_on_exemplar_on"

def parse_md_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Extract Test QWKs
    # Pattern: | 0 | `exp_...` | 0.6392 | **0.5286** |
    qwks = []
    exp_dir_0 = None
    
    lines = content.split('\n')
    for line in lines:
        if "|" in line and "exp_" in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                # parts[2] is Exp Dir, parts[4] is Test QWK (often bolded)
                exp_path_raw = parts[2].replace('`', '').strip()
                qwk_str = parts[4].replace('*', '').strip()
                try:
                    qwk = float(qwk_str)
                    qwks.append(qwk)
                    
                    if not exp_dir_0:
                        exp_dir_0 = exp_path_raw
                except:
                    pass
    
    return exp_dir_0, qwks

def get_prompt_id(exp_dir_name):
    # Try to find config.yaml in the exp dir
    # exp_dir is relative to BASE_DIR? The md file says `exp_...`, usually they are in the same folder as md
    config_path = Path(BASE_DIR) / exp_dir_name / "config.yaml"
    if not config_path.exists():
        # Maybe deeper? Try recursive search or just assume parallel structure
        # User said "40 subfolders" are in result_final/discard.../
        pass
    
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            conf = yaml.safe_load(f)
            return conf.get('data', {}).get('essay_set', 'Unknown')
    return 'Unknown'

def main():
    md_files = sorted([f for f in os.listdir(BASE_DIR) if f.endswith(".md")])
    
    results = {} # prompt_id -> [qwks]
    
    print(f"{'Prompt':<10} | {'Mean QWK':<10} | {'Std Dev':<10} | {'Raw QWKs'}")
    print("-" * 60)
    
    prompt_means = []
    rows = []

    for md_file in md_files:
        path = os.path.join(BASE_DIR, md_file)
        exp_dir_0, qwks = parse_md_file(path)
        
        if len(qwks) != 5:
            print(f"[Warn] File {md_file} has {len(qwks)} folds (expected 5). Skipping.")
            continue
            
        prompt_id = get_prompt_id(exp_dir_0)
        mean = np.mean(qwks)
        std = np.std(qwks)
        
        # Store for sorting later
        rows.append({
            "id": prompt_id,
            "mean": mean,
            "std": std,
            "qwks": qwks
        })

    # Sort by prompt ID (try to convert to int if possible for numeric sort)
    def sort_key(row):
        try:
            return int(row['id'])
        except:
            return row['id']
            
    rows.sort(key=sort_key)
    
    for row in rows:
        prompt_means.append(row['mean'])
        print(f"{row['id']:<10} | {row['mean']:.4f}     | {row['std']:.4f}     | {row['qwks']}")

    print("-" * 60)
    
    # Calculate Macro Average
    if prompt_means:
        macro_mean = np.mean(prompt_means)
        print(f"\nAverage of Means (Macro-QWK): {macro_mean:.4f}")
    else:
        print("No valid data found.")

if __name__ == "__main__":
    main()
