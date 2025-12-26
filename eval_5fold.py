import json
import argparse
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
import yaml

# 复用 wise_aes.py 中的核心类
# 必须确保 wise_aes.py 在 PYTHONPATH 中
from wise_aes import PromptIndividual, SimpleVectorStore

def evaluate_fold_standard_cv(exp_dir, generation, fold_idx, output_csv_writer=None):
    exp_path = Path(exp_dir)
    gen_file = exp_path / "generations" / f"gen_{generation:03d}.json"
    config_path = exp_path / "config.yaml"
    
    if not gen_file.exists():
        print(f"[Error] Generation file not found: {gen_file}")
        return None
    
    if not config_path.exists():
        # Fallback
        config_path = "configs/official.yaml"
        
    print(f"\n>>> Processing Fold {fold_idx} | Exp: {exp_path.name} | Gen: {generation}")
    
    # 1. Load Config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Mock EXP_MANAGER
    import wise_aes
    class MockExpManager:
        def __init__(self, cfg): self.config = cfg
        def log_llm_trace(self, x): pass
    wise_aes.EXP_MANAGER = MockExpManager(config)
    
    # 2. Reconstruct Test Set for this Fold
    data_path = config['data']['asap_path']
    essay_set = config['data']['essay_set']
    
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    df = df[df['essay_set'] == essay_set]
    all_data = []
    for _, row in df.iterrows():
        all_data.append({
            "essay_id": row['essay_id'],
            "essay_text": row['essay'],
            "domain1_score": int(row['domain1_score'])
        })
        
    # Strict Fold Splitting
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(all_data))
    train_val_idx, test_idx = folds[fold_idx]
    
    test_set = [all_data[i] for i in test_idx]
    train_set = [all_data[i] for i in train_val_idx] # Needed for Vector Store exemplars
    
    # 3. Build Vector Store (To recover exemplars content)
    # Note: We don't really use vector store for retrieval here unless dynamic rag is on, 
    # but we need it to hydrate static exemplars.
    print(f"  [VectorStore] Loading Train Set ({len(train_set)} docs) to hydrate exemplars...")
    # 实际上为了加速，我们只需要一个 ID -> Doc 的映射字典，不需要真的 Build 整个 Index
    id_to_doc = {d['essay_id']: d for d in train_set}
    
    # 但如果开启了 Dynamic RAG，我们还是得 Build
    vector_store = SimpleVectorStore(model_name=config['rag']['model_name'])
    vector_store.documents = train_set
    if config['rag']['enabled']:
        # Lazy build
        pass 
        # Actually predict_score WILL call vector_store.search which calls encode.
        # But wait, standard CV implies we use the same retrieval logic as inference.
        vector_store.add_documents(train_set)

    # 4. Load Population & Select Champion (Standard CV Logic)
    with open(gen_file, 'r', encoding='utf-8') as f:
        snap = json.load(f)
        pop_data = snap['population']
        
    # [KEY STEP] Select the ONE best individual based on Validation Fitness ONLY.
    # fitness in json IS validation score.
    champion_data = max(pop_data, key=lambda x: x.get('fitness', -1.0))
    champion_idx = pop_data.index(champion_data)
    
    print(f"  [Selection] Selected Champion: Ind {champion_idx} (Val QWK: {champion_data.get('fitness'):.4f})")
    
    # 5. Evaluate Champion on Test Set
    rubric = champion_data['full_instruction']
    ex_ids = champion_data['static_exemplar_ids']
    
    # Recover Exemplars from Train Set
    exemplars = []
    for eid in ex_ids:
        if eid in id_to_doc:
            exemplars.append(id_to_doc[eid])
        else:
            exemplars.append(train_set[0]) # Fallback
            
    ind = PromptIndividual(rubric, exemplars, config=config)
    
    use_rerank = config['rag'].get('use_rerank_test', True)
    test_qwk = ind.evaluate(test_set, vector_store, enable_rerank=use_rerank)
    
    print(f"  >> Fold {fold_idx} Test QWK: {test_qwk:.4f}")
    
    record = {
        "fold": fold_idx,
        "generation": generation,
        "champion_id": champion_idx,
        "val_fitness": champion_data.get('fitness'),
        "test_qwk": test_qwk,
        "exp_dir": exp_path.name
    }
    
    if output_csv_writer is not None:
        output_csv_writer.append(record)
        
    return test_qwk

def main():
    parser = argparse.ArgumentParser(description="Calc Standard 5-Fold CV: Select Val-Best, Eval on Test.")
    parser.add_argument("--gen", type=int, required=True, help="Generation to evaluate (e.g., 25)")
    parser.add_argument("--dirs", nargs='+', required=True, help="List of 5 experiment directories (e.g. logs/exp_fold0 logs/exp_fold1 ...)")
    parser.add_argument("--output", type=str, default="5fold_standard_results.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    if len(args.dirs) != 5:
        print(f"[Warning] You provided {len(args.dirs)} directories. A standard 5-fold CV requires exactly 5.")
    
    all_records = []
    fold_qwks = []
    
    for i, exp_dir in enumerate(args.dirs):
        # Auto-detect fold index from string if possible, else use list index
        fold_idx = i 
        try:
            if "fold0" in exp_dir: fold_idx = 0
            elif "fold1" in exp_dir: fold_idx = 1
            elif "fold2" in exp_dir: fold_idx = 2
            elif "fold3" in exp_dir: fold_idx = 3
            elif "fold4" in exp_dir: fold_idx = 4
        except: pass
        
        qwk = evaluate_fold_standard_cv(exp_dir, args.gen, fold_idx, all_records)
        if qwk is not None:
            fold_qwks.append(qwk)
            
    # Final Stats
    if fold_qwks:
        avg_qwk = np.mean(fold_qwks)
        print("\n" + "="*40)
        print(f"5-Fold Standard CV Summary (Gen {args.gen})")
        print("="*40)
        for idx, qwk in enumerate(fold_qwks):
            print(f"Fold {idx}: {qwk:.4f}")
        print("-" * 20)
        print(f"Standard CV Test QWK: {avg_qwk:.4f}")
        print("="*40)
        
    # Save CSV
    pd.DataFrame(all_records).to_csv(args.output, index=False)
    print(f"Detailed logs saved to {args.output}")

if __name__ == "__main__":
    main()
