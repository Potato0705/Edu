import json
import argparse
import os
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import KFold
import yaml

# 复用 wise_aes.py 中的核心类
# 注意：这需要 wise_aes.py 在同一目录下，或者在 PYTHONPATH 中
from wise_aes import PromptIndividual, EvolutionOptimizer, SimpleVectorStore, ExperimentManager, EXP_MANAGER, main

def load_generation_file(gen_file_path):
    with open(gen_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"[Loader] Loaded Generation {data.get('generation', '?')} from {gen_file_path}")
    return data['population']

def evaluate_population_on_test(pop_data, test_set, config, vector_store):
    print(f"\n{'='*20} Test Set Evaluation {'='*20}")
    print(f"Test Set Size: {len(test_set)}")
    
    # 强制开启 Rerank (通常测试集评估时我们要最好的效果)
    use_rerank = config['rag'].get('use_rerank_test', True)
    print(f"[Config] Inference Rerank: {'ENABLED' if use_rerank else 'DISABLED'}")

    results = []
    
    for i, ind_dict in enumerate(pop_data):
        # 重建 PromptIndividual 对象
        rubric = ind_dict['full_instruction']
        # 注意：这里我们需要从 vector store 的所有文档中找回 exemplars 的完整信息
        # 因为 gen.json 里只存了 ID。
        static_ex_ids = ind_dict['static_exemplar_ids']
        static_exemplars = []
        
        # 从 vector store 的文档中查找 (这通常是训练集)
        all_docs_map = {d['essay_id']: d for d in vector_store.documents}
        
        missing_ids = []
        for eid in static_ex_ids:
            if eid in all_docs_map:
                static_exemplars.append(all_docs_map[eid])
            else:
                missing_ids.append(eid)
        
        if missing_ids:
            print(f"[Warning] Ind {i}: Missing exemplars {missing_ids} in Vector Store (Train Set).")
            # 如果找不到，这个个体可能无法正常工作，暂时跳过或者用空代替
        
        individual = PromptIndividual(rubric, static_exemplars, config=config)
        
        print(f"  Evaluating Ind {i} (Fitness in Val: {ind_dict.get('fitness', 'N/A'):.4f})...")
        test_qwk = individual.evaluate(test_set, vector_store, enable_rerank=use_rerank)
        print(f"  >> Ind {i} Test QWK: {test_qwk:.4f}")
        
        results.append({
            "individual_id": i,
            "val_fitness": ind_dict.get('fitness', -1),
            "test_qwk": test_qwk,
            "rubric_preview": rubric[:100]
        })

    return results

def main_eval():
    parser = argparse.ArgumentParser(description="Evaluate a specific generation JSON on the Test Set")
    parser.add_argument("gen_file", type=str, help="Path to the generation JSON file (e.g., logs/exp_.../generations/gen_005.json)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml. If not provided, attempts to find it in the experiment dir.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to reconstruct Test set")
    
    args = parser.parse_args()
    
    gen_path = Path(args.gen_file)
    if not gen_path.exists():
        print(f"Error: File not found: {gen_path}")
        sys.exit(1)
        
    # 1. 尝试定位 Experiment 目录和 Config
    exp_dir = gen_path.parent.parent
    if args.config:
        config_path = args.config
    else:
        config_path = exp_dir / "config.yaml"
        
    if not os.path.exists(config_path):
        # Fallback to default if exp config not found (e.g. mixed path)
        print(f"[Warning] Config not found at {config_path}, trying configs/official.yaml")
        config_path = "configs/official.yaml"
        
    # 2. 初始化 ExperimentManager (为了加载 Config)
    # 这一步会创建一个新的 experiment 目录，或者我们可以 mock 一下以免生成垃圾日志
    # 这里我们简单地读取 yaml，手动构建 config 字典
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    # Hack: 注入全局 EXP_MANAGER 以便 PromptIndividual 使用
    # (更优雅的方式是重构代码，但为了复用 wise_aes.py，我们模拟这个全局变量)
    import wise_aes
    # 创建一个伪造的 ExperimentManager 只为了持有 config
    class MockExpManager:
        def __init__(self, cfg): self.config = cfg
        def log_llm_trace(self, x): pass # 甚至不 log
    
    wise_aes.EXP_MANAGER = MockExpManager(config)

    # 3. 加载数据并切分出 Test Set
    data_path = config['data']['asap_path']
    essay_set = config['data']['essay_set']
    print(f"[Data] Loading ASAP Data Set {essay_set} from {data_path}...")
    
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    df = df[df['essay_set'] == essay_set]
    all_data = []
    for _, row in df.iterrows():
        all_data.append({
            "essay_id": row['essay_id'],
            "essay_text": row['essay'],
            "domain1_score": int(row['domain1_score'])
        })
        
    # 切分逻辑 (必须与 wise_aes.py 中的一致)
    # 注意：如果 wise_aes.py 中的随机种子变了，这里也会无法复现 Test Set。
    # 确保 wise_aes.py 中使用的是固定的 random_state=42
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(all_data))
    train_val_idx, test_idx = folds[args.fold]
    
    test_set = [all_data[i] for i in test_idx]
    train_set = [all_data[i] for i in train_val_idx] # 我们需要 Train Set 来构建 Vector Store
    
    # 4. 构建 Vector Store (用于 RAG 和 还原 Exemplars)
    print("[VectorStore] Building store from Train Set...")
    vector_store = SimpleVectorStore(model_name=config['rag']['model_name'])
    vector_store.add_documents(train_set)
    
    # 5. 加载种群
    population_data = load_generation_file(args.gen_file)
    
    # 6. 评测
    results = evaluate_population_on_test(population_data, test_set, config, vector_store)
    
    # 7. 打印摘要
    print("\n" + "="*40)
    print(f"Evaluation Summary for {gen_path.name}")
    print("="*40)
    print(f"{'ID':<4} | {'Val QWK':<10} | {'Test QWK':<10}")
    print("-" * 30)
    for res in results:
        print(f"{res['individual_id']:<4} | {res['val_fitness']:.4f}     | {res['test_qwk']:.4f}")
        
    # 保存结果到 csv
    output_csv = gen_path.with_name(gen_path.stem + "_test_eval.csv")
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to {output_csv}")

if __name__ == "__main__":
    main_eval()
