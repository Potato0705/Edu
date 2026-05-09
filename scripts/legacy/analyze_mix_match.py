
import pandas as pd
import argparse
import sys
import os

def analyze(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty.")
        return
        
    required = ['prompt', 'qwk']
    if not all(col in df.columns for col in required):
        print("Missing columns.")
        return
        
    print("\n--- Mix-and-Match Results (Instruction: ICL+Inst+ | Exemplar: ICL+Ex+) ---")
    print(f"{'Prompt':<10} {'Mean QWK':<15} {'Std Dev':<15} {'Count':<10}")
    print("-" * 50)
    
    stats = df.groupby('prompt')['qwk'].agg(['mean', 'std', 'count']).sort_index()
    
    for p, row in stats.iterrows():
        print(f"{p:<10} {row['mean']:<15.4f} {row['std']:<15.4f} {int(row['count']):<10}")
        
    macro_avg = stats['mean'].mean()
    print("-" * 50)
    print(f"\nFinal Macro Average QWK: {macro_avg:.4f}")

if __name__ == "__main__":
    analyze("mix_match_results.csv")
