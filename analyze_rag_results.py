
import pandas as pd
import sys

def analyze_results(csv_path):
    print(f"Reading results from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File {csv_path} not found.")
        return

    # Ensure required columns exist
    required_columns = ['prompt', 'qwk']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain columns: {required_columns}")
        return

    # Group by prompt and calculate stats
    print("\n--- Per-Prompt Results ---")
    print(f"{'Prompt':<10} {'Mean QWK':<15} {'Std Dev':<15} {'Count':<10}")
    print("-" * 50)

    prompt_stats = df.groupby('prompt')['qwk'].agg(['mean', 'std', 'count']).sort_index()
    
    for prompt, row in prompt_stats.iterrows():
        print(f"{prompt:<10} {row['mean']:<15.4f} {row['std']:<15.4f} {int(row['count']):<10}")

    # Calculate overall average (Macro Average of Prompts)
    overall_average = prompt_stats['mean'].mean()
    
    print("-" * 50)
    print(f"\nFinal Macro Average QWK (across {len(prompt_stats)} prompts): {overall_average:.4f}")

if __name__ == "__main__":
    analyze_results("rag_substitution_results.csv")
