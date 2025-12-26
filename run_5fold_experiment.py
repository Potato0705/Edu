import subprocess
import sys
import argparse
import re
import os
import yaml
from pathlib import Path

def run_command(cmd, cwd=None):
    """Runs a shell command and returns stdout, stderr."""
    print(f"[Run] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"[Error] Command failed with code {result.returncode}")
        print(result.stderr)
        return None
    return result.stdout

def run_fold(config_path, fold):
    """Runs wise_aes.py for a specific fold and returns the exp directory."""
    print(f"\n{'='*20} Running Fold {fold} {'='*20}")
    
    cmd = ["uv", "run", "wise_aes.py", "--config", config_path, "--fold", str(fold)]
    
    # We need to capture stdout to find the exp directory logging
    # But we also want to stream output to console so user sees progress.
    # To do both, we can use Popen.
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    exp_dir = None
    output_lines = []
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.rstrip()) # print to console
            output_lines.append(line)
            # Try to grab exp dir: "Result Directory: logs/exp_..."
            if "Result Directory:" in line:
                match = re.search(r"Result Directory: (.+)", line)
                if match:
                    exp_dir = match.group(1).strip()
    
    return exp_dir

def main():
    parser = argparse.ArgumentParser(description="Run full 5-fold experiment and evaluation.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)
        
    # 1. Read Config to get n_generations
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        n_gens = config_data.get('evolution', {}).get('n_generations', 20)
    
    print(f"Target Generations: {n_gens}")
    
    exp_dirs = []
    
    # 2. Run 5 Folds
    for fold in range(5):
        exp_dir = run_fold(args.config, fold)
        if not exp_dir:
            print(f"[Fatal] Fold {fold} failed to return an experiment directory.")
            sys.exit(1)
        exp_dirs.append(exp_dir)
        
    print("\n" + "="*40)
    print("5 Folds Completed. Directories:")
    for d in exp_dirs:
        print(f" - {d}")
    print("="*40)
    
    # 3. Run Evaluation
    print("\nStarting 5-Fold Standard Evaluation...")
    eval_cmd = [
        "uv", "run", "eval_5fold.py",
        "--gen", str(n_gens),
        "--dirs"
    ] + exp_dirs
    
    run_command(eval_cmd)
    
    print("\nAll Done!")

if __name__ == "__main__":
    main()
