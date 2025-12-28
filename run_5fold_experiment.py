import subprocess
import sys
import argparse
import re
import os
import yaml
import json
import datetime
from pathlib import Path

STATUS_FILE = "5fold_status.json"

def run_command(cmd, cwd=None):
    """Runs a shell command and returns stdout, stderr."""
    print(f"[Run] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, text=True, capture_output=True)
    if result.returncode != 0:
        print(f"[Error] Command failed with code {result.returncode}")
        print(result.stderr)
        return None
    return result.stdout

def save_status(status_data):
    with open(STATUS_FILE, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, indent=2)

def load_status():
    if os.path.exists(STATUS_FILE):
        with open(STATUS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def find_latest_checkpoint(exp_dir):
    """Scan exp_dir/generations to find the latest gen_xxx.json to resume from."""
    if not exp_dir: return None
    gen_dir = Path(exp_dir) / "generations"
    if not gen_dir.exists(): return None
    
    files = sorted(gen_dir.glob("gen_*.json"))
    if files:
        # Return path to the last one
        return str(files[-1])
    return None

def run_fold(config_path, fold, resume_exp_dir=None):
    """Runs wise_aes.py for a specific fold. Can resume if resume_exp_dir is provided."""
    print(f"\n{'='*20} Running Fold {fold} {'='*20}")
    
    cmd = ["uv", "run", "wise_aes.py", "--config", config_path, "--fold", str(fold)]
    
    # Check if we can resume this specific fold
    if resume_exp_dir:
        latest_ckpt = find_latest_checkpoint(resume_exp_dir)
        if latest_ckpt:
            print(f"[Resume] Resuming Fold {fold} from checkpoint: {latest_ckpt}")
            cmd.extend(["--resume", latest_ckpt])
        else:
            print(f"[Resume] Fold {fold} marked as started but no checkpoint found. Restarting.")
    
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    final_exp_dir = resume_exp_dir # Default to what we had if resuming
    
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.rstrip()) 
            # Try to grab exp dir if it's a new run
            if "Result Directory:" in line:
                match = re.search(r"Result Directory: (.+)", line)
                if match:
                    final_exp_dir = match.group(1).strip()
    
    if process.returncode != 0:
        print(f"[Error] Fold {fold} failed.")
        return None
        
    return final_exp_dir

def main():
    parser = argparse.ArgumentParser(description="Run full 5-fold experiment with Resume capability.")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from 5fold_status.json if exists")
    parser.add_argument("--fold_dirs", nargs="*", help="List of existing fold directories to use/resume (ordered Fold 0, 1...). Bypasses status file.")
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load Status or Init New
    status = {"folds": {}} # { "0": {"status": "done", "exp_dir": "..."} }
    
    # Mode 1: Manual Fold Dirs (Highest Priority)
    if args.fold_dirs:
        print(f"[Init] Using provided fold directories ({len(args.fold_dirs)} provided). Ignoring {STATUS_FILE}.")
        for i, d in enumerate(args.fold_dirs):
            if i >= 5: 
                print(f"[Warn] More than 5 directories provided. Ignoring extras: {d}")
                break
            
            d_path = Path(d)
            if not d_path.exists():
                print(f"[Warn] Provided directory does not exist: {d}. Treating as new run (pending).")
                continue
                
            # Check if done
            is_done = (d_path / "final_result.json").exists()
            status["folds"][str(i)] = {
                "status": "done" if is_done else "running",
                "exp_dir": str(d_path)
            }
            
    # Mode 2: Resume from Status File
    elif args.resume:
        loaded = load_status()
        if loaded:
            print(f"[Init] Loaded status from {STATUS_FILE}")
            status = loaded
        else:
            print(f"[Init] No status file found, starting fresh.")
            
    # 1. Read Config to get n_generations
    with open(args.config, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        n_gens = config_data.get('evolution', {}).get('n_generations', 20)
    
    print(f"Target Generations: {n_gens}")
    
    # 2. Run 5 Folds
    exp_dirs_map = {}
    
    for fold in range(5):
        s_fold = str(fold)
        fold_info = status["folds"].get(s_fold, {"status": "pending"})
        
        current_status = fold_info.get("status")
        exp_dir = fold_info.get("exp_dir")
        
        if current_status == "done":
            print(f"[Skip] Fold {fold} already done. Exp: {exp_dir}")
            exp_dirs_map[fold] = exp_dir
            continue
            
        # Run or Resume Fold
        if current_status == "running" and exp_dir:
            # Try to resume interrupted fold
            new_exp_dir = run_fold(args.config, fold, resume_exp_dir=exp_dir)
        else:
            # Start fresh
            new_exp_dir = run_fold(args.config, fold)
            
        if not new_exp_dir:
            print(f"[Fatal] Fold {fold} failed.")
            sys.exit(1)
            
        # Update Status
        status["folds"][s_fold] = {
            "status": "done",
            "exp_dir": new_exp_dir,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Only save status file if NOT using manual fold_dirs (to avoid conflicts)
        if not args.fold_dirs:
            save_status(status)
            
        exp_dirs_map[fold] = new_exp_dir
        
    # Collect ordered dirs
    final_exp_dirs = [exp_dirs_map[i] for i in range(5)]
    
    print("\n" + "="*40)
    print("5 Folds Completed. Directories:")
    for d in final_exp_dirs:
        print(f" - {d}")
    print("="*40)
    
    # 3. Run Evaluation
    print("\nStarting 5-Fold Standard Evaluation...")
    eval_cmd = [
        "uv", "run", "eval_5fold.py",
        "--gen", str(n_gens),
        "--dirs"
    ] + final_exp_dirs
    
    run_command(eval_cmd)
    
    print("\nAll Done!")

if __name__ == "__main__":
    main()
