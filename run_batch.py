import subprocess
import time
import os
import sys
from datetime import datetime

# ==============================================================================
# 配置区域：请在这里手动列出你要跑的 config 文件路径
# ==============================================================================
CONFIG_FILES = [
    # "configs/default.yaml",
    # "configs/ablation_no_rag.yaml",
    # "configs/ablation_static_5.yaml",
    # "configs/ablation_no_evolution.yaml",
    "configs/batch_test_1.yaml",
    "configs/batch_test_2.yaml",
]
# ==============================================================================

def run_experiment(config_path):
    """运行单个实验并记录日志"""
    
    # 1. 检查文件是否存在
    if not os.path.exists(config_path):
        return "MISSING", 0.0

    # 2. 准备日志文件名
    exp_name = os.path.basename(config_path).replace(".yaml", "").replace(".yml", "")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "logs/batch_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}_{timestamp}.log")
    
    print(f"🚀 [{datetime.now().strftime('%H:%M:%S')}] Starting: {config_path}")
    print(f"   Log > {log_file}")

    start_time = time.time()
    
    # 3. 调用 wise_aes.py
    # 使用 subprocess 启动子进程，将 stdout 和 stderr 都重定向到 log 文件
    with open(log_file, "w", encoding="utf-8") as f:
        try:
            # unbuffered (-u) 保证日志实时写入
            process = subprocess.run(
                [sys.executable, "-u", "wise_aes.py", "--config", config_path],
                stdout=f,
                stderr=subprocess.STDOUT,
                encoding="utf-8",
                check=False # 即使报错也不要抛出异常，由返回值判断
            )
            return_code = process.returncode
        except Exception as e:
            f.write(f"\nCRITICAL ERROR EXECUTING SCRIPT: {e}\n")
            return_code = -1

    duration = time.time() - start_time
    
    if return_code == 0:
        print(f"✅ Finished in {duration:.1f}s")
        return "SUCCESS", duration
    else:
        print(f"❌ Failed (Code {return_code}) in {duration:.1f}s. Check log for details.")
        return "FAILED", duration

def main():
    print(f"=== WISE-AES Batch Runner ===")
    print(f"Found {len(CONFIG_FILES)} configurations to run sequentially.\n")
    
    results = []
    total_start = time.time()
    
    for i, config in enumerate(CONFIG_FILES):
        print(f"--- Task {i+1}/{len(CONFIG_FILES)} ---")
        status, duration = run_experiment(config)
        results.append({
            "config": config,
            "status": status,
            "duration": duration
        })
        print("-" * 50)
        
        # 可选：实验间隔休息一下，给 API 喘息时间
        if i < len(CONFIG_FILES) - 1:
            time.sleep(2)

    total_duration = time.time() - total_start
    
    # === 最终报告 ===
    print("\n" + "="*50)
    print("BATCH EXECUTION SUMMARY")
    print("="*50)
    print(f"{'Config File':<35} | {'Status':<10} | {'Time (s)':<10}")
    print("-" * 60)
    
    success_count = 0
    for res in results:
        print(f"{res['config']:<35} | {res['status']:<10} | {res['duration']:.1f}")
        if res['status'] == "SUCCESS":
            success_count += 1
            
    print("-" * 60)
    print(f"Total Time: {total_duration/60:.1f} min")
    print(f"Completed:  {success_count}/{len(CONFIG_FILES)}")
    print("="*50)

if __name__ == "__main__":
    main()