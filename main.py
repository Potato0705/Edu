"""
WISE-AES: Weakly-supervised Integrated Scoring Evolution
针对 ASAP Dataset Prompt 8 (Laughter Topic) 的自动化评分系统

核心特点:
1. 指令诱导 (Instruction Induction) - 自动生成评分 Rubric
2. 双重进化优化:
   - 遗传算法优化 Few-Shot 范文选择
   - LLM 反馈驱动的 Prompt 文本优化
"""

import os
import re
import random
import json
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
import numpy as np
import yaml
import pandas as pd

# 加载环境变量
load_dotenv()


# ============================================================================
# 实验日志记录器
# ============================================================================

class ExperimentLogger:
    """实验过程和结果的详细记录器"""
    
    def __init__(self, result_dir: str = "result", resume_dir: str = None):
        self.result_dir = Path(result_dir)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        
        if resume_dir:
            # 从已有实验目录恢复
            self.exp_dir = Path(resume_dir)
            if not self.exp_dir.exists():
                raise ValueError(f"Resume directory not found: {resume_dir}")
            self.timestamp = self.exp_dir.name.replace("exp_", "")
            self.is_resumed = True
        else:
            # 创建带时间戳的实验目录
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.exp_dir = self.result_dir / f"exp_{self.timestamp}"
            self.exp_dir.mkdir(parents=True, exist_ok=True)
            self.is_resumed = False
        
        # 日志文件
        self.log_file = self.exp_dir / "experiment.log"
        self.llm_calls_file = self.exp_dir / "llm_calls.jsonl"
        
        # 内存中的记录
        self.logs = []
        self.llm_calls = []
        self.generation_details = []
        
        if self.is_resumed:
            self._log(f"Experiment RESUMED at {datetime.now().strftime('%Y%m%d_%H%M%S')}")
            self._log(f"Resuming from: {self.exp_dir}")
            # 加载已有的 generation_details
            self._load_existing_generations()
        else:
            self._log(f"Experiment started at {self.timestamp}")
            self._log(f"Result directory: {self.exp_dir}")
    
    def _load_existing_generations(self):
        """加载已有的 generation 记录"""
        gen_files = sorted(self.exp_dir.glob("generation_*.json"))
        for gen_file in gen_files:
            with open(gen_file, 'r', encoding='utf-8') as f:
                gen_data = json.load(f)
                self.generation_details.append(gen_data)
        if self.generation_details:
            self._log(f"Loaded {len(self.generation_details)} existing generations")
    
    def _log(self, message: str, also_print: bool = True):
        """记录日志"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.logs.append(log_entry)
        
        # 追加写入文件
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry + "\n")
        
        if also_print:
            print(message)
    
    def log_config(self, config: Dict[str, Any]):
        """记录实验配置"""
        self._log("=" * 60)
        self._log("EXPERIMENT CONFIGURATION")
        self._log("=" * 60)
        
        # 保存配置文件
        config_file = self.exp_dir / "config.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
        self._log(f"Config saved to: {config_file}")
        for section, values in config.items():
            self._log(f"\n[{section}]")
            if isinstance(values, dict):
                for k, v in values.items():
                    self._log(f"  {k}: {v}")
            else:
                self._log(f"  {values}")
    
    def log_llm_call(self, call_type: str, prompt: str, response: str, 
                     temperature: float, duration: float = None):
        """记录 LLM 调用详情"""
        call_record = {
            "timestamp": datetime.now().isoformat(),
            "type": call_type,
            "prompt_length": len(prompt),
            "response_length": len(response),
            "temperature": temperature,
            "duration_seconds": duration,
            "prompt": prompt,
            "response": response
        }
        self.llm_calls.append(call_record)
        
        # 追加写入 JSONL 文件
        with open(self.llm_calls_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(call_record, ensure_ascii=False) + "\n")
    
    def log_data_stats(self, train_data: List, val_data: List, all_data: List):
        """记录数据统计"""
        self._log("\n" + "=" * 60)
        self._log("DATA STATISTICS")
        self._log("=" * 60)
        
        stats = {
            "total_essays": len(all_data),
            "train_essays": len(train_data),
            "val_essays": len(val_data),
            "train_scores": {
                "min": min(d['domain1_score'] for d in train_data),
                "max": max(d['domain1_score'] for d in train_data),
                "mean": np.mean([d['domain1_score'] for d in train_data]),
                "std": np.std([d['domain1_score'] for d in train_data])
            },
            "val_scores": {
                "min": min(d['domain1_score'] for d in val_data),
                "max": max(d['domain1_score'] for d in val_data),
                "mean": np.mean([d['domain1_score'] for d in val_data]),
                "std": np.std([d['domain1_score'] for d in val_data])
            }
        }
        
        self._log(f"Total essays: {stats['total_essays']}")
        self._log(f"Training set: {stats['train_essays']} essays")
        self._log(f"Validation set: {stats['val_essays']} essays")
        self._log(f"Train score range: {stats['train_scores']['min']}-{stats['train_scores']['max']}")
        self._log(f"Train score mean: {stats['train_scores']['mean']:.2f} ± {stats['train_scores']['std']:.2f}")
        self._log(f"Val score range: {stats['val_scores']['min']}-{stats['val_scores']['max']}")
        self._log(f"Val score mean: {stats['val_scores']['mean']:.2f} ± {stats['val_scores']['std']:.2f}")
        
        # 保存数据统计
        stats_file = self.exp_dir / "data_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        return stats
    
    def log_initial_rubric(self, rubric: str):
        """记录初始 Rubric"""
        self._log("\n" + "=" * 60)
        self._log("INITIAL RUBRIC (from Instruction Induction)")
        self._log("=" * 60)
        self._log(rubric, also_print=False)
        
        # 保存完整 rubric
        rubric_file = self.exp_dir / "initial_rubric.txt"
        with open(rubric_file, 'w', encoding='utf-8') as f:
            f.write(rubric)
        
        self._log(f"Initial rubric saved to: {rubric_file}")
        self._log(f"Rubric preview: {rubric[:300]}...")
    
    def log_generation_start(self, generation: int):
        """记录代开始"""
        self._log("\n" + "=" * 60)
        self._log(f"GENERATION {generation}")
        self._log("=" * 60)
    
    def log_individual_evaluation(self, ind_idx: int, qwk: float, 
                                   true_scores: List[int], pred_scores: List[int]):
        """记录个体评估结果"""
        errors = [abs(t - p) for t, p in zip(true_scores, pred_scores)]
        mae = np.mean(errors)
        
        self._log(f"  Individual {ind_idx}: QWK={qwk:.4f}, MAE={mae:.2f}")
        
        return {
            "individual_idx": ind_idx,
            "qwk": qwk,
            "mae": mae,
            "true_scores": true_scores,
            "pred_scores": pred_scores,
            "errors": errors
        }
    
    def log_generation_summary(self, generation: int, population: List, 
                                best_individual, best_overall):
        """记录代总结"""
        qwks = [ind.fitness for ind in population]
        
        gen_record = {
            "generation": generation,
            "best_qwk": max(qwks),
            "avg_qwk": np.mean(qwks),
            "min_qwk": min(qwks),
            "std_qwk": np.std(qwks),
            "best_overall_qwk": best_overall.fitness,
            "best_instruction": best_individual.instruction_text,  # 完整 instruction
            "exemplar_ids": [ex['essay_id'] for ex in best_individual.exemplars]
        }
        self.generation_details.append(gen_record)
        
        self._log(f"\n[Generation {generation} Summary]")
        self._log(f"  Best QWK: {gen_record['best_qwk']:.4f}")
        self._log(f"  Avg QWK: {gen_record['avg_qwk']:.4f}")
        self._log(f"  Std QWK: {gen_record['std_qwk']:.4f}")
        self._log(f"  Best Overall: {gen_record['best_overall_qwk']:.4f}")
        
        # 保存每代详情
        gen_file = self.exp_dir / f"generation_{generation:02d}.json"
        with open(gen_file, 'w', encoding='utf-8') as f:
            json.dump(gen_record, f, ensure_ascii=False, indent=2)
        
        return gen_record
    
    def save_checkpoint(self, generation: int, population: List, 
                        best_individual, train_data: List, val_data: List,
                        initial_rubric: str):
        """保存 checkpoint 用于断点续传"""
        checkpoint = {
            "generation": generation,
            "best_individual": {
                "instruction_text": best_individual.instruction_text,
                "exemplar_ids": [ex['essay_id'] for ex in best_individual.exemplars],
                "fitness": best_individual.fitness
            },
            "population": [
                {
                    "instruction_text": ind.instruction_text,
                    "exemplar_ids": [ex['essay_id'] for ex in ind.exemplars],
                    "fitness": ind.fitness
                }
                for ind in population
            ],
            "initial_rubric": initial_rubric,
            "train_essay_ids": [d['essay_id'] for d in train_data],
            "val_essay_ids": [d['essay_id'] for d in val_data],
            "timestamp": datetime.now().isoformat()
        }
        
        checkpoint_file = self.exp_dir / "checkpoint.json"
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint, f, ensure_ascii=False, indent=2)
        
        self._log(f"  [Checkpoint saved at generation {generation}]", also_print=False)
    
    def load_checkpoint(self) -> Optional[Dict]:
        """加载 checkpoint"""
        checkpoint_file = self.exp_dir / "checkpoint.json"
        if not checkpoint_file.exists():
            return None
        
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_last_completed_generation(self) -> int:
        """获取最后完成的代数"""
        if not self.generation_details:
            return 0
        return max(g['generation'] for g in self.generation_details)
    
    def log_feedback(self, elite_idx: int, feedback: str):
        """记录反馈"""
        self._log(f"\n  [Elite {elite_idx} Feedback]")
        self._log(f"  {feedback[:300]}...", also_print=False)
        
        feedback_file = self.exp_dir / f"feedback_elite_{elite_idx}.txt"
        with open(feedback_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*40}\n{datetime.now().isoformat()}\n{'='*40}\n")
            f.write(feedback + "\n")
    
    def log_instruction_evolution(self, elite_idx: int, old_instruction: str, 
                                   new_instruction: str):
        """记录指令进化"""
        self._log(f"  [Elite {elite_idx} Instruction Evolved]")
        
        evo_file = self.exp_dir / f"instruction_evolution_elite_{elite_idx}.txt"
        with open(evo_file, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n\n")
            f.write("=== OLD INSTRUCTION ===\n")
            f.write(old_instruction + "\n\n")
            f.write("=== NEW INSTRUCTION ===\n")
            f.write(new_instruction + "\n")
    
    def plot_evolution_results(self):
        """绘制进化结果图表"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # 非交互式后端
        
        if not self.generation_details:
            self._log("No generation data to plot")
            return
        
        generations = [g['generation'] for g in self.generation_details]
        best_qwks = [g['best_qwk'] for g in self.generation_details]
        avg_qwks = [g['avg_qwk'] for g in self.generation_details]
        min_qwks = [g['min_qwk'] for g in self.generation_details]
        std_qwks = [g['std_qwk'] for g in self.generation_details]
        best_overall_qwks = [g['best_overall_qwk'] for g in self.generation_details]
        exemplar_ids_list = [g['exemplar_ids'] for g in self.generation_details]
        
        # 图1: QWK 变化曲线
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        
        ax1.plot(generations, best_qwks, 'b-o', linewidth=2, markersize=8, label='Best QWK (this gen)')
        ax1.plot(generations, avg_qwks, 'g--s', linewidth=1.5, markersize=6, label='Avg QWK')
        ax1.plot(generations, min_qwks, 'r:^', linewidth=1.5, markersize=6, label='Min QWK')
        ax1.plot(generations, best_overall_qwks, 'purple', linewidth=2.5, marker='*', 
                 markersize=10, label='Best Overall QWK')
        
        # 添加误差带
        avg_arr = np.array(avg_qwks)
        std_arr = np.array(std_qwks)
        ax1.fill_between(generations, avg_arr - std_arr, avg_arr + std_arr, 
                         alpha=0.2, color='green', label='±1 Std Dev')
        
        ax1.set_xlabel('Generation', fontsize=12)
        ax1.set_ylabel('QWK Score', fontsize=12)
        ax1.set_title('Evolution of QWK Scores Across Generations', fontsize=14)
        ax1.legend(loc='lower right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks(generations)
        
        # 标注最终最佳值
        final_best = best_overall_qwks[-1]
        ax1.annotate(f'Final Best: {final_best:.4f}', 
                     xy=(generations[-1], final_best),
                     xytext=(generations[-1] - 0.5, final_best + 0.02),
                     fontsize=10, fontweight='bold',
                     arrowprops=dict(arrowstyle='->', color='purple'))
        
        plt.tight_layout()
        qwk_plot_path = self.exp_dir / "plot_qwk_evolution.png"
        fig1.savefig(qwk_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        self._log(f"QWK evolution plot saved to: {qwk_plot_path}")
        
        # 图2: Exemplar ID 变化热力图
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        
        # 收集所有出现过的 exemplar IDs
        all_ids = set()
        for ids in exemplar_ids_list:
            all_ids.update(ids)
        all_ids = sorted(all_ids)
        
        # 创建热力图数据
        heatmap_data = np.zeros((len(all_ids), len(generations)))
        id_to_idx = {id_: idx for idx, id_ in enumerate(all_ids)}
        
        for gen_idx, ids in enumerate(exemplar_ids_list):
            for id_ in ids:
                heatmap_data[id_to_idx[id_], gen_idx] = 1
        
        im = ax2.imshow(heatmap_data, aspect='auto', cmap='Blues', interpolation='nearest')
        
        ax2.set_xlabel('Generation', fontsize=12)
        ax2.set_ylabel('Exemplar Essay ID', fontsize=12)
        ax2.set_title('Exemplar Selection Across Generations', fontsize=14)
        ax2.set_xticks(range(len(generations)))
        ax2.set_xticklabels(generations)
        ax2.set_yticks(range(len(all_ids)))
        ax2.set_yticklabels(all_ids, fontsize=8)
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax2)
        cbar.set_label('Selected (1) / Not Selected (0)', fontsize=10)
        
        plt.tight_layout()
        exemplar_plot_path = self.exp_dir / "plot_exemplar_evolution.png"
        fig2.savefig(exemplar_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        self._log(f"Exemplar evolution plot saved to: {exemplar_plot_path}")
        
        # 图3: 综合仪表板
        fig3, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 子图1: QWK 趋势
        ax3_1 = axes[0, 0]
        ax3_1.plot(generations, best_qwks, 'b-o', linewidth=2, label='Best')
        ax3_1.plot(generations, avg_qwks, 'g--', linewidth=1.5, label='Avg')
        ax3_1.fill_between(generations, min_qwks, best_qwks, alpha=0.3, color='blue')
        ax3_1.set_xlabel('Generation')
        ax3_1.set_ylabel('QWK')
        ax3_1.set_title('QWK Trend')
        ax3_1.legend()
        ax3_1.grid(True, alpha=0.3)
        
        # 子图2: QWK 改进量
        ax3_2 = axes[0, 1]
        improvements = [0] + [best_overall_qwks[i] - best_overall_qwks[i-1] 
                              for i in range(1, len(best_overall_qwks))]
        colors = ['green' if x >= 0 else 'red' for x in improvements]
        ax3_2.bar(generations, improvements, color=colors, alpha=0.7, edgecolor='black')
        ax3_2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3_2.set_xlabel('Generation')
        ax3_2.set_ylabel('QWK Improvement')
        ax3_2.set_title('QWK Improvement per Generation')
        ax3_2.grid(True, alpha=0.3, axis='y')
        
        # 子图3: 种群多样性 (标准差)
        ax3_3 = axes[1, 0]
        ax3_3.bar(generations, std_qwks, color='orange', alpha=0.7, edgecolor='black')
        ax3_3.set_xlabel('Generation')
        ax3_3.set_ylabel('Std Dev of QWK')
        ax3_3.set_title('Population Diversity (QWK Std Dev)')
        ax3_3.grid(True, alpha=0.3, axis='y')
        
        # 子图4: Exemplar 变化追踪
        ax3_4 = axes[1, 1]
        # 计算每代 exemplar 的变化数量
        changes = [0]
        for i in range(1, len(exemplar_ids_list)):
            prev_set = set(exemplar_ids_list[i-1])
            curr_set = set(exemplar_ids_list[i])
            changed = len(prev_set.symmetric_difference(curr_set))
            changes.append(changed)
        
        ax3_4.bar(generations, changes, color='purple', alpha=0.7, edgecolor='black')
        ax3_4.set_xlabel('Generation')
        ax3_4.set_ylabel('# Exemplars Changed')
        ax3_4.set_title('Exemplar Changes per Generation')
        ax3_4.grid(True, alpha=0.3, axis='y')
        
        # 在每个柱子上标注具体的 exemplar IDs
        for i, (gen, ids) in enumerate(zip(generations, exemplar_ids_list)):
            id_str = '\n'.join([str(id_) for id_ in ids])
            ax3_4.annotate(id_str, xy=(gen, changes[i] + 0.1), 
                          ha='center', va='bottom', fontsize=7, color='gray')
        
        plt.suptitle('WISE-AES Evolution Dashboard', fontsize=16, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        dashboard_path = self.exp_dir / "plot_dashboard.png"
        fig3.savefig(dashboard_path, dpi=150, bbox_inches='tight')
        plt.close(fig3)
        self._log(f"Dashboard plot saved to: {dashboard_path}")
        
        return [qwk_plot_path, exemplar_plot_path, dashboard_path]

    def save_final_results(self, best_individual, optimizer, config: Dict):
        """保存最终结果"""
        self._log("\n" + "=" * 60)
        self._log("FINAL RESULTS")
        self._log("=" * 60)
        
        # 先绘制图表
        self._log("\n[Generating plots...]")
        plot_paths = self.plot_evolution_results()
        
        # 最终结果
        final_result = {
            "experiment_id": self.timestamp,
            "config": config,
            "best_qwk": best_individual.fitness,
            "final_instruction": best_individual.instruction_text,
            "final_exemplar_ids": [ex['essay_id'] for ex in best_individual.exemplars],
            "final_exemplars": [
                {
                    "essay_id": ex['essay_id'],
                    "score": ex['domain1_score'],
                    "essay_preview": ex['essay_text'][:500]
                }
                for ex in best_individual.exemplars
            ],
            "evolution_history": self.generation_details,
            "total_llm_calls": len(self.llm_calls),
            "completed_at": datetime.now().isoformat()
        }
        
        # 保存主结果文件
        result_file = self.exp_dir / "final_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        # 保存最终 instruction
        instruction_file = self.exp_dir / "final_instruction.txt"
        with open(instruction_file, 'w', encoding='utf-8') as f:
            f.write(best_individual.instruction_text)
        
        # 保存进化历史摘要
        history_file = self.exp_dir / "evolution_history.json"
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.generation_details, f, ensure_ascii=False, indent=2)
        
        # 打印摘要
        self._log(f"\nBest QWK: {best_individual.fitness:.4f}")
        self._log(f"Total LLM calls: {len(self.llm_calls)}")
        self._log(f"\nResults saved to: {self.exp_dir}")
        self._log(f"  - final_result.json")
        self._log(f"  - final_instruction.txt")
        self._log(f"  - evolution_history.json")
        self._log(f"  - experiment.log")
        self._log(f"  - llm_calls.jsonl")
        self._log(f"  - plot_qwk_evolution.png")
        self._log(f"  - plot_exemplar_evolution.png")
        self._log(f"  - plot_dashboard.png")
        
        return final_result


# 全局 logger 实例
LOGGER: Optional[ExperimentLogger] = None

# ============================================================================
# 配置加载
# ============================================================================

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

# 全局配置
CONFIG = load_config()

# ============================================================================
# OpenRouter LLM 调用封装
# ============================================================================

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct")
OPENROUTER_PROVIDER = os.getenv("OPENROUTER_PROVIDER", "")


def call_llm(prompt: str, system_prompt: str = "", temperature: float = 0.7,
             call_type: str = "unknown") -> str:
    """
    调用 OpenRouter API 进行 LLM 推理
    """
    import time
    start_time = time.time()
    
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://wise-aes.local",
        "X-Title": "WISE-AES Experiment"
    }
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    
    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    
    # 添加 provider 配置
    if OPENROUTER_PROVIDER:
        parts = OPENROUTER_PROVIDER.split("/")
        payload["provider"] = {"order": [parts[0]]}
        if len(parts) > 1:
            payload["provider"]["quantizations"] = [parts[1]]
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]
        
        # 记录 LLM 调用
        duration = time.time() - start_time
        if LOGGER:
            LOGGER.log_llm_call(call_type, prompt, content, temperature, duration)
        
        return content
    except Exception as e:
        print(f"[LLM Error] {e}")
        return f"[ERROR] LLM call failed: {e}"


# ============================================================================
# 评估指标: Quadratic Weighted Kappa (QWK)
# ============================================================================

def quadratic_weighted_kappa(y_true: List[int], y_pred: List[int], 
                              min_rating: int = None, max_rating: int = None) -> float:
    """计算 Quadratic Weighted Kappa (QWK) 分数"""
    if min_rating is None:
        min_rating = CONFIG['data']['score_min']
    if max_rating is None:
        max_rating = CONFIG['data']['score_max']
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.clip(y_pred, min_rating, max_rating)
    
    num_ratings = max_rating - min_rating + 1
    
    hist_true = np.zeros(num_ratings)
    hist_pred = np.zeros(num_ratings)
    confusion = np.zeros((num_ratings, num_ratings))
    
    for t, p in zip(y_true, y_pred):
        t_idx = int(t - min_rating)
        p_idx = int(p - min_rating)
        hist_true[t_idx] += 1
        hist_pred[p_idx] += 1
        confusion[t_idx, p_idx] += 1
    
    weights = np.zeros((num_ratings, num_ratings))
    for i in range(num_ratings):
        for j in range(num_ratings):
            weights[i, j] = ((i - j) ** 2) / ((num_ratings - 1) ** 2)
    
    n = len(y_true)
    expected = np.outer(hist_true, hist_pred) / n
    
    observed_weighted = np.sum(weights * confusion)
    expected_weighted = np.sum(weights * expected)
    
    if expected_weighted == 0:
        return 1.0
    
    return 1.0 - (observed_weighted / expected_weighted)


# ============================================================================
# ASAP 数据加载
# ============================================================================

def load_asap_data(config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    加载真实 ASAP 数据集
    
    ASAP 数据格式: TSV 文件，包含 essay_id, essay_set, essay, domain1_score 等列
    """
    if config is None:
        config = CONFIG
    
    data_path = config['data']['asap_path']
    essay_set = config['data']['essay_set']
    
    print(f"[Data] Loading ASAP data from {data_path}...")
    
    # 读取 TSV 文件
    df = pd.read_csv(data_path, sep='\t', encoding='latin-1')
    
    # 筛选指定的 essay_set
    df = df[df['essay_set'] == essay_set]
    
    # 转换为字典列表
    data = []
    for _, row in df.iterrows():
        data.append({
            'essay_id': row['essay_id'],
            'essay_text': row['essay'],
            'domain1_score': row['domain1_score'],
            'essay_set': row['essay_set']
        })
    
    print(f"[Data] Loaded {len(data)} essays for Prompt {essay_set}")
    
    # 显示分数分布
    scores = [d['domain1_score'] for d in data]
    print(f"[Data] Score range: {min(scores)} - {max(scores)}")
    print(f"[Data] Mean score: {np.mean(scores):.1f}, Std: {np.std(scores):.1f}")
    
    return data


# ============================================================================
# 模块 A: InstructionInductor - 指令诱导器
# ============================================================================

class InstructionInductor:
    """
    指令诱导器: 通过观察高分和低分样本，自动生成评分 Rubric
    """
    
    INDUCTION_PROMPT_TEMPLATE = """You are an expert essay grading specialist. I will show you several student essays about "Laughter" (describing an experience where laughter played an important role).

These essays have been scored by human raters on a scale of 0-60.

## HIGH-SCORING ESSAYS (45-60 points):
{high_score_examples}

## LOW-SCORING ESSAYS (0-19 points):
{low_score_examples}

## YOUR TASK:
Based on the differences between high-scoring and low-scoring essays, generate a comprehensive SCORING RUBRIC that captures what makes an essay excellent vs. poor.

Your rubric should include:
1. **Key Scoring Dimensions** (e.g., narrative detail, emotional depth, structure, language use)
2. **Score Ranges and Criteria** (what characterizes essays at different score levels)
3. **Specific Indicators** (concrete features to look for)

Output your rubric in a clear, structured format that can be used as instructions for scoring new essays.

SCORING RUBRIC:"""

    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = CONFIG
        self.n_high = config['induction']['n_high_samples']
        self.n_low = config['induction']['n_low_samples']
        self.config = config
    
    def induce(self, data: List[Dict[str, Any]]) -> str:
        """从数据中诱导出评分 Rubric"""
        sorted_data = sorted(data, key=lambda x: x['domain1_score'], reverse=True)
        
        high_samples = sorted_data[:self.n_high]
        low_samples = sorted_data[-self.n_low:]
        
        high_text = self._format_samples(high_samples)
        low_text = self._format_samples(low_samples)
        
        prompt = self.INDUCTION_PROMPT_TEMPLATE.format(
            high_score_examples=high_text,
            low_score_examples=low_text
        )
        
        print("[InstructionInductor] Generating initial rubric from data...")
        temperature = self.config['llm']['temperature_generation']
        rubric = call_llm(prompt, temperature=temperature, call_type="induction")
        
        return rubric
    
    def _format_samples(self, samples: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, sample in enumerate(samples, 1):
            # 截断过长的文章
            text = sample['essay_text'][:1500] + "..." if len(sample['essay_text']) > 1500 else sample['essay_text']
            formatted.append(
                f"### Essay {i} (Score: {sample['domain1_score']}/60)\n{text}\n"
            )
        return "\n".join(formatted)


# ============================================================================
# 模块 B & C: PromptIndividual - 进化个体
# ============================================================================

@dataclass
class PromptIndividual:
    """
    进化种群中的个体，代表一个完整的评分 Prompt 配置
    """
    
    instruction_text: str
    exemplars: List[Dict[str, Any]] = field(default_factory=list)
    fitness: float = 0.0
    config: Dict[str, Any] = field(default_factory=dict)
    
    SCORING_PROMPT_TEMPLATE = """You are an expert essay grader. Score the following student essay about "Laughter" on a scale of {score_min}-{score_max}.

## SCORING RUBRIC:
{instruction}

## EXAMPLE ESSAYS FOR REFERENCE:
{exemplars}

## ESSAY TO SCORE:
{essay}

## YOUR TASK:
1. Analyze the essay based on the rubric
2. Provide a score from {score_min} to {score_max}
3. Output ONLY the numeric score, nothing else

SCORE:"""

    FEEDBACK_PROMPT_TEMPLATE = """You are an expert in essay grading analysis. I will show you cases where our automated grading system made significant errors.

## CURRENT SCORING RUBRIC:
{instruction}

## ERROR CASES (Predicted vs. Actual):
{error_cases}

## YOUR TASK:
Analyze these errors and provide specific, actionable feedback on how to improve the scoring rubric. Focus on:
1. What patterns did the model miss?
2. What criteria need to be added or clarified?
3. What specific language or indicators should be included?

Be specific and constructive. Your feedback will be used to improve the rubric.

FEEDBACK:"""

    REWRITE_PROMPT_TEMPLATE = """You are an expert prompt engineer. Your task is to improve a scoring rubric based on feedback.

## CURRENT RUBRIC:
{current_instruction}

## FEEDBACK FROM ERROR ANALYSIS:
{feedback}

## YOUR TASK:
Rewrite the scoring rubric to address the issues identified in the feedback. 
- Keep the overall structure
- Add or modify criteria as suggested
- Make the rubric more precise and actionable

OUTPUT the improved rubric directly, without any preamble:

IMPROVED RUBRIC:"""

    def __post_init__(self):
        if not self.config:
            self.config = CONFIG

    def build_scoring_prompt(self, essay_text: str) -> str:
        exemplar_text = self._format_exemplars()
        return self.SCORING_PROMPT_TEMPLATE.format(
            instruction=self.instruction_text,
            exemplars=exemplar_text,
            essay=essay_text,
            score_min=self.config['data']['score_min'],
            score_max=self.config['data']['score_max']
        )
    
    def _format_exemplars(self) -> str:
        if not self.exemplars:
            return "(No examples provided)"
        
        max_len = self.config['llm']['max_exemplar_length']
        formatted = []
        for i, ex in enumerate(self.exemplars, 1):
            text = ex['essay_text'][:max_len] + "..." if len(ex['essay_text']) > max_len else ex['essay_text']
            formatted.append(
                f"### Example {i} (Score: {ex['domain1_score']}/{self.config['data']['score_max']})\n{text}"
            )
        return "\n\n".join(formatted)
    
    def predict_score(self, essay_text: str) -> int:
        prompt = self.build_scoring_prompt(essay_text)
        temperature = self.config['llm']['temperature_scoring']
        response = call_llm(prompt, temperature=temperature, call_type="scoring")
        
        try:
            numbers = re.findall(r'\d+', response)
            if numbers:
                score = int(numbers[0])
                return max(self.config['data']['score_min'], 
                          min(self.config['data']['score_max'], score))
        except:
            pass
        
        return (self.config['data']['score_min'] + self.config['data']['score_max']) // 2
    
    def evaluate(self, val_set: List[Dict[str, Any]]) -> Tuple[float, List[int], List[int]]:
        true_scores = []
        pred_scores = []
        
        verbose = self.config.get('output', {}).get('verbose', True)
        if verbose:
            print(f"  [Evaluate] Scoring {len(val_set)} essays...")
        
        for item in val_set:
            true_scores.append(item['domain1_score'])
            pred = self.predict_score(item['essay_text'])
            pred_scores.append(pred)
        
        qwk = quadratic_weighted_kappa(true_scores, pred_scores)
        self.fitness = qwk
        
        return qwk, true_scores, pred_scores
    
    def get_feedback(self, val_set: List[Dict[str, Any]], 
                     predictions: List[int], n_errors: int = None) -> str:
        if n_errors is None:
            n_errors = self.config['evolution']['n_error_cases']
        
        errors = []
        for i, (item, pred) in enumerate(zip(val_set, predictions)):
            true_score = item['domain1_score']
            error = abs(pred - true_score)
            errors.append({
                'index': i,
                'essay': item['essay_text'],
                'true_score': true_score,
                'pred_score': pred,
                'error': error
            })
        
        errors.sort(key=lambda x: x['error'], reverse=True)
        top_errors = errors[:n_errors]
        
        error_text = []
        for e in top_errors:
            excerpt = e['essay'][:400] + "..." if len(e['essay']) > 400 else e['essay']
            error_text.append(
                f"### Case (Error: {e['error']} points)\n"
                f"- Predicted: {e['pred_score']}/{self.config['data']['score_max']}\n"
                f"- Actual: {e['true_score']}/{self.config['data']['score_max']}\n"
                f"- Essay excerpt: {excerpt}\n"
            )
        
        prompt = self.FEEDBACK_PROMPT_TEMPLATE.format(
            instruction=self.instruction_text,
            error_cases="\n".join(error_text)
        )
        
        verbose = self.config.get('output', {}).get('verbose', True)
        if verbose:
            print("  [Feedback] Analyzing errors and generating feedback...")
        
        temperature = self.config['llm']['temperature_generation']
        return call_llm(prompt, temperature=temperature, call_type="feedback")
    
    def evolve_instruction(self, feedback: str) -> str:
        prompt = self.REWRITE_PROMPT_TEMPLATE.format(
            current_instruction=self.instruction_text,
            feedback=feedback
        )
        
        verbose = self.config.get('output', {}).get('verbose', True)
        if verbose:
            print("  [Evolve] Rewriting instruction based on feedback...")
        
        temperature = self.config['llm']['temperature_generation']
        return call_llm(prompt, temperature=temperature, call_type="rewrite")
    
    def clone(self) -> 'PromptIndividual':
        return PromptIndividual(
            instruction_text=self.instruction_text,
            exemplars=self.exemplars.copy(),
            fitness=self.fitness,
            config=self.config
        )


# ============================================================================
# 模块 C: EvolutionOptimizer - 双重进化优化器
# ============================================================================

class EvolutionOptimizer:
    """双重进化优化器"""
    
    def __init__(self, train_data: List[Dict[str, Any]], val_data: List[Dict[str, Any]],
                 config: Dict[str, Any] = None):
        if config is None:
            config = CONFIG
        
        self.config = config
        self.train_data = train_data
        
        # 根据配置决定验证集大小
        val_size = config['data']['val_size']
        if val_size > 0 and val_size < len(val_data):
            self.val_data = val_data[:val_size]
        else:
            self.val_data = val_data
        
        self.population_size = config['evolution']['population_size']
        self.n_exemplars = config['evolution']['n_exemplars']
        self.elite_ratio = config['evolution']['elite_ratio']
        self.mutation_rate = config['evolution']['mutation_rate']
        self.tournament_size = config['evolution']['tournament_size']
        
        self.population: List[PromptIndividual] = []
        self.best_individual: Optional[PromptIndividual] = None
        self.history: List[Dict[str, Any]] = []
    
    def initialize_population(self, base_instruction: str):
        verbose = self.config.get('output', {}).get('verbose', True)
        if verbose:
            print(f"\n[Init] Creating population of {self.population_size} individuals...")
            print(f"[Init] Few-shot exemplars per individual: {self.n_exemplars}")
            print(f"[Init] Exemplar candidate pool size: {len(self.train_data)}")
        
        for i in range(self.population_size):
            exemplars = self._sample_diverse_exemplars()
            individual = PromptIndividual(
                instruction_text=base_instruction,
                exemplars=exemplars,
                config=self.config
            )
            self.population.append(individual)
            if verbose:
                print(f"  Individual {i+1}: {len(exemplars)} exemplars selected")
        
        # 保存初始 rubric 用于 checkpoint
        self.initial_rubric = base_instruction
    
    def restore_from_checkpoint(self, checkpoint: Dict):
        """从 checkpoint 恢复状态"""
        verbose = self.config.get('output', {}).get('verbose', True)
        
        # 创建 essay_id 到数据的映射
        train_id_map = {d['essay_id']: d for d in self.train_data}
        
        # 恢复 population
        self.population = []
        for ind_data in checkpoint['population']:
            exemplars = [train_id_map[eid] for eid in ind_data['exemplar_ids'] 
                        if eid in train_id_map]
            individual = PromptIndividual(
                instruction_text=ind_data['instruction_text'],
                exemplars=exemplars,
                fitness=ind_data['fitness'],
                config=self.config
            )
            self.population.append(individual)
        
        # 恢复 best_individual
        best_data = checkpoint['best_individual']
        best_exemplars = [train_id_map[eid] for eid in best_data['exemplar_ids']
                         if eid in train_id_map]
        self.best_individual = PromptIndividual(
            instruction_text=best_data['instruction_text'],
            exemplars=best_exemplars,
            fitness=best_data['fitness'],
            config=self.config
        )
        
        # 恢复 initial_rubric
        self.initial_rubric = checkpoint.get('initial_rubric', '')
        
        if verbose:
            print(f"\n[Resume] Restored population of {len(self.population)} individuals")
            print(f"[Resume] Best individual QWK: {self.best_individual.fitness:.4f}")
    
    def _sample_diverse_exemplars(self) -> List[Dict[str, Any]]:
        sorted_data = sorted(self.train_data, key=lambda x: x['domain1_score'])
        n = len(sorted_data)
        
        low_tier = sorted_data[:n//3]
        mid_tier = sorted_data[n//3:2*n//3]
        high_tier = sorted_data[2*n//3:]
        
        exemplars = []
        samples_per_tier = max(1, self.n_exemplars // 3)
        
        for tier in [low_tier, mid_tier, high_tier]:
            if tier:
                sampled = random.sample(tier, min(samples_per_tier, len(tier)))
                exemplars.extend(sampled)
        
        while len(exemplars) < self.n_exemplars:
            exemplars.append(random.choice(self.train_data))
        
        return exemplars[:self.n_exemplars]
    
    def tournament_selection(self) -> PromptIndividual:
        candidates = random.sample(self.population, min(self.tournament_size, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)
    
    def crossover(self, parent1: PromptIndividual, parent2: PromptIndividual) -> PromptIndividual:
        all_exemplars = parent1.exemplars + parent2.exemplars
        
        seen_ids = set()
        unique_exemplars = []
        for ex in all_exemplars:
            if ex['essay_id'] not in seen_ids:
                seen_ids.add(ex['essay_id'])
                unique_exemplars.append(ex)
        
        new_exemplars = random.sample(
            unique_exemplars, 
            min(self.n_exemplars, len(unique_exemplars))
        )
        
        better_parent = parent1 if parent1.fitness >= parent2.fitness else parent2
        
        return PromptIndividual(
            instruction_text=better_parent.instruction_text,
            exemplars=new_exemplars,
            config=self.config
        )
    
    def mutate(self, individual: PromptIndividual) -> PromptIndividual:
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = individual.clone()
        
        if mutated.exemplars:
            idx = random.randint(0, len(mutated.exemplars) - 1)
            current_ids = {ex['essay_id'] for ex in mutated.exemplars}
            candidates = [d for d in self.train_data if d['essay_id'] not in current_ids]
            
            if candidates:
                mutated.exemplars[idx] = random.choice(candidates)
        
        return mutated
    
    def evolve_elite_instructions(self, generation: int):
        n_elite = self.config['evolution']['n_elite_evolve']
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        elites = sorted_pop[:n_elite]
        
        verbose = self.config.get('output', {}).get('verbose', True)
        if verbose:
            print(f"\n[Text Evolution] Evolving instructions for top {n_elite} elites...")
        
        # 使用较小的子集进行反馈生成
        feedback_size = min(10, len(self.val_data))
        feedback_val = self.val_data[:feedback_size]
        
        for i, elite in enumerate(elites):
            if verbose:
                print(f"\n  Elite {i+1} (QWK: {elite.fitness:.4f}):")
            
            _, true_scores, pred_scores = elite.evaluate(feedback_val)
            feedback = elite.get_feedback(feedback_val, pred_scores)
            
            if verbose:
                print(f"  Feedback preview: {feedback[:200]}...")
            
            # 记录反馈
            if LOGGER:
                LOGGER.log_feedback(i+1, feedback)
            
            old_instruction = elite.instruction_text
            new_instruction = elite.evolve_instruction(feedback)
            elite.instruction_text = new_instruction
            
            # 记录指令进化
            if LOGGER:
                LOGGER.log_instruction_evolution(i+1, old_instruction, new_instruction)
            
            if verbose:
                print(f"  New instruction preview: {new_instruction[:200]}...")
    
    def evolve_one_generation(self, generation: int):
        verbose = self.config.get('output', {}).get('verbose', True)
        
        if LOGGER:
            LOGGER.log_generation_start(generation)
        elif verbose:
            print(f"\n{'='*60}")
            print(f"GENERATION {generation}")
            print(f"{'='*60}")
        
        if verbose:
            print(f"\n[Step 1] Evaluating population on {len(self.val_data)} validation samples...")
        
        eval_details = []
        for i, ind in enumerate(self.population):
            qwk, true_scores, pred_scores = ind.evaluate(self.val_data)
            if verbose:
                print(f"  Individual {i+1}: QWK = {qwk:.4f}")
            
            if LOGGER:
                eval_detail = LOGGER.log_individual_evaluation(i+1, qwk, true_scores, pred_scores)
                eval_details.append(eval_detail)
        
        current_best = max(self.population, key=lambda x: x.fitness)
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best.clone()
        
        if verbose:
            print(f"\n[Stats] Best QWK this generation: {current_best.fitness:.4f}")
            print(f"[Stats] Best QWK overall: {self.best_individual.fitness:.4f}")
        
        # 记录代总结
        if LOGGER:
            LOGGER.log_generation_summary(generation, self.population, 
                                          current_best, self.best_individual)
        
        self.history.append({
            'generation': generation,
            'best_qwk': current_best.fitness,
            'avg_qwk': np.mean([ind.fitness for ind in self.population]),
            'best_instruction_preview': current_best.instruction_text[:100]
        })
        
        if verbose:
            print("\n[Step 2] GA Evolution for Exemplars...")
        
        n_elite = max(1, int(self.population_size * self.elite_ratio))
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_population = [ind.clone() for ind in sorted_pop[:n_elite]]
        
        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_population.append(child)
        
        self.population = new_population
        
        if verbose:
            print(f"  New population created: {len(self.population)} individuals")
            print("\n[Step 3] LLM Evolution for Instructions...")
        
        self.evolve_elite_instructions(generation)
        
        # 保存 checkpoint
        if LOGGER:
            LOGGER.save_checkpoint(
                generation=generation,
                population=self.population,
                best_individual=self.best_individual,
                train_data=self.train_data,
                val_data=self.val_data,
                initial_rubric=getattr(self, 'initial_rubric', '')
            )
    
    def run(self, n_generations: int = None, start_generation: int = 1) -> PromptIndividual:
        if n_generations is None:
            n_generations = self.config['evolution']['n_generations']
        
        verbose = self.config.get('output', {}).get('verbose', True)
        
        if verbose:
            print("\n" + "="*60)
            if start_generation > 1:
                print(f"WISE-AES EVOLUTION RESUMED (from generation {start_generation})")
            else:
                print("WISE-AES EVOLUTION STARTED")
            print("="*60)
        
        for gen in range(start_generation, n_generations + 1):
            self.evolve_one_generation(gen)
        
        if verbose:
            print("\n" + "="*60)
            print("EVOLUTION COMPLETED")
            print("="*60)
        
        return self.best_individual


# ============================================================================
# 主流程
# ============================================================================

def main(config_path: str = "config.yaml", resume_dir: str = None):
    """WISE-AES 主流程
    
    Args:
        config_path: 配置文件路径
        resume_dir: 断点续传的实验目录路径 (如 result/exp_20251208_101150)
    """
    
    # 加载配置
    global CONFIG, LOGGER
    CONFIG = load_config(config_path)
    
    # 初始化实验日志记录器
    LOGGER = ExperimentLogger(result_dir="result", resume_dir=resume_dir)
    
    print("="*60)
    if resume_dir:
        print("WISE-AES: RESUMING EXPERIMENT")
    else:
        print("WISE-AES: Weakly-supervised Integrated Scoring Evolution")
    print("="*60)
    
    # 记录配置 (仅新实验)
    if not resume_dir:
        LOGGER.log_config(CONFIG)
    
    print(f"\n[Config] Loaded from {config_path}")
    print(f"  - Data path: {CONFIG['data']['asap_path']}")
    print(f"  - Essay set: {CONFIG['data']['essay_set']}")
    print(f"  - Score range: {CONFIG['data']['score_min']}-{CONFIG['data']['score_max']}")
    print(f"  - Population size: {CONFIG['evolution']['population_size']}")
    print(f"  - Generations: {CONFIG['evolution']['n_generations']}")
    print(f"  - Few-shot exemplars: {CONFIG['evolution']['n_exemplars']}")
    print(f"  - Validation size: {CONFIG['data']['val_size']}")
    
    # ========== 1. 数据准备 ==========
    print("\n[Phase 1] Data Preparation")
    print("-" * 40)
    
    all_data = load_asap_data(CONFIG)
    
    # 检查是否从 checkpoint 恢复
    checkpoint = None
    start_generation = 1
    
    if resume_dir:
        checkpoint = LOGGER.load_checkpoint()
        if checkpoint:
            start_generation = checkpoint['generation'] + 1
            print(f"\n[Resume] Found checkpoint at generation {checkpoint['generation']}")
            print(f"[Resume] Will continue from generation {start_generation}")
            
            # 使用 checkpoint 中保存的数据划分
            train_ids = set(checkpoint['train_essay_ids'])
            val_ids = set(checkpoint['val_essay_ids'])
            
            train_data = [d for d in all_data if d['essay_id'] in train_ids]
            val_data = [d for d in all_data if d['essay_id'] in val_ids]
            
            print(f"[Resume] Restored train/val split: {len(train_data)}/{len(val_data)}")
        else:
            print("[Resume] No checkpoint found, starting fresh")
            resume_dir = None  # 回退到新实验模式
    
    if not checkpoint:
        # 新实验：划分训练集和验证集
        random.shuffle(all_data)
        split_idx = int(len(all_data) * CONFIG['data']['train_ratio'])
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
    
    print(f"Training set: {len(train_data)} essays (exemplar candidate pool)")
    print(f"Validation set: {len(val_data)} essays")
    
    # 记录数据统计 (仅新实验)
    if not checkpoint:
        LOGGER.log_data_stats(train_data, val_data, all_data)
    
    # ========== 2. 指令诱导 / 恢复 ==========
    optimizer = EvolutionOptimizer(
        train_data=train_data,
        val_data=val_data,
        config=CONFIG
    )
    
    if checkpoint:
        # 从 checkpoint 恢复
        print("\n[Phase 2] Restoring from Checkpoint")
        print("-" * 40)
        optimizer.restore_from_checkpoint(checkpoint)
        initial_rubric = checkpoint.get('initial_rubric', '')
    else:
        # 新实验：指令诱导
        print("\n[Phase 2] Instruction Induction")
        print("-" * 40)
        
        inductor = InstructionInductor(CONFIG)
        initial_rubric = inductor.induce(train_data)
        
        # 记录初始 rubric
        LOGGER.log_initial_rubric(initial_rubric)
        
        print("\n[Initial Rubric Generated]")
        print("-" * 40)
        print(initial_rubric[:500] + "..." if len(initial_rubric) > 500 else initial_rubric)
        
        optimizer.initialize_population(initial_rubric)
    
    # ========== 3. 进化优化 ==========
    print("\n[Phase 3] Evolution Optimization")
    print("-" * 40)
    
    best_individual = optimizer.run(start_generation=start_generation)
    
    # ========== 4. 结果输出 ==========
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    print("\n[Evolution History]")
    print("-" * 40)
    for record in optimizer.history:
        print(f"Generation {record['generation']}: "
              f"Best QWK = {record['best_qwk']:.4f}, "
              f"Avg QWK = {record['avg_qwk']:.4f}")
    
    print("\n[Best Evolved Prompt]")
    print("-" * 40)
    print(f"Final QWK Score: {best_individual.fitness:.4f}")
    print(f"\nInstruction Text:\n{best_individual.instruction_text}")
    
    print(f"\nSelected Exemplars ({len(best_individual.exemplars)} essays):")
    for i, ex in enumerate(best_individual.exemplars, 1):
        print(f"  {i}. Essay ID {ex['essay_id']} (Score: {ex['domain1_score']}/{CONFIG['data']['score_max']})")
    
    # 保存最终结果到 result 文件夹
    LOGGER.save_final_results(best_individual, optimizer, CONFIG)
    
    return best_individual


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="WISE-AES: Automated Essay Scoring with Evolution")
    parser.add_argument("--config", "-c", type=str, default="config.yaml",
                        help="Path to config file (default: config.yaml)")
    parser.add_argument("--resume", "-r", type=str, default=None,
                        help="Resume from experiment directory (e.g., result/exp_20251208_101150)")
    
    args = parser.parse_args()
    
    main(config_path=args.config, resume_dir=args.resume)
