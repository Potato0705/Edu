import os
import shutil
import yaml
import re
from pathlib import Path

SOURCE_DIR = Path("full_logs")
DEST_DIR = Path("full_logs_classified")

def load_config(exp_dir_path):
    config_path = exp_dir_path / "config.yaml"
    if not config_path.exists():
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        return None

def generate_group_name(config):
    # Extract key features
    rag = config.get('rag', {})
    evo = config.get('evolution', {})
    
    rag_enabled = rag.get('enabled', False)
    rag_mode = "on" if rag_enabled else "off"
    
    # ICL usually means static exemplars? Or is it a specific flag?
    # Looking at user's filenames: "icl_off", "rag_off", "instruction_on", "exemplar_off"
    # "instruction_on" -> evolve_instruction?
    # "exemplar_off" -> evolve_static_exemplars OR n_static_exemplars=0?
    
    n_static = evo.get('n_static_exemplars', 0)
    evolve_instr = evo.get('evolve_instruction', False)
    evolve_exem = evo.get('evolve_static_exemplars', False)
    
    # Heuristic mapping based on observed naming conventions
    instruction_mode = "instruction_on" if evolve_instr else "instruction_off"
    
    # "exemplar_off" likely means NO static exemplars or NO evolution of them?
    # If n_static > 0 and evolve=True -> exemplar_evolve
    # If n_static > 0 and evolve=False -> exemplar_static
    # If n_static == 0 -> exemplar_off
    if n_static == 0:
        exemplar_mode = "exemplar_off"
    elif evolve_exem:
        exemplar_mode = "exemplar_evolve"
    else:
        exemplar_mode = "exemplar_static"

    # ICL? Maybe it refers to In-Context Learning (few-shot)? 
    # If n_static > 0 it is ICL.
    # But user used "icl_off" AND "exemplar_off".
    # Let's inspect config for an explicit 'icl' section or assume 'icl_off' if n_static=0.
    # Wait, user's previous folder: `discard_icl_off_rag_off_instruction_on_exemplar_off`
    # This implies ICL and Exemplar are distinct concepts in their naming, or redundant.
    # Let's check if there is an 'icl' key.
    # If not, I will construct a name based on what I have.
    
    parts = []
    
    # RAG
    parts.append(f"rag_{rag_mode}")
    
    # ICL/Exemplar complex
    if n_static == 0:
        parts.append("icl_off") # No shots
    else:
        parts.append(f"icl_{n_static}shot")
        
    parts.append(instruction_mode)
    parts.append(exemplar_mode)
    
    # Maybe Model Name?
    model_name = config.get('model', {}).get('name', 'unknown').split('/')[-1]
    # Simplify model name (e.g. Llama-3-8B -> llama3)
    if 'llama' in model_name.lower():
        parts.append('llama')
    elif 'gpt' in model_name.lower():
        parts.append('gpt')
        
    return "_".join(parts)

def get_group_for_md(md_path):
    # Read MD to find first exp dir
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'`(exp_[\w\d_]+)`', content)
            if match:
                exp_dir_name = match.group(1)
                exp_dir_path = SOURCE_DIR / exp_dir_name
                # It might have been moved already? 
                # Better strategy: Do dirs first, map dir_name -> group_name.
                return exp_dir_name
    except:
        pass
    return None

def main():
    if not SOURCE_DIR.exists():
        print(f"Source {SOURCE_DIR} does not exist.")
        return
        
    DEST_DIR.mkdir(exist_ok=True)
    
    all_items = sorted(os.listdir(SOURCE_DIR))
    exp_dirs = [i for i in all_items if i.startswith("exp_") and (SOURCE_DIR / i).is_dir()]
    md_files = [i for i in all_items if i.startswith("5fold_") and i.endswith(".md")]
    
    print(f"Found {len(exp_dirs)} experiment directories and {len(md_files)} MD files.")
    
    # 1. Map exp_dir -> Group Name
    dir_group_map = {}
    
    print("Processing directories...")
    for d in exp_dirs:
        config = load_config(SOURCE_DIR / d)
        if config:
            group_name = generate_group_name(config)
        else:
            group_name = "uncategorized"
            
        dir_group_map[d] = group_name
        
        # Create group dir
        target_group_dir = DEST_DIR / group_name
        target_group_dir.mkdir(exist_ok=True)
        
        # Move
        src = SOURCE_DIR / d
        dst = target_group_dir / d
        if not dst.exists():
            shutil.move(str(src), str(dst))
        else:
            print(f"  [Skip] {d} already in {group_name}")

    print("Processing MD files...")
    for m in md_files:
        exp_name = get_group_for_md(SOURCE_DIR / m)
        if exp_name and exp_name in dir_group_map:
            group_name = dir_group_map[exp_name]
        else:
            # If we referred to an exp_name that is not in the current iterate list (maybe moved previously?), 
            # we can't easily track it unless we scan DEST_DIR too. 
            # But here we assume we are running on a fresh full_logs or at least consistent one.
            # Fallback: try to find where that exp_name went
            group_name = "uncategorized"
            # Optimization: could search in dir_group_map
            
        target_group_dir = DEST_DIR / group_name
        target_group_dir.mkdir(exist_ok=True)
        
        src = SOURCE_DIR / m
        dst = target_group_dir / m
        if not dst.exists():
            shutil.move(str(src), str(dst))
            
    print("Done classification.")

if __name__ == "__main__":
    main()
