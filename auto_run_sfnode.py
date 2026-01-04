import os
import json
import argparse
import numpy as np
import itertools
import sys
import torch
import wandb
# 导入 main 函数
from main import main as run_training

SOTA_JSON_PATH = "../Data/sota.json"
DATA_ROOT = "../Data"

# 搜索空间 (完全解耦，对齐 Classification 逻辑)
SEARCH_SPACE = {
    "lr": [1e-3, 5e-4],
    "hidden_dim": [256],  # ODE 网络内部宽度
    "latent_dim": [256],  # 特征投影空间维度 (Feature Space)
    "batch_size": [64]
}

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        pass

def load_sota_values(dataset_name):
    if not os.path.exists(SOTA_JSON_PATH):
        raise FileNotFoundError(f"❌ SOTA file not found: {SOTA_JSON_PATH}")
    with open(SOTA_JSON_PATH, 'r', encoding='utf-8') as f:
        content = json.load(f)
    if dataset_name not in content.get('data', {}):
        raise ValueError(f"❌ Dataset {dataset_name} not found in sota.json")
    d = content['data'][dataset_name]
    return [d['Cheby']['mean'], d['Clark']['mean'], d['Canbe']['mean'],
            d['KL']['mean'], d['Cosine']['mean'], d['Inter']['mean']]

def get_dims(dataset_name):
    feat_run_dir = os.path.join(DATA_ROOT, "feature", dataset_name, 'run_0')
    img_run_dir = os.path.join(DATA_ROOT, "image", dataset_name, 'run_0')
    target_dir = feat_run_dir if os.path.exists(feat_run_dir) else img_run_dir
    
    if not os.path.exists(target_dir):
        raise FileNotFoundError(f"❌ Data not found for {dataset_name}")

    feat_path = os.path.join(target_dir, 'train_feature.npy')
    label_path = os.path.join(target_dir, 'train_label.npy')
    
    if not os.path.exists(feat_path):
        raise FileNotFoundError(f"❌ Feature file missing: {feat_path}")

    feat = np.load(feat_path, mmap_mode='r')
    label = np.load(label_path, mmap_mode='r')
    return feat.shape[1], label.shape[1]

def run_experiment(dataset, device):
    # 1. 准备搜索日志
    log_dir = f"../Logs/SFNO/{dataset}"
    os.makedirs(log_dir, exist_ok=True)
    sys.stdout = Logger(os.path.join(log_dir, "search_log.txt"))
    
    print(f"======== 🚀 Start Task: {dataset} ========")
    print(f"Device: cuda:{device}")
    
    sota_vals = load_sota_values(dataset)
    data_dim, output_dim = get_dims(dataset)
    sota_str = str(sota_vals).replace(" ", "")
    
    print(f"Input Dim (Data): {data_dim}")
    print(f"Output Dim (Label): {output_dim}")
    print(f"SOTA Baseline: {sota_vals}")

    # === Phase 1: Internal 5-Fold Grid Search (No Leakage) ===
    print(f"\n🔍 [Phase 1] Internal 5-Fold CV on Run_0 Train Set...")
    
    best_avg_imp = -float('inf')
    best_params = None
    
    keys, values = zip(*SEARCH_SPACE.items())
    
    for params_tuple in itertools.product(*values):
        params = dict(zip(keys, params_tuple))
        print(f"\n👉 Testing Params: {params}")
        
        cv_scores = []
        for fold in range(5): 
            # 实验名：把 latent 和 hidden 都写进去
            exp_name = f"Search_{dataset}_fold{fold}_lr{params['lr']}_h{params['hidden_dim']}_lat{params['latent_dim']}"
            
            args = [
                "fit",
                "--config", "configs/ldl.yaml",
                f"--name={exp_name}",
                "--data.dataset", dataset,
                "--data.split_num", "0",
                "--data.fold_idx", str(fold),
                "--data.batch_size", str(params['batch_size']),
                "--data.test_batch_size", str(params['batch_size']),
                
                # --- 维度设置：完全解耦 ---
                "--model.init_args.data_dim", str(data_dim),
                "--model.init_args.output_dim", str(output_dim),
                f"--model.init_args.hidden_dim={params['hidden_dim']}", 
                f"--model.init_args.latent_dim={params['latent_dim']}",
                
                f"--model.init_args.lr={params['lr']}",
                f"--model.init_args.sota_values={sota_str}",
                "--trainer.accelerator", "gpu",
                f"--trainer.devices=[{device}]",
                "--trainer.max_steps", "10000",
                "--trainer.enable_checkpointing", "true",
            ]
            
            print(f"   [Fold {fold}] ... ", end="")
            sys.stdout.flush()
            imp = run_training(args)
            if wandb.run is not None:
                wandb.finish()
            cv_scores.append(imp)
            print(f"Imp: {imp:.4f}")
        
        avg_score = np.mean(cv_scores)
        print(f"   >>> Param CV Avg Score: {avg_score:.4f}")
        
        if avg_score > best_avg_imp:
            best_avg_imp = avg_score
            best_params = params

    print(f"\n🏆 Best Params Found: {best_params} (Avg 5-Fold Imp: {best_avg_imp:.4f})")
    
    # === Phase 2: 10-Fold Full Evaluation ===
    print(f"\n🏃 [Phase 2] Running 10-Fold Full Evaluation...")
    results = []
    
    for run_idx in range(10):
        # 评估阶段使用最佳参数
        exp_name = f"Eval_{dataset}_run{run_idx}_best"
        args = [
            "fit",
            "--config", "configs/ldl.yaml",
            f"--name={exp_name}",
            "--data.dataset", dataset,
            "--data.split_num", str(run_idx),
            "--data.fold_idx", "-1",          # 正常模式
            "--data.batch_size", str(best_params['batch_size']),
            "--data.test_batch_size", str(best_params['batch_size']),
            
            "--model.init_args.data_dim", str(data_dim),
            "--model.init_args.output_dim", str(output_dim),
            f"--model.init_args.hidden_dim={best_params['hidden_dim']}",
            f"--model.init_args.latent_dim={best_params['latent_dim']}",
            
            f"--model.init_args.lr={best_params['lr']}",
            f"--model.init_args.sota_values={sota_str}",
            "--trainer.accelerator", "gpu",
            f"--trainer.devices=[{device}]",
            "--trainer.max_steps", "100000",
            "--trainer.enable_checkpointing", "true",
        ]
        
        print(f"--> Full Run {run_idx}/10 ... ", end="")
        sys.stdout.flush()
        imp = run_training(args)
        results.append(imp)
        print(f"Done. Imp: {imp:.4f}")

    if results:
        mean_imp = np.mean(results)
        std_imp = np.std(results)
        print(f"\n✅ {dataset} Final 10-Fold AvgImp: {mean_imp:.4f} ± {std_imp:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    run_experiment(args.dataset, args.device)