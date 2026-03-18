import os
import json
import argparse
import numpy as np
import itertools
import sys
import torch
import wandb
from main import main as run_training

SOTA_JSON_PATH = "../Data/sota.json"
DATA_ROOT = "../Data"

SEARCH_SPACE = {
    "lr": [1e-3, 5e-4, 1e-4],
    "hidden_dim": [256, 512],
    "latent_dim": [256,512,1024],
    "batch_size": [64,128]
}

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a") # 改为追加模式，防止 fold 循环覆盖
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()

def load_sota_values(dataset_name):
    with open(SOTA_JSON_PATH, 'r', encoding='utf-8') as f:
        content = json.load(f)
    d = content['data'][dataset_name]
    return [d['Cheby']['mean'], d['Clark']['mean'], d['Canbe']['mean'],
            d['KL']['mean'], d['Cosine']['mean'], d['Inter']['mean']]

def get_dims(dataset_name):
    # 定义可能的两个根目录
    feature_root = os.path.join(DATA_ROOT, "feature", dataset_name, 'run_0')
    image_root = os.path.join(DATA_ROOT, "image", dataset_name, 'run_0')
    
    # 自动探测路径
    if os.path.exists(feature_root):
        target_dir = feature_root
    elif os.path.exists(image_root):
        target_dir = image_root
    else:
        raise FileNotFoundError(f"❌ 找不到数据集 {dataset_name} 的数据。请检查 {feature_root} 或 {image_root} 是否存在。")
    
    print(f"📂 自动识别数据路径: {target_dir}")
    
    feat_path = os.path.join(target_dir, 'train_feature.npy')
    label_path = os.path.join(target_dir, 'train_label.npy')
    
    # 使用 mmap_mode 读取以节省内存，仅获取维度
    feat = np.load(feat_path, mmap_mode='r')
    label = np.load(label_path, mmap_mode='r')
    
    return feat.shape[1], label.shape[1]

def run_experiment(dataset, device):
    base_log_dir = f"../Logs/SFNO/{dataset}"
    os.makedirs(base_log_dir, exist_ok=True)
    
    # 建立主控日志
    master_log_path = os.path.join(base_log_dir, "overall_process.log")
    master_logger = Logger(master_log_path)
    sys.stdout = master_logger
    
    print(f"======== 🚀 Start Task: {dataset} ========")
    sota_vals = load_sota_values(dataset)
    data_dim, output_dim = get_dims(dataset)
    sota_str = str(sota_vals).replace(" ", "")

    # === Phase 1: Grid Search ===
    best_avg_imp = -float('inf')
    best_params = None
    keys, values = zip(*SEARCH_SPACE.items())
    
    for p_idx, params_tuple in enumerate(itertools.product(*values)):
        params = dict(zip(keys, params_tuple))
        param_dir = os.path.join(base_log_dir, "search", f"params_{p_idx}")
        os.makedirs(param_dir, exist_ok=True)
        
        # 重定向到该组参数的独立日志
        param_logger = Logger(os.path.join(param_dir, "param_search.log"))
        sys.stdout = param_logger
        
        cv_scores = []
        for fold in range(5):
            print(f"\n--- Fold {fold} ---")
            args = [
                "fit", "--config", "configs/ldl.yaml",
                f"--name=Search_F{fold}",
                "--data.dataset", dataset,
                "--data.fold_idx", str(fold),
                "--data.batch_size", str(params['batch_size']),
                f"--model.init_args.data_dim={data_dim}",
                f"--model.init_args.output_dim={output_dim}",
                f"--model.init_args.hidden_dim={params['hidden_dim']}",
                f"--model.init_args.latent_dim={params['latent_dim']}",
                f"--model.init_args.lr={params['lr']}",
                f"--model.init_args.sota_values={sota_str}",
                f"--trainer.default_root_dir={param_dir}", # 关键：存入 search/params_n
                "--trainer.accelerator", "gpu",
                f"--trainer.devices=[{device}]",
                "--trainer.max_steps", "6000",
            ]
            res_dict = run_training(args)  # 接收返回的字典
            imp = res_dict['avg_imp']      # 提取用于比较和打印的浮点数分数
            if wandb.run: wandb.finish()
            cv_scores.append(imp)
            print(f"Fold {fold} Imp: {imp:.4f}")  # 此时 imp 是数字，不再报错
            
        avg_score = np.mean(cv_scores)
        sys.stdout = master_logger # 切回主日志汇报进度
        print(f"👉 Params {params} Avg Score: {avg_score:.4f}")
        
        if avg_score > best_avg_imp:
            best_avg_imp = avg_score
            best_params = params

    # === Phase 2: 10-Fold Evaluation ===
    print(f"\n🏆 Final Eval with: {best_params}")
    detailed_results = []
    
    for run_idx in range(10):
        run_dir = os.path.join(base_log_dir, "formal", f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)
        
        sys.stdout = Logger(os.path.join(run_dir, "run.log"))
        
        args = [
            "fit", "--config", "configs/ldl.yaml",
            f"--name=Eval_R{run_idx}",
            "--data.dataset", dataset,
            "--data.split_num", str(run_idx),
            "--data.fold_idx", "-1",
            "--data.batch_size", str(best_params['batch_size']),
            f"--model.init_args.data_dim={data_dim}",
            f"--model.init_args.output_dim={output_dim}",
            f"--model.init_args.hidden_dim={best_params['hidden_dim']}",
            f"--model.init_args.latent_dim={best_params['latent_dim']}",
            f"--model.init_args.lr={best_params['lr']}",
            f"--model.init_args.sota_values={sota_str}",
            f"--trainer.default_root_dir={run_dir}", # 关键：存入 formal/run_n
            "--trainer.accelerator", "gpu",
            f"--trainer.devices=[{device}]",
            "--trainer.max_steps", "100000",
        ]
        print(f"--> Full Run {run_idx}/10 ... ", end="")
        sys.stdout.flush()
        # 这里的 run_training 会执行 validation_epoch_end，打印出具体指标
        res_dict = run_training(args) 
        detailed_results.append(res_dict)
        if wandb.run: wandb.finish()
        print(f"Done. Imp: {res_dict['avg_imp']:.4f}")
        sys.stdout = master_logger

    # === 写入最终 result.txt (包含所有 split 的数据) ===
    with open(os.path.join(base_log_dir, "result.txt"), "w") as f:
        f.write(f"Dataset: {dataset}\n")
        f.write(f"Best Params: {json.dumps(best_params)}\n\n")
        f.write(f"Best Search AvgImp: {best_avg_imp:.4f}\n")
        metrics_keys = ['avg_imp', 'Cheby', 'Clark', 'Canbe', 'KL', 'Cosine', 'Inter']
        
        # 1. 输出每个 split 的详细数据表格
        header = "Run_Idx".ljust(10) + "".join([k.ljust(12) for k in metrics_keys])
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        for i, res in enumerate(detailed_results):
            row = str(i).ljust(10) + "".join([f"{res.get(k, 0):.4f}".ljust(12) for k in metrics_keys])
            f.write(row + "\n")
        
        f.write("-" * len(header) + "\n\n")
        
        # 2. 计算并输出 Mean ± Std
        f.write("📊 Final Statistics (Mean ± Std):\n")
        for k in metrics_keys:
            vals = [res.get(k, 0) for res in detailed_results]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            f.write(f"{k.ljust(10)}: {mean_val:.4f} ± {std_val:.4f}\n")

    print(f"✅ {dataset} All results saved to result.txt")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    sys.argv = [sys.argv[0]]
    run_experiment(args.dataset, args.device)
