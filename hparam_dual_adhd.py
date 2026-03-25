import os
import json
import time
import random
import argparse
import subprocess
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Dict, Any, List

# ============================================
# 配置区域 - 直接修改这里的参数
# ============================================
CONFIG = {
    # 核心路径配置
    "model_py": r"/root/autodl-tmp/model_fusion_multi_gnn_commu.py",  # model.py 的路径
    "pt_dir": r"/root/autodl-tmp/dataset/adhd_5_0.5/adhd_5_0.5",  # 第一个通道训练数据目录
    "pt_dir2": r"/root/autodl-tmp/dataset/adhd_global",  # 第二个通道训练数据目录 (None=单通道, 填入路径=双通道)
                       # 示例: r"E:\webnet\dataset\adhd_5_0.5_channel2\adhd_5_0.5"
    "log_root": r"/root/autodl-tmp/rebuttal/44/adhd",  # 日志输出目录
    
    # 搜索配置
    "n_trials": 1000,  # 搜索次数
    "epochs": 1000,     # 每次训练的轮数
    "py_exec": "python",  # Python 解释器路径
    
    # 模型训练参数（可选）
    "no_oversample": False,  # True: 使用平衡采样器, False: 使用加权采样
    "init_bias_target": 0.75,  # 分类器偏置初始化 (0-1之间，None表示自动)
}
# ============================================

def generate_hparams() -> Dict[str, Any]:
    """
    改进的超参数生成：包含GAT、动态loss等关键参数
    """
    # ========== 关键参数 ==========
    lr = 10**random.uniform(-5.0, -3.0)
    
    # ========== Local通道（Transformer）==========
    win_emb = random.choice([64, 96, 128, 160, 192, 256])
    tf_layers = random.choice([2, 3])
    
    # ========== Global通道（GNN）- GAT参数 ==========
    # gnn_type = random.choice(['gat', 'gcn'])
    gnn_type = 'gcn'  # 🔧 固定为GCN
    gnn_hidden = random.choice([64, 96, 128, 160])
    gnn_layers = random.choice([2, 3])
    gat_heads = random.choice([2, 4, 8]) if gnn_type == 'gat' else 4

    # ========== OCREAD池化参数 ==========
    ocread_clusters = random.choice([4, 8, 16])

    # ========== 门控融合参数 ==========
    fusion_hidden = random.choice([64, 128, 256])
    
    # ========== 动态Loss参数（必须启用）==========
    # 🔧 改为100%启用（因为已经删掉fusion了）
    use_dynamic_loss = True  # 不再随机，固定启用
    warmup_epochs = random.choice([15, 20, 25])
    learning_epochs = random.choice([25, 30, 35])
    min_weight = random.choice([0.20, 0.25, 0.30])
    balance_penalty = round(random.uniform(0.5, 2.0), 2)
    
    # ========== 通用 ==========
    dropout = round(random.uniform(0.2, 0.4), 2)
    
    # 确保 nhead 是 win_emb 的因子
    possible_heads = [h for h in [2, 4, 8] if win_emb % h == 0]
    if not possible_heads:
        possible_heads = [4]
    
    params = {
        'lr': lr,
        'weight_decay': 10**random.uniform(-6.0, -4.0),
        'batch_size': 32,
        'win_emb': win_emb,
        'tf_layers': tf_layers,
        'tf_heads': random.choice(possible_heads),
        
        # GAT参数
        'gnn_type': gnn_type,
        'gnn_hidden': gnn_hidden,
        'gnn_layers': gnn_layers,
        'gat_heads': gat_heads,

        # 🆕 OCREAD和融合参数
        # 'ocread_clusters': ocread_clusters,
        # 'fusion_hidden': fusion_hidden,
        # 'use_node_memory': True,
        
        # 动态Loss参数（固定启用）
        'use_dynamic_loss': use_dynamic_loss,
        'warmup_epochs': warmup_epochs,
        'learning_epochs': learning_epochs,
        'min_weight': min_weight,
        'balance_penalty': balance_penalty,
        
        'dropout': dropout,
        'seed': 44,
        # 'seed': random.choice([42, 123, 456, 789, 2024]) if random.random() < 0.5 else int(random.randint(0, 2**31-1)),
        # 'seed': 1247725342,
    }
    return params

def hparams_to_cmd_args(hparams: Dict[str, Any]) -> List[str]:
    """
    将超参数字典转换为命令行参数列表。
    布尔值会被转换为flag（--flag），其他类型转换为 --key value
    """
    args = []
    for k, v in hparams.items():
        if isinstance(v, bool):
            # 🔧 布尔值：只有True时才添加flag
            if v:
                args.append(f"--{k}")
        else:
            # 其他类型：正常添加 --key value
            args.append(f"--{k}")
            args.append(str(v))
    return args

def dict_to_cli_args(d: Dict[str, Any]) -> List[str]:
    """
    将 base_args 字典转换为 CLI 参数列表。
    布尔且为 True 的项会被转换为开关（--flag）。
    None 值会被忽略。
    其它类型会被转换为 --key value。
    """
    out = []
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        else:
            out.append(f"--{k}")
            out.append(str(v))
    return out

# 在 run_trials() 函数之前添加这个函数
def stratified_split(dataset, train_ratio=0.6, val_ratio=0.2, seed=42):
    """
    使用分层抽样划分数据集，确保train/val/test的类别分布一致
    """
    from sklearn.model_selection import train_test_split
    
    # 获取所有标签
    n = len(dataset)
    labels = []
    for i in range(n):
        sample = dataset[i]
        labels.append(int(sample['label'].item()))
    
    labels = np.array(labels)
    indices = np.arange(n)
    
    # 第一次划分：train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=1.0 - train_ratio - val_ratio,
        stratify=labels,
        random_state=seed
    )
    
    # 第二次划分：train vs val
    train_val_labels = labels[train_val_idx]
    val_size = val_ratio / (train_ratio + val_ratio)
    
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=train_val_labels,
        random_state=seed
    )
    
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

def run_trials(py_exec: str, model_py: Path, pt_dir: str, pt_dir2: str, n_trials: int, epochs: int, base_args: dict, log_root: str):
    os.makedirs(log_root, exist_ok=True)
    master_path = Path(log_root) / "master.log"
    with open(master_path, "a", encoding="utf-8", errors="replace") as master_f:
        for tid in range(1, n_trials + 1):
            # 1. 生成新的Transformer超参数
            hparams = generate_hparams()
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trial_name = f"trial_{tid:04d}_{timestamp}"
            trial_dir = Path(log_root) / trial_name
            trial_dir.mkdir(parents=True, exist_ok=True)

            # 2. 保存配置
            full_config = {**base_args, **hparams, 'epochs': epochs, 'trial_id': tid}
            with open(trial_dir / 'config.json', 'w', encoding='utf-8') as f:
                json.dump(full_config, f, indent=2, ensure_ascii=False)

            # 3. 动态构建命令行
            cmd_hparams = hparams_to_cmd_args(hparams)

            # build base args CLI list from base_args dict (so flags like --no_oversample and --init_bias_target are forwarded)
            base_cli = dict_to_cli_args(base_args)

            cmd = [
                py_exec, str(model_py),
                *base_cli,
                "--epochs", str(epochs),
                "--save_model", str(trial_dir / "best.pth"),
                *cmd_hparams
            ]

            log_path = trial_dir / "run.log"
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            # Ensure child processes default to CPU-only to avoid torch/CUDA init issues.
            env.setdefault("HPARAM_FORCE_CPU", "1")
            print(f"[{tid}/{n_trials}] Starting trial -> {trial_dir.name}")
            master_f.write(f"\n--- START TRIAL {tid} {trial_dir.name} {timestamp} ---\n")
            master_f.flush()
            start = time.time()
            
            # FIX: Open the trial-specific log file here
            with open(log_path, "w", encoding="utf-8", errors='replace') as trial_f:
                # 运行子进程
                with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                                      universal_newlines=True, env=env, encoding='utf-8', errors='replace') as proc:
                    for line in proc.stdout:
                        pref = f"[T{tid:04d}] "
                        out_line = pref + line
                        # Now trial_f is defined and can be written to
                        trial_f.write(line)
                        trial_f.flush()
                        master_f.write(out_line)
                        master_f.flush()
                        print(out_line, end="")
                    proc.wait()
                    rc = proc.returncode
                
            elapsed = time.time() - start
            meta = {
                "returncode": rc,
                "elapsed_seconds": elapsed,
                "logfile": str(log_path.name),
                "model_path": "best.pth"
            }
            with open(trial_dir / 'meta.json', 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            master_f.write(f"--- END TRIAL {tid} rc={rc} elapsed={elapsed:.1f}s ---\n")
            master_f.flush()
            print(f"[{tid}/{n_trials}] Finished (rc={rc}) elapsed={elapsed:.1f}s -> {trial_dir}")

if __name__ == "__main__":
    # 打印当前配置
    print("=" * 60)
    print("超参数搜索配置:")
    print("=" * 60)
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60)
    print()
    
    # 从 CONFIG 字典读取参数
    pt_dir = CONFIG["pt_dir"]
    pt_dir2 = CONFIG.get("pt_dir2", None)
    n_trials = CONFIG["n_trials"]
    epochs = CONFIG["epochs"]
    log_root = CONFIG["log_root"]
    py_exec = CONFIG["py_exec"]
    model_py = Path(CONFIG["model_py"])
    
    # 构建 base_args
    base_args = {"pt_dir": pt_dir}
    
    # 如果提供了pt_dir2，则添加到base_args
    if pt_dir2:
        base_args["pt_dir2"] = pt_dir2
        print(f"双通道模式: channel 2 = {pt_dir2}")
    
    # forward flags only when provided / meaningful
    if CONFIG.get("no_oversample", False):
        base_args["no_oversample"] = True
    if CONFIG.get("init_bias_target") is not None:
        base_args["init_bias_target"] = CONFIG["init_bias_target"]
    # 开始搜索
    print(f"开始运行 {n_trials} 次超参数搜索...")
    print(f"日志将保存到: {log_root}\n")
    
    run_trials(py_exec, model_py, pt_dir, pt_dir2, n_trials, epochs, base_args, log_root)
    
    print("\n" + "=" * 60)
    print("搜索完成！")
    print(f"查看主日志: {Path(log_root) / 'master.log'}")
    print("=" * 60)
