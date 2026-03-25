import os
import glob
import random
import math

import torch
import torch.nn as nn
import numpy as np
import argparse
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, Sampler, Subset
import time
import warnings
from collections import deque

# ============================================================
# 双通道模型：Local (Transformer) + Global (GNN)
# 
# Local 通道: (B, T, N, N) - 有时序，用 WindowEncoder + Transformer
# Global 通道: (B, N, N) - 无时序，用 GNN
# ============================================================
# -----------------------------
# Data utilities
# -----------------------------
class PTGraphSequenceDataset(Dataset):
    """
    双通道数据集：
    - 通道1 (Local): 有时序 (T, N, N)
    - 通道2 (Global): 无时序 (N, N)
    """
    def __init__(self, pt_dir: str, labels_map: Dict[str, int] = None, pt_dir2: str = None):
        self.pt_dir = pt_dir
        self.pt_dir2 = pt_dir2
        self.dual_channel = pt_dir2 is not None
        
        if self.dual_channel:
            print(f"Initializing dataset in DUAL-CHANNEL mode")
            print(f"  Channel 1 (Local): {pt_dir} - with temporal dimension")
            print(f"  Channel 2 (Global): {pt_dir2} - single graph (no temporal)")
            
        all_pts = sorted(glob.glob(os.path.join(pt_dir, '**', '*.pt'), recursive=True))
        if len(all_pts) == 0:
            raise RuntimeError(f"No .pt files found in {pt_dir}")
        
        if self.dual_channel:
            all_pts2 = sorted(glob.glob(os.path.join(pt_dir2, '**', '*.pt'), recursive=True))
            if len(all_pts2) == 0:
                raise RuntimeError(f"No .pt files found in second channel directory {pt_dir2}")
            self.pt_dir2_map = {}
            for p2 in all_pts2:
                rel_path = os.path.relpath(p2, pt_dir2)
                self.pt_dir2_map[rel_path] = p2
            print(f"  Found {len(all_pts2)} files in channel 2")
            
        provided_map = labels_map or {}
        files = []
        files2 = []
        inferred_map = {}
        
        for p in all_pts:
            fname = os.path.basename(p)
            rel_path = os.path.relpath(p, pt_dir)
            
            if self.dual_channel:
                if rel_path not in self.pt_dir2_map:
                    print(f"Warning: skipping {p} - no matching file in channel 2")
                    continue
                p2 = self.pt_dir2_map[rel_path]
            else:
                p2 = None
                
            if fname in provided_map:
                inferred_map[fname] = int(provided_map[fname])
                files.append(p)
                if self.dual_channel:
                    files2.append(p2)
                continue
                
            parts = os.path.normpath(p).split(os.sep)
            label = None
            for part in reversed(parts[:-1]):
                if part in ('0', '1'):
                    label = int(part)
                    break
            if label is not None:
                inferred_map[fname] = label
                files.append(p)
                if self.dual_channel:
                    files2.append(p2)
            else:
                print(f"Warning: skipping {p} (no ancestor folder named '0' or '1' and not in labels_map)")
                
        if len(files) == 0:
            raise RuntimeError(f"No labeled .pt files found under {pt_dir}")
            
        self.files = files
        self.files2 = files2 if self.dual_channel else None
        self.labels_map = {**inferred_map, **(provided_map or {})}
        self._compute_max_N_and_validate()
        self._compute_max_community_id()  # 🆕 添加这一行

    def _compute_max_N_and_validate(self):
        maxN = 0
        orig = {}
        good_files = []
        good_files2 = []
        
        for i, p in enumerate(self.files):
            p2 = self.files2[i] if self.dual_channel else None
            try:
                d = torch.load(p, map_location='cpu')
                adj = d['adjacency']
                # Local: (T, N, N) 取第二个维度
                n = int(adj.shape[1]) if adj.dim() == 3 else int(adj.shape[0])
                
                if self.dual_channel:
                    d2 = torch.load(p2, map_location='cpu')
                    adj2 = d2['adjacency']
                    # Global: (N, N) 或 (1, N, N)
                    n2 = int(adj2.shape[0]) if adj2.dim() == 2 else int(adj2.shape[1])
                    if n != n2:
                        n = max(n, n2)
            except Exception as e:
                print(f"Warning: unable to read {p}{' or ' + p2 if self.dual_channel else ''}: {e}")
                continue
                
            orig[p] = n
            maxN = max(maxN, n)
            good_files.append(p)
            if self.dual_channel:
                good_files2.append(p2)
                
        if len(good_files) == 0:
            raise RuntimeError("No readable .pt files found when validating node counts.")
            
        removed = [p for p in self.files if p not in good_files]
        if len(removed) > 0:
            print(f"Warning: {len(removed)} files unreadable and skipped.")
        self.files = sorted(good_files)
        if self.dual_channel:
            self.files2 = sorted(good_files2)
        self.orig_N = orig
        self.N = int(maxN)
        print(f"Dataset node-count range: min={min(orig.values())}, max={self.N}; padding smaller graphs to N={self.N}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        
        # 加载 Local 通道 (有时序)
        data1 = torch.load(p)
        adj1 = data1['adjacency'].float()  # (T, N, N)
        comm1 = data1.get('community_sequence', None)
        if comm1 is not None:
            comm1 = comm1.long()
        coords1 = data1.get('coords', None)
        if coords1 is not None:
            coords1 = coords1.float()
        
        adj1, comm1, coords1 = self._pad_local_channel(adj1, comm1, coords1)
        
        # 加载 Global 通道 (无时序)
        if self.dual_channel:
            p2 = self.files2[idx]
            data2 = torch.load(p2)
            adj2 = data2['adjacency'].float()  # (N, N) 无时序！
            
            # 确保是2D
            if adj2.dim() == 3:
                adj2 = adj2[0]
            
            coords2 = data2.get('coords', None)
            if coords2 is not None:
                coords2 = coords2.float()
            
            # ← 新增：加载社区标签
            comm2 = data2.get('community_labels', None)
            if comm2 is not None:
                comm2 = comm2.long()
            
            adj2, coords2, comm2 = self._pad_global_channel(adj2, coords2, comm2)
        else:
            adj2, coords2, comm2 = None, None, None        

        # 推断标签
        fname = os.path.basename(p)
        if fname in self.labels_map:
            label = int(self.labels_map[fname])
        else:
            if '_label0' in fname or fname.endswith('_0.pt'):
                label = 0
            elif '_label1' in fname or fname.endswith('_1.pt'):
                label = 1
            else:
                raise RuntimeError(f"No label found for file {fname}")
        
        result = {
            'adj1': adj1,           # (T, N, N) Local with temporal
            'comm1': comm1,         # (T, N)
            'coords1': coords1,     # (N, 3)
            'len1': adj1.shape[0],  # T
            'label': torch.tensor(label, dtype=torch.float32),
            'file': p
        }
        
        if self.dual_channel:
            result['adj2'] = adj2       # (N, N) Global without temporal
            result['coords2'] = coords2 # (N, 3)
            result['comm2'] = comm2     # (N,) ← 新增        
        return result
    
    def _pad_local_channel(self, adj, comm, coords):
        """填充 Local 通道数据 (有时序)"""
        # adj: (T, n, n)
        n_adj = adj.shape[1]
        
        if coords is not None:
            n_coords = coords.shape[0]
            if n_coords > n_adj:
                coords = coords[:n_adj]
            elif n_coords < n_adj:
                pad_coords = torch.zeros((n_adj - n_coords, coords.shape[1]), dtype=coords.dtype)
                coords = torch.cat([coords, pad_coords], dim=0)
        
        if comm is not None:
            n_comm = comm.shape[1]
            if n_comm > n_adj:
                comm = comm[:, :n_adj]
            elif n_comm < n_adj:
                comm = torch.nn.functional.pad(comm, (0, n_adj - n_comm), mode='constant', value=-1)
        
        n = adj.shape[1]
        if n < self.N:
            pad = self.N - n
            adj = torch.nn.functional.pad(adj, (0, pad, 0, pad), mode='constant', value=0.0)
            if comm is not None:
                comm = torch.nn.functional.pad(comm, (0, pad), mode='constant', value=-1)
            if coords is not None:
                pad_coords = torch.zeros((pad, coords.shape[1]), dtype=coords.dtype)
                coords = torch.cat([coords, pad_coords], dim=0)
        elif n > self.N:
            adj = adj[:, :self.N, :self.N]
            if comm is not None:
                comm = comm[:, :self.N]
            if coords is not None:
                coords = coords[:self.N]
        
        return adj, comm, coords
    
    def _pad_global_channel(self, adj, coords, comm):
        """填充 Global 通道数据 (无时序)"""
        # adj: (n, n) - 2D!
        n = adj.shape[0]
        
        if coords is not None:
            n_coords = coords.shape[0]
            if n_coords > n:
                coords = coords[:n]
            elif n_coords < n:
                pad_coords = torch.zeros((n - n_coords, coords.shape[1]), dtype=coords.dtype)
                coords = torch.cat([coords, pad_coords], dim=0)
        
        # ← 新增：处理社区标签
        if comm is not None:
            n_comm = comm.shape[0]
            if n_comm > n:
                comm = comm[:n]
            elif n_comm < n:
                # 用-1填充（表示无社区）
                pad_comm = torch.full((n - n_comm,), -1, dtype=comm.dtype)
                comm = torch.cat([comm, pad_comm], dim=0)
        
        if n < self.N:
            pad = self.N - n
            adj = torch.nn.functional.pad(adj, (0, pad, 0, pad), mode='constant', value=0.0)
            if coords is not None:
                pad_coords = torch.zeros((pad, coords.shape[1]), dtype=coords.dtype)
                coords = torch.cat([coords, pad_coords], dim=0)
            # ← 新增：填充社区标签
            if comm is not None:
                pad_comm = torch.full((pad,), -1, dtype=comm.dtype)
                comm = torch.cat([comm, pad_comm], dim=0)
        elif n > self.N:
            adj = adj[:self.N, :self.N]
            if coords is not None:
                coords = coords[:self.N]
            if comm is not None:
                comm = comm[:self.N]
        
        return adj, coords, comm

    def _compute_max_community_id(self):
        """
        计算数据集中的最大社区ID
        扫描所有文件，找到 Local 和 Global 通道中的最大社区标签
        """
        max_comm_id = 0
        has_comm1 = False
        has_comm2 = False
        comm1_count = 0  # 有多少个文件包含 comm1
        comm2_count = 0  # 有多少个文件包含 comm2
        
        print("Computing max community ID from all files...")
        
        for i, p in enumerate(self.files):
            try:
                # ========== 加载 Local 通道 (comm1) ==========
                d1 = torch.load(p, map_location='cpu')
                comm1 = d1.get('community_sequence', None)
                
                if comm1 is not None:
                    has_comm1 = True
                    comm1_count += 1
                    # comm1 是 (T, N) 形状，需要展平后过滤
                    valid = comm1[comm1 >= 0]
                    if len(valid) > 0:
                        max_id_in_file = int(valid.max().item())
                        max_comm_id = max(max_comm_id, max_id_in_file)
                
                # ========== 加载 Global 通道 (comm2) ==========
                if self.dual_channel:
                    p2 = self.files2[i]
                    d2 = torch.load(p2, map_location='cpu')
                    comm2 = d2.get('community_labels', None)
                    
                    if comm2 is not None:
                        has_comm2 = True
                        comm2_count += 1
                        # comm2 是 (N,) 形状
                        valid = comm2[comm2 >= 0]
                        if len(valid) > 0:
                            max_id_in_file = int(valid.max().item())
                            max_comm_id = max(max_comm_id, max_id_in_file)
                            
            except Exception as e:
                print(f"Warning: unable to read community labels from {p}: {e}")
                continue
        
        # 存储结果
        self.max_community_id = max_comm_id
        self.num_communities = max_comm_id + 1  # 社区数量 = 最大ID + 1
        
        # 打印统计信息
        print(f"\n{'='*60}")
        print(f"COMMUNITY DETECTION RESULTS")
        print(f"{'='*60}")
        print(f"Max community ID found: {self.max_community_id}")
        print(f"Total communities: {self.num_communities}")
        print(f"Local channel (comm1):")
        print(f"  - Has communities: {has_comm1}")
        print(f"  - Files with communities: {comm1_count}/{len(self.files)}")
        
        if self.dual_channel:
            print(f"Global channel (comm2):")
            print(f"  - Has communities: {has_comm2}")
            print(f"  - Files with communities: {comm2_count}/{len(self.files2)}")
        
        print(f"{'='*60}\n")
        
        # 如果没有检测到任何社区，给出警告
        if max_comm_id == 0 and not has_comm1 and not has_comm2:
            print("⚠️  WARNING: No community labels found in dataset!")
            print("   The community-enhanced features will not be effective.")

def collate_fn(batch):
    """Collate function: Local有时序，Global无时序"""
    # Local 通道
    Ls1 = [b['len1'] for b in batch]
    T_max1 = max(Ls1)
    
    B = len(batch)
    _, N, _ = batch[0]['adj1'].shape
    
    # Local: (B, T, N, N)
    adj1_batch = torch.zeros((B, T_max1, N, N), dtype=torch.float32)
    comm1_batch = None
    if batch[0]['comm1'] is not None:
        comm1_batch = torch.full((B, T_max1, N), -1, dtype=torch.long)
    coords1_batch = torch.zeros((B, N, 3), dtype=torch.float32)
    lengths1 = torch.tensor(Ls1, dtype=torch.long)
    
    # Global: (B, N, N) - 无时序维度！
    has_ch2 = 'adj2' in batch[0]
    if has_ch2:
        adj2_batch = torch.zeros((B, N, N), dtype=torch.float32)
        coords2_batch = torch.zeros((B, N, 3), dtype=torch.float32)
        comm2_batch = None  # ← 新增
        if batch[0].get('comm2') is not None:
            comm2_batch = torch.full((B, N), -1, dtype=torch.long)    
    labels = torch.stack([b['label'] for b in batch])
    files = [b['file'] for b in batch]
    
    for i, b in enumerate(batch):
        t1 = b['len1']
        adj1_batch[i, :t1] = b['adj1']
        if comm1_batch is not None:
            comm1_batch[i, :t1] = b['comm1']
        if b['coords1'] is not None:
            coords1_batch[i] = b['coords1']
        
        if has_ch2:
            adj2_batch[i] = b['adj2']
            if b['coords2'] is not None:
                coords2_batch[i] = b['coords2']
            # ← 新增：填充社区标签
            if b.get('comm2') is not None and comm2_batch is not None:
                comm2_batch[i] = b['comm2']    
                
    result = {
        'adj1': adj1_batch,
        'comm1': comm1_batch,
        'coords1': coords1_batch,
        'lengths1': lengths1,
        'label': labels,
        'files': files
    }
    
    if has_ch2:
        result['adj2'] = adj2_batch
        result['coords2'] = coords2_batch
        result['comm2'] = comm2_batch  # ← 新增
    
    return result


# -----------------------------
# Balanced batch sampler
# -----------------------------
from typing import Iterable

class BalancedBatchSampler(Sampler):
    def __init__(self, labels: List[int], batch_size: int, seed: int = 42):
        super().__init__(labels)
        self.labels = list(labels)
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        self.pos_idx = [i for i, l in enumerate(self.labels) if int(l) == 1]
        self.neg_idx = [i for i, l in enumerate(self.labels) if int(l) == 0]
        if len(self.pos_idx) == 0 or len(self.neg_idx) == 0:
            raise RuntimeError("BalancedBatchSampler requires both positive and negative samples.")
        self.num_batches = math.ceil(len(self.labels) / float(self.batch_size))
        self._rng = random.Random(seed)

    def __iter__(self) -> Iterable[List[int]]:
        for _ in range(self.num_batches):
            n_pos = self.batch_size // 2
            n_neg = self.batch_size - n_pos
            if len(self.pos_idx) >= n_pos:
                pos_sel = self._rng.sample(self.pos_idx, n_pos)
            else:
                pos_sel = [self._rng.choice(self.pos_idx) for _ in range(n_pos)]
            if len(self.neg_idx) >= n_neg:
                neg_sel = self._rng.sample(self.neg_idx, n_neg)
            else:
                neg_sel = [self._rng.choice(self.neg_idx) for _ in range(n_neg)]
            batch = pos_sel + neg_sel
            self._rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# -----------------------------
# Community anomaly detector (仅用于 Local 通道)
# -----------------------------
class CommunityAnomalyDetector:
    def __init__(self, freq_tensor: torch.Tensor, total: int):
        self.freq = freq_tensor.clone().float()
        self.total = float(total)
        eps = 1e-8
        self.log_probs = torch.log((self.freq + eps) / (self.total + eps))

    def score_batch(self, comm_batch: torch.Tensor) -> torch.Tensor:
        B, T, N = comm_batch.shape
        device = comm_batch.device
        log_probs = self.log_probs.to(device)
        comm_ids = (comm_batch + 1).clamp(min=0, max=len(log_probs)-1)
        lp = log_probs[comm_ids]
        valid = (comm_batch >= 0).float()
        sum_lp = (lp * valid).sum(dim=-1)
        count = valid.sum(dim=-1).clamp(min=1)
        avg_lp = sum_lp / count
        return -avg_lp


def load_labels_csv(csv_path: str) -> Dict[str, int]:
    labels = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        import csv
        reader = csv.DictReader(f)
        for row in reader:
            labels[row['file']] = int(row['label'])
    return labels


def compute_community_frequencies(dataset: PTGraphSequenceDataset, indices: List[int]) -> Tuple[torch.Tensor, int]:
    from collections import defaultdict
    freq = defaultdict(int)
    total = 0
    max_id = 0
    for idx in indices:
        sample = dataset[idx]
        comm = sample['comm1']
        if comm is None:
            continue
        valid = comm[comm >= 0]
        for cid in valid.flatten().tolist():
            cid = int(cid)
            freq[cid] += 1
            max_id = max(max_id, cid)
            total += 1
    if total == 0:
        return torch.zeros(1), 0
    freq_tensor = torch.zeros(max_id + 1, dtype=torch.float32)
    for cid, cnt in freq.items():
        freq_tensor[cid] = cnt
    return freq_tensor, total

# -----------------------------
# Positional Encoding
# -----------------------------
class PositionalEncoding(nn.Module):
    """
    位置编码模块：为时间序列添加位置信息
    使用正弦-余弦函数编码，与Transformer原文一致
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度
        
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, d_model) 输入序列
        Returns:
            x + PE: (B, seq_len, d_model) 添加位置编码后的序列
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# -----------------------------
# Model Components
# -----------------------------

# -----------------------------
# GAT Layer (新增)
# -----------------------------
class GATLayer(nn.Module):
    """Graph Attention Layer for brain network analysis"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, 
                 dropout: float = 0.1, concat: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.concat = concat
        
        # 每个head的输出维度
        if concat:
            assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads when concat=True"
            self.head_dim = out_dim // num_heads
        else:
            self.head_dim = out_dim
        
        # 线性变换（所有heads共享）
        self.W = nn.Linear(in_dim, self.head_dim * num_heads, bias=False)
        
        # 注意力参数
        self.a = nn.Parameter(torch.randn(num_heads, 2 * self.head_dim))
        
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim) 节点特征
            adj: (B, N, N) 邻接矩阵
        Returns:
            out: (B, N, out_dim) 输出特征
        """
        B, N, _ = x.shape
        device = x.device
        
        # 线性变换: (B, N, head_dim * num_heads)
        h = self.W(x)
        h = h.view(B, N, self.num_heads, self.head_dim)  # (B, N, num_heads, head_dim)
        h = h.transpose(1, 2)  # (B, num_heads, N, head_dim)
        
        # 计算注意力分数
        # 为每对节点计算注意力
        h_i = h.unsqueeze(3).expand(-1, -1, -1, N, -1)  # (B, num_heads, N, N, head_dim)
        h_j = h.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, num_heads, N, N, head_dim)
        
        # 拼接 [h_i || h_j]
        concat_h = torch.cat([h_i, h_j], dim=-1)  # (B, num_heads, N, N, 2*head_dim)
        
        # 计算注意力系数: (B, num_heads, N, N)
        e = torch.einsum('bhnmd,hd->bhnm', concat_h, self.a)
        e = self.leakyrelu(e)
        
        # 邻接矩阵mask: 只对有边的节点对计算注意力
        # adj: (B, N, N) -> (B, 1, N, N)
        adj_mask = adj.unsqueeze(1)
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj_mask > 0, e, zero_vec)
        
        # Softmax归一化
        attention = torch.softmax(attention, dim=-1)  # (B, num_heads, N, N)
        attention = self.dropout(attention)
        
        # 加权聚合
        h = h.transpose(2, 3)  # (B, num_heads, head_dim, N)
        out = torch.matmul(attention, h.transpose(2, 3))  # (B, num_heads, N, head_dim)
        
        # 合并多头
        if self.concat:
            out = out.transpose(1, 2).contiguous()  # (B, N, num_heads, head_dim)
            out = out.view(B, N, -1)  # (B, N, out_dim)
        else:
            out = out.mean(dim=1)  # (B, N, head_dim)
        
        return out


class MultiHeadGATLayer(nn.Module):
    """Multi-head GAT with residual connection and layer norm"""
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.gat = GATLayer(in_dim, out_dim, num_heads, dropout, concat=True)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 残差连接的投影（如果维度不匹配）
        self.residual_proj = None
        if in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # GAT
        out = self.gat(x, adj)
        out = self.dropout(out)
        
        # 残差连接
        if self.residual_proj is not None:
            identity = self.residual_proj(identity)
        out = out + identity
        
        # Layer normalization
        out = self.norm(out)
        
        return out

class WindowEncoder(nn.Module):
    """Local 通道的窗口编码器（保持不变）"""
    def __init__(self, N: int, coord_dim: int = 3, comm_vocab: int = 128, out_dim: int = 128):
        super().__init__()
        self.N = N
        self.comm_emb = nn.Embedding(comm_vocab + 1, 32, padding_idx=0)
        feat_dim = 1 + coord_dim + coord_dim + 32 + 1
        self.input_norm = nn.LayerNorm(feat_dim)
        self.input_proj = nn.Linear(feat_dim, out_dim)
        self.ffn = nn.Sequential(
            nn.Linear(out_dim, out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim)
        )
        self.ln_out = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(0.1)

    def forward(self, W_curr: torch.Tensor, W_prev: torch.Tensor, coords: torch.Tensor, comm: torch.Tensor = None):
        B, N = W_curr.shape[0], W_curr.shape[1]
        device = W_curr.device
        coords_b = coords if coords is not None else torch.zeros((B, N, 3), device=device)
        if comm is None:
            comm = torch.zeros((B, N), dtype=torch.long, device=device)
        
        deg_curr = W_curr.abs().sum(dim=-1)
        deg_prev = W_prev.abs().sum(dim=-1)
        deg_diff = deg_curr - deg_prev
        
        norm_adj = W_curr / (deg_curr.unsqueeze(-1).clamp(min=1))
        neigh_coords = torch.matmul(norm_adj, coords_b)
        
        deg_feat = deg_curr.unsqueeze(-1)
        deg_diff_feat = deg_diff.unsqueeze(-1)
        comm_ids = (comm + 1).clamp(min=0)
        comm_feat = self.comm_emb(comm_ids)
        
        feat = torch.cat([deg_feat, coords_b, neigh_coords, comm_feat, deg_diff_feat], dim=-1)
        feat = self.input_norm(feat)
        x_proj = self.input_proj(feat)
        x_res = self.ffn(x_proj)
        x = self.ln_out(x_proj + x_res)
        x = self.drop(x)
        pooled = x.mean(dim=1)
        return pooled


class EnhancedGlobalGNN(nn.Module):
    """增强版全局GNN - 支持GCN/GAT切换"""
    def __init__(self, N: int, coord_dim: int = 3, hidden_dim: int = 64, 
                 out_dim: int = 128, num_layers: int = 3, 
                 num_communities: int = 20, dropout: float = 0.1,
                 gnn_type: str = 'gat', num_heads: int = 4):  # 🆕 新增参数
        super().__init__()
        self.N = N
        self.num_layers = num_layers
        self.num_communities = num_communities
        self.gnn_type = gnn_type
        
        print(f"[EnhancedGlobalGNN] Using {gnn_type.upper()} architecture with {num_heads} heads")
        
        # 连接性特征
        init_feat_dim = N + 1 + coord_dim
        
        self.input_proj = nn.Sequential(
            nn.Linear(init_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # 社区嵌入（带门控机制）
        self.community_embeddings = nn.Parameter(
            torch.randn(num_communities + 1, hidden_dim) * 0.01
        )
        self.community_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )
        
        # 🆕 根据类型构建GNN层
        self.gnn_layers = nn.ModuleList()
        
        if gnn_type == 'gat':
            # GAT层
            for _ in range(num_layers):
                self.gnn_layers.append(
                    MultiHeadGATLayer(hidden_dim, hidden_dim, 
                                     num_heads=num_heads, dropout=dropout)
                )
        else:  # 'gcn'
            # 原有的GCN层
            self.gcn_norms = nn.ModuleList()
            for _ in range(num_layers):
                self.gnn_layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.gcn_norms.append(nn.LayerNorm(hidden_dim))
        
        # 🆕 多尺度池化（mean + max + attention）
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim * 3, out_dim),  # 3种池化
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, adj: torch.Tensor, coords: torch.Tensor = None, 
                comm: torch.Tensor = None):
        B, N, _ = adj.shape
        device = adj.device
        
        # 初始特征
        connection_features = adj
        deg = adj.abs().sum(dim=-1, keepdim=True)
        if coords is None:
            coords = torch.zeros((B, N, 3), device=device)
        x = torch.cat([connection_features, deg, coords], dim=-1)
        x = self.input_proj(x)
        
        # 🆕 社区信息融合（门控机制）
        if comm is not None:
            comm_ids = comm.clone()
            comm_ids[comm_ids == -1] = self.num_communities
            comm_ids = comm_ids.clamp(0, self.num_communities)
            comm_emb = self.community_embeddings[comm_ids]
            
            # 门控融合
            gate_input = torch.cat([x, comm_emb], dim=-1)
            gate = self.community_gate(gate_input)
            x = x * (1 - gate) + comm_emb * gate
        
        # GNN传播
        if self.gnn_type == 'gat':
            # 🆕 GAT传播（不需要归一化邻接矩阵）
            for gat_layer in self.gnn_layers:
                x = gat_layer(x, adj)
        else:
            # GCN传播（需要归一化邻接矩阵）
            eye = torch.eye(N, device=device).unsqueeze(0).expand(B, -1, -1)
            adj_self = adj + eye
            deg_self = adj_self.abs().sum(dim=-1).clamp(min=1e-6)
            deg_inv_sqrt = deg_self.pow(-0.5)
            norm_adj = deg_inv_sqrt.unsqueeze(-1) * adj_self * deg_inv_sqrt.unsqueeze(-2)
            
            for i in range(self.num_layers):
                identity = x
                x_neigh = torch.bmm(norm_adj, x)
                x_new = self.gnn_layers[i](x_neigh)
                x_new = self.gcn_norms[i](x_new)
                x_new = torch.relu(x_new)
                x_new = self.dropout(x_new)
                x = identity + x_new
        
        # 🆕 多尺度池化
        x_mean = x.mean(dim=1)
        x_max = x.max(dim=1).values
        
        # 注意力池化
        attn_scores = self.attn_pool(x)  # (B, N, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        x_attn = (x * attn_weights).sum(dim=1)
        
        graph_emb = torch.cat([x_mean, x_max, x_attn], dim=-1)
        graph_emb = self.output_proj(graph_emb)
        
        return graph_emb
        
class MemoryModule(nn.Module):
    """记忆模块（两个通道共享结构）"""
    def __init__(self, mem_slots: int, mem_dim: int, input_dim: int):
        super().__init__()
        self.mem_slots = mem_slots
        self.mem_dim = mem_dim
        self.memory = nn.Parameter(torch.randn(1, mem_slots, mem_dim))
        self.input_proj = nn.Linear(input_dim, mem_dim)
        self.ln_in = nn.LayerNorm(input_dim)
        self.gate = nn.Sequential(nn.Linear(mem_dim * 2, mem_dim), nn.Sigmoid())

    def read(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x_proj = self.input_proj(self.ln_in(x))
        memory_b = self.memory.expand(B, -1, -1)
        attn_scores = torch.matmul(x_proj.unsqueeze(1), memory_b.transpose(1, 2))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        read_content = torch.matmul(attn_probs, memory_b).squeeze(1)
        return read_content

    def write(self, x: torch.Tensor):
        B = x.shape[0]
        x_proj = self.input_proj(self.ln_in(x))
        memory_b = self.memory.expand(B, -1, -1)
        attn_scores = torch.matmul(x_proj.unsqueeze(1), memory_b.transpose(1, 2))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        gate_input = torch.cat([memory_b, x_proj.unsqueeze(1).expand(-1, self.mem_slots, -1)], dim=-1)
        update_gate = self.gate(gate_input)
        write_info = attn_probs.transpose(1, 2) * x_proj.unsqueeze(1)
        avg_update_gate = update_gate.mean(dim=0, keepdim=True)
        avg_write_info = write_info.mean(dim=0, keepdim=True)
        self.memory.data = (self.memory.data * (1 - avg_update_gate) + avg_write_info * avg_update_gate).detach()


# -----------------------------
# Main Model
# -----------------------------
class DualChannelClassifier(nn.Module):
    def __init__(self, N: int, coords_dim: int = 3, comm_vocab: int = 128,
                 win_emb: int = 128, tf_layers: int = 2, tf_heads: int = 4,
                 gnn_layers: int = 3, gnn_hidden: int = 64,
                 mem_slots: int = 32, mem_dim: int = 128,
                 dropout: float = 0.3, comm_freq=None, 
                 num_communities=20, gnn_type='gat', gat_heads=4):
        super().__init__()
        self.N = N
        self.win_emb = win_emb
        
        # ========== Local 通道: WindowEncoder + Transformer ==========
        self.local_win_enc = WindowEncoder(N=N, coord_dim=coords_dim, comm_vocab=comm_vocab, out_dim=win_emb)
        self.local_cls_token = nn.Parameter(torch.randn(1, 1, win_emb))
        
        # 🆕 添加位置编码器
        self.pos_encoder = PositionalEncoding(d_model=win_emb, max_len=200, dropout=0.0)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=win_emb, nhead=tf_heads, dim_feedforward=win_emb*4, 
            dropout=dropout, batch_first=True
        )
        self.local_transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        self.local_memory = MemoryModule(mem_slots=mem_slots, mem_dim=mem_dim, input_dim=win_emb)
        
        # Anomaly detector (仅 Local)
        self.anom_proj = nn.Linear(1, win_emb)
        self.anom_detector = None
        if comm_freq is not None:
            freq_tensor, total = comm_freq
            self.anom_detector = CommunityAnomalyDetector(freq_tensor=freq_tensor, total=total)
        
        # ========== Global 通道: GNN ==========
        self.global_gnn = EnhancedGlobalGNN(
            N=N, coord_dim=coords_dim, hidden_dim=gnn_hidden,
            out_dim=win_emb, num_layers=gnn_layers,
            num_communities=num_communities,
            dropout=dropout,
            gnn_type=gnn_type,      # 🆕
            num_heads=gat_heads     # 🆕
        )
        self.global_memory = MemoryModule(mem_slots=mem_slots, mem_dim=mem_dim, input_dim=win_emb)
        
        # ========== 图级统计特征 ==========
        self.graph_feat_dim = 3
        self.local_graph_feat_proj = nn.Sequential(
            nn.LayerNorm(self.graph_feat_dim),
            nn.Linear(self.graph_feat_dim, mem_dim),
            nn.GELU()
        )
        self.global_graph_feat_proj = nn.Sequential(
            nn.LayerNorm(self.graph_feat_dim),
            nn.Linear(self.graph_feat_dim, mem_dim),
            nn.GELU()
        )
        
        # ========== 分类器 ==========
        clf_in = win_emb + mem_dim + mem_dim
        self.local_classifier = nn.Sequential(
            nn.LayerNorm(clf_in),
            nn.Linear(clf_in, 1)
        )
        self.global_classifier = nn.Sequential(
            nn.LayerNorm(clf_in),
            nn.Linear(clf_in, 1)
        )
        
        self.memory_write_enabled = False

        # 🆕 添加：用于存储动态loss的引用
        self.dynamic_loss = None  # 会在训练时设置

    def forward_local(self, adj_batch: torch.Tensor, coords: torch.Tensor, 
                    comm: torch.Tensor, lengths: torch.Tensor):
        """处理 Local 通道 (有时序)"""
        B, T, N, _ = adj_batch.shape
        device = adj_batch.device

        adj_prev = torch.cat([torch.zeros_like(adj_batch[:, :1]), adj_batch[:, :-1]], dim=1)

        # 异常检测分数
        anom_scores = None
        if self.anom_detector is not None and comm is not None:
            with torch.no_grad():
                anom_scores = self.anom_detector.score_batch(comm.to(device))

        # 编码每个时间窗口
        win_embs = []
        for t in range(T):
            W_curr = adj_batch[:, t]
            W_prev = adj_prev[:, t]
            comm_t = comm[:, t] if comm is not None else None
            
            pooled = self.local_win_enc(W_curr, W_prev, coords, comm_t)
            if anom_scores is not None:
                extra = self.anom_proj(anom_scores[:, t].unsqueeze(-1).to(device))
                pooled = pooled + extra
            win_embs.append(pooled.unsqueeze(1))
        X = torch.cat(win_embs, dim=1)  # (B, T, d)

        # 🆕 步骤1：添加位置编码
        X = self.pos_encoder(X)  # X' = X + PE, (B, T, d)

        # 🆕 步骤2：添加CLS token
        cls_tokens = self.local_cls_token.expand(B, -1, -1)  # (B, 1, d)
        X = torch.cat((cls_tokens, X), dim=1)  # (B, T+1, d)
        
        # 🆕 步骤3：Transformer编码
        mask = torch.arange(T + 1, device=device)[None, :] >= (lengths + 1)[:, None]
        tf_out = self.local_transformer(X, src_key_padding_mask=mask)
        cls_out = tf_out[:, 0]

        # Memory
        read_mem = self.local_memory.read(cls_out)
        if self.training and self.memory_write_enabled:
            self.local_memory.write(cls_out)

        # 图级统计特征
        with torch.no_grad():
            abs_adj = adj_batch.abs()
            mean_abs = abs_adj.mean(dim=(1,2,3))
            std_abs = abs_adj.view(B, -1).std(dim=1)
            mean_deg = abs_adj.sum(dim=(2,3)).mean(dim=1)
        graph_feats = torch.stack([mean_abs, std_abs, mean_deg], dim=1).to(device)
        proj_feats = self.local_graph_feat_proj(graph_feats)

        final_repr = torch.cat([cls_out, read_mem, proj_feats], dim=1)
        logits = self.local_classifier(final_repr).squeeze(-1)
        return logits, final_repr
    def forward_global(self, adj: torch.Tensor, coords: torch.Tensor, comm: torch.Tensor = None):
        """
        处理 Global 通道 (无时序，使用 GNN)
        
        Args:
            adj: (B, N, N) 全局邻接矩阵
            coords: (B, N, 3) 坐标
            comm: (B, N) 社区标签（可选）← 新增
        """
        B = adj.shape[0]
        device = adj.device

        # GNN 编码（传入社区标签）← 修改这里
        gnn_out = self.global_gnn(adj, coords, comm)  # (B, win_emb)
        # Memory
        read_mem = self.global_memory.read(gnn_out)
        if self.training and self.memory_write_enabled:
            self.global_memory.write(gnn_out)

        # 图级统计特征
        with torch.no_grad():
            abs_adj = adj.abs()
            mean_abs = abs_adj.mean(dim=(1,2))
            std_abs = abs_adj.view(B, -1).std(dim=1)
            mean_deg = abs_adj.sum(dim=(1,2)) / adj.shape[1]
        graph_feats = torch.stack([mean_abs, std_abs, mean_deg], dim=1).to(device)
        proj_feats = self.global_graph_feat_proj(graph_feats)

        final_repr = torch.cat([gnn_out, read_mem, proj_feats], dim=1)
        logits = self.global_classifier(final_repr).squeeze(-1)
        return logits, final_repr

    def forward(self, batch_data: dict):
        has_ch2 = 'adj2' in batch_data
        
        # Local 通道
        local_logits, local_repr = self.forward_local(
            batch_data['adj1'],
            batch_data['coords1'],
            batch_data['comm1'],
            batch_data['lengths1']
        )
        
        if not has_ch2:
            return local_logits
        
        # Global 通道
        global_logits, global_repr = self.forward_global(
            batch_data['adj2'],
            batch_data['coords2'],
            batch_data.get('comm2')
        )        

        if self.training:
            # 训练时：返回两个logits，由动态loss处理
            return local_logits, global_logits
        else:
            # 🆕 推理时：使用动态loss学到的权重
            return (local_logits + global_logits) / 2

# -----------------------------
# Training / evaluation helpers
# -----------------------------

# -----------------------------
# Staged Dynamic Loss for Dual-Channel Fusion (新增)
# -----------------------------
class StagedDynamicLoss(nn.Module):
    """
    分阶段动态权重学习
    - Stage 1 (warmup): 固定权重0.5:0.5
    - Stage 2 (learning): 学习权重 + 多重约束
    - Stage 3 (stable): 冻结权重
    """
    def __init__(self, warmup_epochs: int = 20, learning_epochs: int = 30, 
                 min_weight: float = 0.25, balance_penalty: float = 1.0):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.learning_epochs = learning_epochs
        self.min_weight = min_weight
        self.balance_penalty = balance_penalty
        
        # 🔥 可学习权重（对数域）
        self.log_weight_local = nn.Parameter(torch.tensor(0.0))
        
        # 监控统计
        self.register_buffer('local_pred_mean', torch.tensor(0.5))
        self.register_buffer('global_pred_mean', torch.tensor(0.5))
        self.register_buffer('collapse_counter', torch.tensor(0))
        self.register_buffer('weight_history', torch.zeros(1000))
        self.register_buffer('history_idx', torch.tensor(0))
        
        print(f"[StagedDynamicLoss] warmup={warmup_epochs}, learning={learning_epochs}, "
              f"min_weight={min_weight}, balance_penalty={balance_penalty}")
    
    def get_stage(self, epoch: int) -> str:
        """判断当前训练阶段"""
        if epoch <= self.warmup_epochs:
            return 'warmup'
        elif epoch <= self.warmup_epochs + self.learning_epochs:
            return 'learning'
        else:
            return 'stable'
    
    def get_weights(self, epoch: int):
        """获取当前epoch的权重"""
        stage = self.get_stage(epoch)
        
        if stage == 'warmup':
            # Stage 1: 固定权重
            return 0.5, 0.5, stage
        
        elif stage == 'learning':
            # Stage 2: 可学习权重（带约束）
            w_local = torch.sigmoid(self.log_weight_local)
            w_local = torch.clamp(w_local, self.min_weight, 1 - self.min_weight)
            w_global = 1 - w_local
            return w_local.item(), w_global.item(), stage
        
        else:  # stable
            # Stage 3: 冻结权重
            with torch.no_grad():
                w_local = torch.sigmoid(self.log_weight_local)
                w_local = torch.clamp(w_local, self.min_weight, 1 - self.min_weight)
            w_global = 1 - w_local
            return w_local.item(), w_global.item(), stage
    
    def forward(self, logits_local: torch.Tensor, logits_global: torch.Tensor, 
                labels: torch.Tensor, criterion, epoch: int = 1):
        """
        Args:
            logits_local: (B,) local通道的logits
            logits_global: (B,) global通道的logits  
            labels: (B,) 真实标签
            criterion: 基础loss函数
            epoch: 当前epoch
        Returns:
            loss_total: 总loss
            info: 统计信息字典
        """
        stage = self.get_stage(epoch)
        w_local, w_global, _ = self.get_weights(epoch)
        
        # 基础分类loss
        loss_local = criterion(logits_local, labels)
        loss_global = criterion(logits_global, labels)
        loss_main = w_local * loss_local + w_global * loss_global
        
        # 预测统计
        with torch.no_grad():
            pred_local = torch.sigmoid(logits_local).mean()
            pred_global = torch.sigmoid(logits_global).mean()
            target_mean = labels.mean()
        
        # 构建返回信息
        info = {
            'stage': stage,
            'w_local': w_local,
            'w_global': w_global,
            'pred_local': pred_local.item(),
            'pred_global': pred_global.item(),
            'loss_main': loss_main.item()
        }
        
        if stage == 'warmup':
            # Stage 1: 只返回主loss
            return loss_main, info
        
        elif stage == 'learning':
            # Stage 2: 添加多重约束
            
            # 🔥 约束1: 类别平衡惩罚
            balance_loss = (
                (pred_local - target_mean) ** 2 + 
                (pred_global - target_mean) ** 2
            )
            
            # 🔥 约束2: 熵正则（防止预测过于确定）
            entropy_local = self._entropy(torch.sigmoid(logits_local))
            entropy_global = self._entropy(torch.sigmoid(logits_global))
            diversity_loss = (
                (0.5 - entropy_local).clamp(min=0) + 
                (0.5 - entropy_global).clamp(min=0)
            )
            
            # 🔥 约束3: 检测坍缩
            collapsed = (pred_local > 0.85 or pred_local < 0.15 or
                        pred_global > 0.85 or pred_global < 0.15)
            
            if collapsed:
                self.collapse_counter += 1
                collapse_penalty = 10.0 * balance_loss  # 强力拉回
                info['collapsed'] = True
            else:
                collapse_penalty = 0.0
                info['collapsed'] = False
            
            # 🔥 动态调整惩罚强度
            progress = (epoch - self.warmup_epochs) / self.learning_epochs
            alpha_balance = self.balance_penalty * max(0.1, 1.0 - progress)
            alpha_diversity = 0.3
            
            # 总loss
            loss_total = (loss_main + 
                         alpha_balance * balance_loss + 
                         alpha_diversity * diversity_loss + 
                         collapse_penalty)
            
            info.update({
                'balance_loss': balance_loss.item(),
                'diversity_loss': diversity_loss.item(),
                'alpha_balance': alpha_balance,
                'collapse_count': int(self.collapse_counter)
            })
            
            # 更新统计
            self.local_pred_mean = 0.9 * self.local_pred_mean + 0.1 * pred_local
            self.global_pred_mean = 0.9 * self.global_pred_mean + 0.1 * pred_global
            
            # 记录权重历史
            idx = int(self.history_idx % 1000)
            self.weight_history[idx] = w_local
            self.history_idx += 1
            
            return loss_total, info
        
        else:  # stable
            # Stage 3: 冻结权重，只返回主loss
            return loss_main, info
    
    def _entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """计算二分类熵"""
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        return -(probs * torch.log(probs) + 
                (1 - probs) * torch.log(1 - probs)).mean()
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        if self.history_idx < 1:
            return {}
        
        n = min(int(self.history_idx), 1000)
        history = self.weight_history[:n]
        
        return {
            'weight_mean': history.mean().item(),
            'weight_std': history.std().item(),
            'local_pred_mean': self.local_pred_mean.item(),
            'global_pred_mean': self.global_pred_mean.item(),
            'collapse_count': int(self.collapse_counter)
        }

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, thresh=0.5) -> Dict[str, float]:
    y_pred = (y_score >= thresh).astype(int)
    try:
        from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix
        acc = accuracy_score(y_true, y_pred)
        auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) > 1 else float('nan')
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception:
        y_true_arr = np.asarray(y_true).astype(int)
        y_pred_arr = np.asarray(y_pred).astype(int)
        acc = float((y_true_arr == y_pred_arr).mean())
        tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
        fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
        tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())
        fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        auc = float('nan')
        sen = recall
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal = 0.5 * (sen + spe)
    return {'acc': acc, 'auc': auc, 'f1': f1, 'sen': sen, 'spe': spe, 'bal': bal}


def train_epoch(model, loader, opt, device, criterion, has_dual_channel=False,
                dynamic_loss=None, epoch=1):  # 🆕 新增参数
    """
    训练一个epoch
    Args:
        dynamic_loss: 可选的动态loss模块（用于分阶段训练）
        epoch: 当前epoch数
    """
    model.train()
    losses = []
    loss_details = {
        'main': [],
        'balance': [],
        'diversity': []
    }
    
    # 用于监控预测分布
    all_pred_local = []
    all_pred_global = []
    all_labels = []
    
    for batch in loader:
        batch_data = {
            'adj1': batch['adj1'].to(device),
            'comm1': batch['comm1'].to(device) if batch['comm1'] is not None else None,
            'coords1': batch['coords1'].to(device),
            'lengths1': batch['lengths1'].to(device),
        }
        
        if has_dual_channel:
            batch_data['adj2'] = batch['adj2'].to(device)
            batch_data['coords2'] = batch['coords2'].to(device)
            batch_data['comm2'] = batch['comm2'].to(device) if batch.get('comm2') is not None else None
        
        labels = batch['label'].to(device)
        opt.zero_grad()
        
        output = model(batch_data)
        
        if has_dual_channel and model.training:
            local_logits, global_logits = output
            
            # 🆕 使用动态loss
            if dynamic_loss is not None:
                loss, info = dynamic_loss(local_logits, global_logits, labels, 
                                        criterion, epoch)
                
                # 记录详细loss
                loss_details['main'].append(info.get('loss_main', 0))
                if 'balance_loss' in info:
                    loss_details['balance'].append(info['balance_loss'])
                if 'diversity_loss' in info:
                    loss_details['diversity'].append(info['diversity_loss'])
            else:
                # 原有的简单加权
                loss_local = criterion(local_logits, labels)
                loss_global = criterion(global_logits, labels)
                loss = loss_local + loss_global
            
            # 收集预测统计
            with torch.no_grad():
                all_pred_local.append(torch.sigmoid(local_logits).cpu())
                all_pred_global.append(torch.sigmoid(global_logits).cpu())
                all_labels.append(labels.cpu())
        else:
            loss = criterion(output, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()
        losses.append(loss.item())
    
    # 计算epoch级别的统计
    result = {'loss': float(np.mean(losses))}
    
    if has_dual_channel and len(all_pred_local) > 0:
        all_pred_local = torch.cat(all_pred_local)
        all_pred_global = torch.cat(all_pred_global)
        all_labels = torch.cat(all_labels)
        
        result['pred_local_mean'] = all_pred_local.mean().item()
        result['pred_global_mean'] = all_pred_global.mean().item()
        result['label_mean'] = all_labels.mean().item()
        
        # 详细loss统计
        for key, values in loss_details.items():
            if values:
                result[f'loss_{key}'] = float(np.mean(values))
    
    return result

def eval_epoch(model, loader, device, has_dual_channel=False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    model.eval()
    ys = []
    ys_score = []
    files = []
    with torch.no_grad():
        for batch in loader:
            batch_data = {
                'adj1': batch['adj1'].to(device),
                'comm1': batch['comm1'].to(device) if batch['comm1'] is not None else None,
                'coords1': batch['coords1'].to(device),
                'lengths1': batch['lengths1'].to(device),
            }
            
            if has_dual_channel:
                batch_data['adj2'] = batch['adj2'].to(device)
                batch_data['coords2'] = batch['coords2'].to(device)
                batch_data['comm2'] = batch['comm2'].to(device) if batch.get('comm2') is not None else None
            
            labels = batch['label'].cpu().numpy()
            logits = model(batch_data)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            ys.append(labels)
            ys_score.append(probs)
            files.extend(batch['files'])
    
    return np.concatenate(ys), np.concatenate(ys_score), files


def find_best_threshold(y_true: np.ndarray, y_score: np.ndarray, 
                       metric: str = 'balanced', grid=None) -> float:
    """
    Find threshold on y_score maximizing chosen metric.
    metric: 'accuracy' | 'f1' | 'balanced' | 'youden'
    Uses a dense grid 0.01..0.99. 
    Tie-break: prefer smaller threshold (more positives) to avoid trivial all-negative.
    """
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)  # ← 99个点，更精细
    best_t = 0.5
    best_val = -1.0

    def _stats(y_true_arr, y_pred_arr):
        y_true_arr = np.asarray(y_true_arr).astype(int)
        y_pred_arr = np.asarray(y_pred_arr).astype(int)
        tp = int(((y_pred_arr == 1) & (y_true_arr == 1)).sum())
        fp = int(((y_pred_arr == 1) & (y_true_arr == 0)).sum())
        tn = int(((y_pred_arr == 0) & (y_true_arr == 0)).sum())
        fn = int(((y_pred_arr == 0) & (y_true_arr == 1)).sum())
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = float((y_true_arr == y_pred_arr).mean())
        bal = 0.5 * (sens + spec)
        youden = sens + spec - 1.0
        return {
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'sens': sens, 'spec': spec, 'f1': f1,
            'acc': acc, 'bal': bal, 'youden': youden
        }

    # Try to use sklearn for f1 if available (more accurate)
    use_sklearn_f1 = False
    try:
        from sklearn.metrics import f1_score as _sk_f1
        use_sklearn_f1 = True
    except Exception:
        use_sklearn_f1 = False

    for t in grid:
        preds = (y_score >= t).astype(int)
        s = _stats(y_true, preds)
        
        if metric == 'accuracy':
            val = s['acc']
        elif metric == 'f1':
            if use_sklearn_f1:
                val = float(_sk_f1(y_true, preds, zero_division=0))
            else:
                val = s['f1']
        elif metric == 'youden':
            val = s['youden']
        else:  # 'balanced' default
            val = s['bal']

        # Tie-breaking: prefer smaller threshold on ties (encourage some positives)
        if (val > best_val) or (abs(val - best_val) < 1e-8 and t < best_t):
            best_val = val
            best_t = float(t)
    
    return best_t

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        if self.label_smoothing > 0:
            targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        bce = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = ((1 - pt) ** self.gamma) * bce
        return focal.mean()


def main(args):
    start_time = time.time()
    
    # ============================================
    # 改进的随机种子设置
    # ============================================
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using fixed random seed = {args.seed}")

    labels_map = load_labels_csv(args.labels_csv) if args.labels_csv else None
    dataset = PTGraphSequenceDataset(args.pt_dir, labels_map=labels_map, pt_dir2=args.pt_dir2)
    has_dual_channel = dataset.dual_channel
    
    if has_dual_channel:
        print("=" * 60)
        print("DUAL-CHANNEL MODE: Local (Transformer) + Global (GNN)")
        print("=" * 60)
    
    # 🆕 获取自动检测的社区数量
    num_communities = dataset.num_communities
    print(f"✓ Using auto-detected num_communities = {num_communities}\n")

    # Split
    # ============================================
    # 🔧 修复：使用分层划分替代随机划分
    # ============================================
    n_total = len(dataset)
    
    # 获取所有标签
    all_labels = [int(dataset[i]['label'].item()) for i in range(n_total)]
    labels_arr = np.array(all_labels, dtype=int)
    
    # 打印标签分布
    n_pos = int(labels_arr.sum())
    n_neg = int(n_total - n_pos)
    print(f"\nTotal samples: {n_total}, Positive: {n_pos}, Negative: {n_neg}")
    
    # --- stratified 60/20/20 split to preserve class balance across splits ---
    try:
        from sklearn.model_selection import StratifiedShuffleSplit
        
        # 第一步：划分 test (20%)
        sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
        trainval_idx, test_idx = next(sss1.split(np.zeros(n_total), labels_arr))
        
        # 第二步：从 trainval 中划分 val (25% of trainval = 20% of total)
        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
        train_idx_rel, val_idx_rel = next(sss2.split(
            np.zeros(len(trainval_idx)), 
            labels_arr[trainval_idx]
        ))
        
        train_idx = trainval_idx[train_idx_rel].tolist()
        val_idx = trainval_idx[val_idx_rel].tolist()
        test_idx = test_idx.tolist()
        
        print("Using StratifiedShuffleSplit for balanced data splits")
        
    except Exception as e:
        warnings.warn(f"sklearn StratifiedShuffleSplit import failed ({e}), using fallback stratified split.")
        
        # Fallback：手动分层划分
        indices = np.arange(n_total)
        train_idx = []
        val_idx = []
        test_idx = []
        
        # 对每个类别分别进行 60/20/20 划分
        for cls in np.unique(labels_arr):
            cls_inds = indices[labels_arr == cls]
            np.random.shuffle(cls_inds)
            
            n = len(cls_inds)
            n_test = int(round(0.2 * n))   # 20% test
            n_val = int(round(0.2 * n))    # 20% val
            n_train = n - n_val - n_test   # 60% train
            
            train_idx.extend(cls_inds[:n_train].tolist())
            val_idx.extend(cls_inds[n_train:n_train + n_val].tolist())
            test_idx.extend(cls_inds[n_train + n_val:].tolist())
        
        # 打乱每个集合内部的顺序
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)
        
        print("Using fallback stratified split")
    
    # 创建 Subset 对象
    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)
    test_ds_split = Subset(dataset, test_idx)

    # ============================================
    # 🔍 添加：数据划分质量诊断
    # ============================================
    print("\n" + "="*60)
    print("DATA SPLIT QUALITY DIAGNOSIS")
    print("="*60)
    
    def get_split_stats(indices, name):
        labels = [all_labels[i] for i in indices]
        pos = sum(labels)
        neg = len(labels) - pos
        ratio = pos / len(labels) if len(labels) > 0 else 0
        print(f"{name:6s}: n={len(indices):4d}, pos={pos:3d}({ratio:5.1%}), neg={neg:3d}")
        return ratio
    
    train_ratio = get_split_stats(train_idx, "Train")
    val_ratio = get_split_stats(val_idx, "Val")
    test_ratio = get_split_stats(test_idx, "Test")
    
    # 检查类别比例差异
    max_diff = max(abs(train_ratio - val_ratio), 
                   abs(train_ratio - test_ratio), 
                   abs(val_ratio - test_ratio))
    
    print(f"\nMaximum class ratio difference: {max_diff:.3%}")
    
    if max_diff < 0.01:
        print("✓ Excellent balance! (diff < 1%)")
    elif max_diff < 0.05:
        print("✓ Good balance (diff < 5%)")
    elif max_diff < 0.10:
        print("⚠️  Moderate imbalance (diff < 10%) - may cause minor issues")
    else:
        print("❌ Poor balance (diff >= 10%) - this WILL cause test > val problems!")
    
    print("="*60 + "\n")
    
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test (split): {len(test_ds_split)}")
    
    # Community frequencies (Local only)
    comm_freq = compute_community_frequencies(dataset, train_idx)
    print(f"Community frequency computed from TRAIN SET ONLY: {comm_freq[0].shape}, total={comm_freq[1]}")

    # Balanced loader
    train_labels = [int(dataset[i]['label'].item()) for i in train_idx]
    pos_count = sum(train_labels)
    neg_count = len(train_labels) - pos_count
    print(f"Train class distribution: pos={pos_count}, neg={neg_count}")

    if not args.force_oversample:
        sampler = BalancedBatchSampler(train_labels, batch_size=args.batch_size, seed=args.seed)
        train_loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate_fn)
        print("Using BalancedBatchSampler (default)")
    else:
        weights = [1.0 / neg_count if l == 0 else 1.0 / pos_count for l in train_labels]
        sampler = WeightedRandomSampler(weights, num_samples=len(train_labels), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, collate_fn=collate_fn)
        print("Using WeightedRandomSampler (--force_oversample)")

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader_split = DataLoader(test_ds_split, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    model = DualChannelClassifier(
        N=dataset.N,
        coords_dim=3,
        comm_vocab=args.comm_vocab,
        win_emb=args.win_emb,
        tf_layers=args.tf_layers,
        tf_heads=args.tf_heads,
        gnn_layers=args.gnn_layers,
        gnn_hidden=args.gnn_hidden,
        mem_slots=args.mem_slots,
        mem_dim=args.mem_dim,
        dropout=args.dropout,
        comm_freq=comm_freq,
        num_communities=num_communities,
        gnn_type=args.gnn_type,        # 🆕
        gat_heads=args.gat_heads        # 🆕
    )
    
    if args.enable_memory_write:
        model.memory_write_enabled = True
        print("Memory write enabled during training")

    if args.init_bias_target is not None:
        prior = float(args.init_bias_target)
        init_bias = np.log(prior / (1.0 - prior + 1e-8))
        with torch.no_grad():
            model.local_classifier[-1].bias.fill_(init_bias)
            model.global_classifier[-1].bias.fill_(init_bias)
        print(f"Initialized classifier bias to {init_bias:.3f} (target prior={prior:.3f})")

    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    # 🆕 创建分阶段动态loss（如果使用双通道）
    dynamic_loss = None
    if has_dual_channel and args.use_dynamic_loss:
        dynamic_loss = StagedDynamicLoss(
            warmup_epochs=args.warmup_epochs,
            learning_epochs=args.learning_epochs,
            min_weight=args.min_weight,
            balance_penalty=args.balance_penalty
        )
        dynamic_loss = dynamic_loss.to(device)
        
        print(f"\n{'='*60}")
        print("USING STAGED DYNAMIC LOSS")
        print(f"  Stage 1 (Warmup): Epochs 1-{args.warmup_epochs}, Fixed weights 0.5:0.5")
        print(f"  Stage 2 (Learning): Epochs {args.warmup_epochs+1}-{args.warmup_epochs+args.learning_epochs}, "
              f"Learning weights with constraints")
        print(f"  Stage 3 (Stable): Epochs {args.warmup_epochs+args.learning_epochs+1}+, Frozen weights")
        print(f"{'='*60}\n")
        
    # Optimizer
    # 🆕 为动态loss权重参数使用更小的学习率
    if dynamic_loss is not None:
        opt = torch.optim.AdamW([
            {'params': [p for n, p in model.named_parameters()], 
             'lr': args.lr},
            {'params': dynamic_loss.parameters(), 
             'lr': args.lr * 0.1}  # 10倍小的学习率
        ], weight_decay=args.weight_decay)
        print(f"Using separate learning rates: model={args.lr:.2e}, dynamic_loss={args.lr*0.1:.2e}")
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    # Loss
    pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0
    pos_weight_tensor = torch.tensor([pos_weight_val], device=device)
    
    if args.use_focal:
        criterion = FocalLoss(gamma=2.0, label_smoothing=args.label_smoothing)
        print(f"Using FocalLoss with label_smoothing={args.label_smoothing}")
    else:
        class BCELossWithLS(nn.Module):
            def __init__(self, pos_weight, label_smoothing=0.0):
                super().__init__()
                self.pos_weight = pos_weight
                self.label_smoothing = label_smoothing
            
            def forward(self, inputs, targets):
                if self.label_smoothing > 0:
                    targets = targets * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
                return torch.nn.functional.binary_cross_entropy_with_logits(
                    inputs, targets, pos_weight=self.pos_weight
                )
        
        criterion = BCELossWithLS(pos_weight=pos_weight_tensor, label_smoothing=args.label_smoothing)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight_val:.3f}")

    # Training
    print("\n" + "="*60)
    print("TRACKING THREE BEST MODELS:")
    print("  1. Best ACC model")
    print("  2. Best AUC model")
    print("  3. Best Balanced (ACC+AUC) model")
    print("="*60 + "\n")

    best_acc = -1.0
    best_auc = -1.0
    best_balanced = -1.0
    best_state_acc = None
    best_state_auc = None
    best_state_balanced = None
    best_thresh_acc = 0.5
    best_thresh_auc = 0.5
    best_thresh_balanced = 0.5

    # 🆕 近期最佳模型队列 (按balanced metric排序)
    recent_best_models = deque(maxlen=args.save_recent_best if args.save_recent_best > 0 else None)

    # 🔧 改进的早停机制
    best_balanced_for_early_stop = -1.0
    no_improve = 0
    for epoch in range(1, args.epochs + 1):
        # 🆕 传入dynamic_loss和epoch
        train_result = train_epoch(model, train_loader, opt, device, criterion, 
                                   has_dual_channel, dynamic_loss, epoch)
        train_loss = train_result['loss']
        y_val, y_score, _ = eval_epoch(model, val_loader, device, has_dual_channel)
        best_thresh = find_best_threshold(y_val, y_score, metric=args.thresh_opt)
        metrics = compute_metrics(y_val, y_score, thresh=best_thresh)
        
        # 🆕 增强的日志输出
        log_str = (f"Epoch {epoch:03d} TrainLoss {train_loss:.4f} "
                  f"Val acc {metrics['acc']:.4f} auc {metrics['auc']:.4f} "
                  f"f1 {metrics['f1']:.4f} sen {metrics['sen']:.4f} "
                  f"spe {metrics['spe']:.4f} thresh {best_thresh:.3f} "
                  f"lr {opt.param_groups[0]['lr']:.2e}")
        
        # 显示动态loss信息
        if dynamic_loss is not None and has_dual_channel:
            stage = dynamic_loss.get_stage(epoch)
            w_local, w_global, _ = dynamic_loss.get_weights(epoch)
            log_str += f"\n  Stage: {stage:8s} | Weights: L={w_local:.3f} G={w_global:.3f}"
            
            if 'pred_local_mean' in train_result:
                log_str += (f" | PredMean: L={train_result['pred_local_mean']:.3f} "
                          f"G={train_result['pred_global_mean']:.3f} "
                          f"True={train_result['label_mean']:.3f}")
            
            # 检测坍缩
            if stage == 'learning':
                pred_l = train_result.get('pred_local_mean', 0.5)
                pred_g = train_result.get('pred_global_mean', 0.5)
                if pred_l > 0.85 or pred_l < 0.15:
                    log_str += "\n  ⚠️  WARNING: Local channel prediction collapsed!"
                if pred_g > 0.85 or pred_g < 0.15:
                    log_str += "\n  ⚠️  WARNING: Global channel prediction collapsed!"
        
        print(log_str)
        
        try:
            from sklearn.metrics import confusion_matrix
            y_pred = (y_score >= best_thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0,1]).ravel()
            print(f"  CM: tn={tn} fp={fp} fn={fn} tp={tp}")
        except Exception:
            pass

# 🔧 添加最小epoch限制
        can_save = epoch >= args.min_epoch_for_best
        
        if metrics['acc'] > best_acc:
            if can_save:
                best_acc = metrics['acc']
                best_state_acc = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_thresh_acc = best_thresh
                print(f"  → New best ACC: {best_acc:.4f} (epoch {epoch})")
            else:
                print(f"  → ACC improved to {metrics['acc']:.4f} but epoch < {args.min_epoch_for_best}, not saving")        
        if not np.isnan(metrics['auc']) and metrics['auc'] > best_auc:
            if can_save:
                best_auc = metrics['auc']
                best_state_auc = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_thresh_auc = best_thresh
                print(f"  → New best AUC: {best_auc:.4f} (epoch {epoch})")
            else:
                print(f"  → AUC improved to {metrics['auc']:.4f} but epoch < {args.min_epoch_for_best}, not saving")
        
        if not np.isnan(metrics['auc']):
            balanced_metric = (metrics['acc'] + metrics['auc']) / 2.0
        else:
            balanced_metric = metrics['acc']
        
        if balanced_metric > best_balanced:
            if can_save:
                best_balanced = balanced_metric
                best_state_balanced = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_thresh_balanced = best_thresh
                print(f"  → New best Balanced: {best_balanced:.4f} (epoch {epoch})")
            else:
                print(f"  → Balanced improved to {balanced_metric:.4f} but epoch < {args.min_epoch_for_best}, not saving")

        # 🔧 早停计数器更新 - 基于Balanced指标
        if balanced_metric > best_balanced_for_early_stop:
            best_balanced_for_early_stop = balanced_metric
            no_improve = 0
            print(f"  ✓ Balanced improved to {balanced_metric:.4f} (early stop counter reset)")
        else:
            no_improve += 1

        # 🆕 保存近期最佳模型
        if args.save_recent_best > 0 and can_save:
            # 保存当前epoch的模型状态
            current_model = {
                'epoch': epoch,
                'balanced': balanced_metric,
                'acc': metrics['acc'],
                'auc': metrics['auc'],
                'threshold': best_thresh,
                'state': {k: v.cpu().clone() for k, v in model.state_dict().items()}
            }
            
            # 插入到队列中（按balanced降序排序）
            insert_pos = len(recent_best_models)
            for i, m in enumerate(recent_best_models):
                if balanced_metric > m['balanced']:
                    insert_pos = i
                    break
            
            # 插入新模型
            recent_best_models_list = list(recent_best_models)
            recent_best_models_list.insert(insert_pos, current_model)
            
            # 只保留前N个
            recent_best_models_list = recent_best_models_list[:args.save_recent_best]
            
            # 更新队列
            recent_best_models.clear()
            recent_best_models.extend(recent_best_models_list)
            
            print(f"  → Recent best models queue updated (size: {len(recent_best_models)})")
        
        scheduler.step()

        # 🔧 改进的早停检查
        if epoch >= args.min_epochs and no_improve >= args.early_stop_patience:
            print(f"\n{'='*60}")
            print(f"EARLY STOPPING at epoch {epoch}")
            print(f"Reason: Balanced metric hasn't improved for {args.early_stop_patience} epochs")
            print(f"Best Balanced achieved: {best_balanced_for_early_stop:.4f}")
            print(f"Current - ACC: {best_acc:.4f}, AUC: {best_auc:.4f}, Balanced: {best_balanced:.4f}")
            print(f"{'='*60}\n")
            break
        elif no_improve >= args.early_stop_patience:
            print(f"  ⚠️  {args.early_stop_patience} epochs without Balanced improvement, "
                  f"but epoch < {args.min_epochs}, continuing training...")
    # Save models
    if args.save_model:
        base_path = args.save_model
        base_dir = os.path.dirname(base_path) or '.'
        base_name = os.path.splitext(os.path.basename(base_path))[0]
        
        os.makedirs(base_dir, exist_ok=True)
        
        model_paths = {
            'acc': os.path.join(base_dir, f"{base_name}_best_acc.pt"),
            'auc': os.path.join(base_dir, f"{base_name}_best_auc.pt"),
            'balanced': os.path.join(base_dir, f"{base_name}_best_balanced.pt")
        }
        
        model_info = {
            'acc': (best_state_acc, best_thresh_acc, best_acc),
            'auc': (best_state_auc, best_thresh_auc, best_auc),
            'balanced': (best_state_balanced, best_thresh_balanced, best_balanced)
        }
        
        print("\n" + "="*60)
        print("SAVING THREE BEST MODELS:")
        for model_type, (state, thresh, score) in model_info.items():
            if state is not None:
                torch.save({
                    'model_state': state,
                    'args': vars(args),
                    'best_thresh': thresh,
                    f'best_{model_type}': score
                }, model_paths[model_type])
                print(f"  {model_type.upper()}: {model_paths[model_type]}")
                print(f"    → Score: {score:.4f}, Threshold: {thresh:.3f}")
        print("="*60 + "\n")
        
        # 🆕 保存近期最佳模型
        if args.save_recent_best > 0 and len(recent_best_models) > 0:
            print("\n" + "="*60)
            print(f"SAVING TOP-{args.save_recent_best} RECENT BEST MODELS:")
            print("="*60)
            
            for rank, model_info in enumerate(recent_best_models, 1):
                save_path = os.path.join(base_dir, f"{base_name}_recent_best_rank{rank}_epoch{model_info['epoch']}.pt")
                torch.save({
                    'model_state': model_info['state'],
                    'args': vars(args),
                    'epoch': model_info['epoch'],
                    'balanced': model_info['balanced'],
                    'acc': model_info['acc'],
                    'auc': model_info['auc'],
                    'best_thresh': model_info['threshold']
                }, save_path)
                print(f"  Rank {rank} (Epoch {model_info['epoch']:3d}): {save_path}")
                print(f"    → Balanced: {model_info['balanced']:.4f}, ACC: {model_info['acc']:.4f}, "
                      f"AUC: {model_info['auc']:.4f}")
            print("="*60 + "\n")
    # Test
    if args.test_pt_dir:
        test_labels_map = load_labels_csv(args.test_labels_csv) if args.test_labels_csv else None
        ext_test_ds = PTGraphSequenceDataset(args.test_pt_dir, labels_map=test_labels_map, 
                                            pt_dir2=args.pt_dir2 if has_dual_channel else None)
        test_loader = DataLoader(ext_test_ds, batch_size=args.batch_size, 
                               shuffle=False, collate_fn=collate_fn)
        print("Using external test set")
    else:
        test_loader = test_loader_split
        print("Using split test set")

    print("\n" + "="*60)
    print("EVALUATING THREE MODELS ON TEST SET")
    print("="*60 + "\n")
    
    model_info = {
        'acc': (best_state_acc, best_thresh_acc, best_acc),
        'auc': (best_state_auc, best_thresh_auc, best_auc),
        'balanced': (best_state_balanced, best_thresh_balanced, best_balanced)
    }
    
    test_results = {}
    
    for model_type in ['acc', 'auc', 'balanced']:
        print(f"{'='*60}")
        print(f"Evaluating {model_type.upper()} model")
        print(f"{'='*60}")
        
        state, thresh, val_score = model_info[model_type]
        if state is None:
            print(f"No {model_type} model saved, skipping...")
            continue
        
        model.load_state_dict(state)
        model.to(device)
        
        y_val, y_score_val, _ = eval_epoch(model, val_loader, device, has_dual_channel)
        val_metrics = compute_metrics(y_val, y_score_val, thresh=thresh)
        print(f"[VAL {model_type.upper()}] metrics (thresh={thresh:.3f}):")
        print(f"  ACC: {val_metrics['acc']:.4f}, AUC: {val_metrics['auc']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, Balanced: {val_metrics['bal']:.4f}")
        
        y_test, y_score_test, files_test = eval_epoch(model, test_loader, device, has_dual_channel)
        test_metrics = compute_metrics(y_test, y_score_test, thresh=thresh)
        
        print(f"[TEST {model_type.upper()}] metrics (thresh={thresh:.3f}):")
        print(f"  ACC: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}, "
              f"F1: {test_metrics['f1']:.4f}, Balanced: {test_metrics['bal']:.4f}")
        print(f"  Sensitivity: {test_metrics['sen']:.4f}, Specificity: {test_metrics['spe']:.4f}")
        
        try:
            from sklearn.metrics import confusion_matrix
            y_pred = (y_score_test >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
            print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        except Exception:
            pass
        
        print()
        
        test_results[model_type] = {
            'metrics': test_metrics,
            'predictions': (y_test, y_score_test, files_test),
            'threshold': thresh
        }
        
        if args.save_model:
            out_csv = args.test_out_csv or os.path.join(base_dir, f'test_predictions_{model_type}.csv')
            os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
            
            import csv
            with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['file', 'score', 'pred', 'label'])
                for fn, sc, lbl in zip(files_test, y_score_test, y_test):
                    pred = int(sc >= thresh)
                    w.writerow([fn, f"{float(sc):.6f}", pred, int(lbl)])
            print(f"Saved {model_type.upper()} predictions to {out_csv}\n")
    
    print("="*60)
    print("SUMMARY: TEST SET COMPARISON")
    print("="*60)
    print(f"{'Model':<12} {'ACC':<8} {'AUC':<8} {'F1':<8} {'Balanced':<10} {'Sens':<8} {'Spec':<8}")
    print("-"*60)
    for model_type in ['acc', 'auc', 'balanced']:
        if model_type in test_results:
            m = test_results[model_type]['metrics']
            print(f"{model_type.upper():<12} {m['acc']:<8.4f} {m['auc']:<8.4f} {m['f1']:<8.4f} "
                  f"{m['bal']:<10.4f} {m['sen']:<8.4f} {m['spe']:<8.4f}")
    print("="*60)
    # 评估近期最佳模型
    if args.save_recent_best > 0 and len(recent_best_models) > 0:
        print("\n" + "="*60)
        print(f"EVALUATING TOP-{args.save_recent_best} RECENT BEST MODELS ON TEST SET")
        print("="*60 + "\n")
        
        recent_test_results = {}
        
        for rank, model_info in enumerate(recent_best_models, 1):
            print(f"{'='*60}")
            print(f"Evaluating Recent Best Rank {rank} (Epoch {model_info['epoch']})")
            print(f"{'='*60}")
            
            # 加载模型
            model.load_state_dict(model_info['state'])
            model.to(device)
            thresh = model_info['threshold']
            
            # 验证集评估
            y_val, y_score_val, _ = eval_epoch(model, val_loader, device, has_dual_channel)
            val_metrics = compute_metrics(y_val, y_score_val, thresh=thresh)
            print(f"[VAL Rank {rank}] metrics (thresh={thresh:.3f}):")
            print(f"  ACC: {val_metrics['acc']:.4f}, AUC: {val_metrics['auc']:.4f}, "
                  f"F1: {val_metrics['f1']:.4f}, Balanced: {val_metrics['bal']:.4f}")
            
            # 测试集评估
            y_test, y_score_test, files_test = eval_epoch(model, test_loader, device, has_dual_channel)
            test_metrics = compute_metrics(y_test, y_score_test, thresh=thresh)
            
            print(f"[TEST Rank {rank}] metrics (thresh={thresh:.3f}):")
            print(f"  ACC: {test_metrics['acc']:.4f}, AUC: {test_metrics['auc']:.4f}, "
                  f"F1: {test_metrics['f1']:.4f}, Balanced: {test_metrics['bal']:.4f}")
            print(f"  Sensitivity: {test_metrics['sen']:.4f}, Specificity: {test_metrics['spe']:.4f}")
            
            try:
                from sklearn.metrics import confusion_matrix
                y_pred = (y_score_test >= thresh).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0,1]).ravel()
                print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            except Exception:
                pass
            
            print()
            
            recent_test_results[f'rank{rank}'] = {
                'metrics': test_metrics,
                'predictions': (y_test, y_score_test, files_test),
                'threshold': thresh,
                'epoch': model_info['epoch']
            }
            
            # 保存预测结果
            if args.save_model:
                out_csv = os.path.join(base_dir, f'test_predictions_recent_rank{rank}_epoch{model_info["epoch"]}.csv')
                os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
                
                import csv
                with open(out_csv, 'w', newline='', encoding='utf-8') as f:
                    w = csv.writer(f)
                    w.writerow(['file', 'score', 'pred', 'label'])
                    for fn, sc, lbl in zip(files_test, y_score_test, y_test):
                        pred = int(sc >= thresh)
                        w.writerow([fn, f"{float(sc):.6f}", pred, int(lbl)])
                print(f"Saved Rank {rank} predictions to {out_csv}\n")
        
        # 汇总对比
        print("="*60)
        print("SUMMARY: RECENT BEST MODELS TEST SET COMPARISON")
        print("="*60)
        print(f"{'Model':<20} {'Epoch':<7} {'ACC':<8} {'AUC':<8} {'F1':<8} {'Balanced':<10} {'Sens':<8} {'Spec':<8}")
        print("-"*80)
        for rank_key in sorted(recent_test_results.keys(), key=lambda x: int(x.replace('rank', ''))):
            result = recent_test_results[rank_key]
            m = result['metrics']
            epoch = result['epoch']
            rank_num = rank_key.replace('rank', '')
            print(f"Rank {rank_num:<15} {epoch:<7} {m['acc']:<8.4f} {m['auc']:<8.4f} {m['f1']:<8.4f} "
                  f"{m['bal']:<10.4f} {m['sen']:<8.4f} {m['spe']:<8.4f}")
        print("="*80)
    
    print(f"\nTotal training time: {time.time()-start_time:.1f}s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_dir", required=True, help="Path to channel 1 (local) .pt data directory")
    parser.add_argument("--pt_dir2", default=None, help="Path to channel 2 (global) .pt data directory")
    parser.add_argument("--labels_csv", default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--use_focal", action='store_true')
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_model", default=None)
    parser.add_argument("--test_pt_dir", default=None)
    parser.add_argument("--test_labels_csv", default=None)
    parser.add_argument("--test_out_csv", default=None)
    parser.add_argument("--enable_memory_write", action='store_true')
    parser.add_argument("--comm_vocab", type=int, default=128)
    parser.add_argument("--win_emb", type=int, default=128)
    parser.add_argument("--tf_layers", type=int, default=3)
    parser.add_argument("--tf_heads", type=int, default=4)
    parser.add_argument("--gnn_layers", type=int, default=3, help="Number of GNN layers for global channel")
    parser.add_argument("--gnn_hidden", type=int, default=64, help="Hidden dimension for GNN")
    parser.add_argument("--mem_slots", type=int, default=32)
    parser.add_argument("--mem_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--init_bias_target", type=float, default=None)
    parser.add_argument("--force_oversample", action='store_true')
    parser.add_argument("--thresh_opt", choices=['accuracy','f1','balanced','youden'], default='balanced')
    parser.add_argument("--min_epoch_for_best", type=int, default=1, 
                    help="Minimum epoch before saving best models")
    parser.add_argument("--save_recent_best", type=int, default=10,
                        help="Save top-N recent best models (0 to disable)")
    parser.add_argument("--min_epochs", type=int, default=30,
                    help="Minimum epochs before early stopping can trigger")
    parser.add_argument("--early_stop_patience", type=int, default=25,
                        help="Early stopping patience (epochs without improvement)")
    # 🆕 GAT相关参数
    parser.add_argument("--gnn_type", choices=['gcn', 'gat'], default='gcn',
                       help="GNN type for global channel: gcn or gat")
    parser.add_argument("--gat_heads", type=int, default=4,
                       help="Number of attention heads for GAT")
    
    # 🆕 动态loss相关参数
    parser.add_argument("--use_dynamic_loss", action='store_true',
                       help="Use staged dynamic loss for dual-channel fusion")
    parser.add_argument("--warmup_epochs", type=int, default=20,
                       help="Number of warmup epochs (fixed weights 0.5:0.5)")
    parser.add_argument("--learning_epochs", type=int, default=30,
                       help="Number of learning epochs (learn weights with constraints)")
    parser.add_argument("--min_weight", type=float, default=0.25,
                       help="Minimum weight for each channel (prevent collapse)")
    parser.add_argument("--balance_penalty", type=float, default=1.0,
                       help="Strength of balance penalty in dynamic loss")
    args = parser.parse_args()
    main(args)
