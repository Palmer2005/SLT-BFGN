import nibabel as nib
import numpy as np
from scipy import ndimage
from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from sklearn.covariance import GraphicalLassoCV
from sklearn.covariance import LedoitWolf
import networkx as nx
import os
import torch
import warnings
import random
from pathlib import Path
import re

# Fetch atlas once
new_cache_dir = r'e:\图序列\model\nilearn_atlas_cache'
os.makedirs(new_cache_dir, exist_ok=True)
aal = datasets.fetch_atlas_aal(data_dir=new_cache_dir)
atlas_filename = aal.maps
atlas_labels = aal.labels

estimator = GraphicalLassoCV(cv=3)

def compute_partial_robust(window_data, estimator=None, ridge=1e-3):
    if np.isnan(window_data).any():
        col_mean = np.nanmean(window_data, axis=0)
        inds = np.where(np.isnan(window_data))
        window_data[inds] = np.take(col_mean, inds[1])
    n_samp, n_feat = window_data.shape
    var = window_data.var(axis=0)
    tiny_var_idx = np.where(var < 1e-8)[0]
    if tiny_var_idx.size > 0:
        window_data[:, tiny_var_idx] += np.random.RandomState(42).normal(scale=1e-6, size=(n_samp, tiny_var_idx.size))

    precision = None
    if n_samp <= n_feat:
        try:
            lw = LedoitWolf().fit(window_data)
            cov = lw.covariance_
            cov += ridge * np.eye(n_feat)
            precision = np.linalg.pinv(cov)
        except Exception:
            cov = np.cov(window_data, rowvar=False)
            cov += ridge * np.eye(n_feat)
            precision = np.linalg.pinv(cov)
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                estimator.fit(window_data)
            precision = estimator.precision_
            if not np.isfinite(precision).all():
                raise FloatingPointError("precision contains non-finite values")
        except Exception:
            try:
                lw = LedoitWolf().fit(window_data)
                cov = lw.covariance_
                cov += ridge * np.eye(n_feat)
                precision = np.linalg.pinv(cov)
            except Exception:
                cov = np.cov(window_data, rowvar=False)
                cov += ridge * np.eye(n_feat)
                precision = np.linalg.pinv(cov)

    denom = np.sqrt(np.abs(np.diag(precision)) + 1e-12)
    partial = - precision / np.outer(denom, denom)
    np.fill_diagonal(partial, 0.0)
    return partial

def process_one(func_filename: str, out_pt_path: str, window_size: int = 10, global_percentile: float = 50.0):
    # extract time-series
    masker = NiftiLabelsMasker(
        labels_img=atlas_filename,
        standardize=True,
        memory='nilearn_cache',
        verbose=0,
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=2.0,
    )
    try:
        time_series = masker.fit_transform(func_filename, confounds=None)
    except ValueError as e:
        msg = str(e).lower()
        if 'padlen' in msg or 'must be greater' in msg:
            # Retry without temporal filtering for short time-series
            masker_nofilt = NiftiLabelsMasker(
                labels_img=atlas_filename,
                standardize=True,
                memory='nilearn_cache',
                verbose=0,
                detrend=True,
                low_pass=None,
                high_pass=None,
                t_r=2.0,
            )
            time_series = masker_nofilt.fit_transform(func_filename, confounds=None)
            print(f"Warning: time-series too short for bandpass filter — disabled filtering for: {func_filename}")
        else:
            raise

    # 如果返回是一维（单个体积、无时间维度），跳过该样本（不生成 .pt）
    if getattr(time_series, "ndim", None) == 1:
        print(f"Skipping (single-volume / no time points): {func_filename}")
        return

    n_time_points, n_rois = time_series.shape

    step_size = window_size
    num_windows = n_time_points // window_size
    window_starts = np.arange(0, num_windows * window_size, step_size)

    # compute partials (robust)
    np.random.seed(42)
    random.seed(42)
    partial_matrices = []
    abs_vals_all = []
    for start in window_starts:
        wd = time_series[start:start+window_size, :].copy()
        partial = compute_partial_robust(wd, estimator=estimator, ridge=1e-3)
        np.fill_diagonal(partial, 0.0)
        partial_matrices.append(partial)
        tri_idx = np.triu_indices_from(partial, k=1)
        vals = np.abs(partial[tri_idx])
        vals = vals[~np.isnan(vals)]
        if vals.size > 0:
            abs_vals_all.append(vals)
    abs_vals_all = np.concatenate(abs_vals_all) if len(abs_vals_all) > 0 else np.array([0.0])
    global_thr = float(np.percentile(abs_vals_all, global_percentile))

    # build sparse adjacency sequence
    adj_matrices = []
    community_sequence = []
    for partial in partial_matrices:
        mask = np.abs(partial) >= global_thr
        W = partial * mask
        W[np.abs(W) < 1e-12] = 0.0
        W = (W + W.T) / 2.0
        adj_matrices.append(W.astype(np.float32))

        G = nx.Graph()
        G.add_nodes_from(range(n_rois))
        iu = np.triu_indices(n_rois, k=1)
        for i, j in zip(iu[0], iu[1]):
            w = W[i, j]
            if w != 0.0 and not np.isnan(w):
                G.add_edge(int(i), int(j), weight=float(w))
        if G.number_of_edges() > 0:
            communities = nx.community.greedy_modularity_communities(G, weight='weight')
            partition = {node: i for i, comm in enumerate(communities) for node in comm}
        else:
            partition = {i: -1 for i in range(n_rois)}
        for i in range(n_rois):
            partition.setdefault(i, -1)
        community_sequence.append([partition[i] for i in range(n_rois)])

    adj_tensor = torch.from_numpy(np.stack(adj_matrices, axis=0)) if len(adj_matrices) > 0 else torch.zeros((0, n_rois, n_rois), dtype=torch.float32)

    # make tensors for community_sequence and window_starts (avoid NameError)
    community_tensor = torch.tensor(np.array(community_sequence, dtype=np.int64)) if len(community_sequence) > 0 else torch.zeros((0, n_rois), dtype=torch.int64)
    window_starts_tensor = torch.tensor(window_starts, dtype=torch.int64)

    # sanitize adjacency: replace non-finite with 0, enforce symmetry and dtype
    if adj_tensor.numel() > 0:
        adj_np = adj_tensor.numpy()
        adj_np[~np.isfinite(adj_np)] = 0.0
        # enforce symmetry
        adj_np = 0.5 * (adj_np + np.transpose(adj_np, (0, 2, 1)))
        adj_tensor = torch.from_numpy(adj_np.astype(np.float32))

    # --- determine node names and align coords robustly ---
    # build node_names in atlas label order (skip background/0) and align atlas_centroids if possible
    try:
        lab_img = nib.load(atlas_filename).get_fdata().astype(int)
        label_vals = [int(v) for v in np.unique(lab_img) if int(v) != 0]
    except Exception:
        # fallback: assume sequential labels 1..n_rois
        label_vals = list(range(1, n_rois + 1))

    node_names_aligned = []
    coords_aligned = []
    for v in label_vals:
        # atlas_labels from nilearn typically list names in the same order as label indices starting at 1.
        # use v-1 to index atlas_labels safely; fallback if out of range.
        if atlas_labels is not None and (v - 1) < len(atlas_labels):
            node_names_aligned.append(atlas_labels[v - 1])
        else:
            node_names_aligned.append(f'roi_{v}')
        # coords: atlas_centroids is 0-based for label 1..max -> index v-1
        if 'atlas_centroids' in globals() and isinstance(globals()['atlas_centroids'], np.ndarray) and (v - 1) < globals()['atlas_centroids'].shape[0]:
            coords_aligned.append(globals()['atlas_centroids'][v - 1].tolist())
        else:
            coords_aligned.append([0.0, 0.0, 0.0])

    # If atlas label list length != masker ROI count, try to detect masker ordering:
    if len(node_names_aligned) != n_rois:
        # fallback: generate generic names and attempt to trim/expand coords to match n_rois
        node_names = [f'roi_{i}' for i in range(n_rois)]
        coords_final = torch.zeros((n_rois, 3), dtype=torch.float32)
        # if atlas_centroids exists and has at least n_rois rows, use its first n_rois
        if 'atlas_centroids' in globals() and isinstance(globals()['atlas_centroids'], np.ndarray) and globals()['atlas_centroids'].shape[0] >= n_rois:
            coords_final = torch.tensor(globals()['atlas_centroids'][:n_rois, :], dtype=torch.float32)
        else:
            # leave zeros; caller should inspect these files
            pass
    else:
        node_names = node_names_aligned
        coords_final = torch.tensor(np.array(coords_aligned, dtype=np.float32), dtype=torch.float32)

    # final sanity checks
    if coords_final.shape != (n_rois, 3):
        coords_final = coords_final.reshape((-1, 3))[:n_rois]
        if coords_final.shape != (n_rois, 3):
            coords_final = torch.zeros((n_rois, 3), dtype=torch.float32)

    os.makedirs(os.path.dirname(out_pt_path), exist_ok=True)
    torch.save({
        'adjacency': adj_tensor,
        'coords': coords_final,
        'node_names': node_names,
        'community_sequence': community_tensor,
        'window_starts': window_starts_tensor,
        'global_threshold': float(global_thr),
        't_r': float(masker.t_r) if hasattr(masker, 't_r') and masker.t_r is not None else None,
    }, out_pt_path)
    print(f"Saved: {out_pt_path}")

# atlas_filename 已在你的脚本中有
atlas_img = nib.load(atlas_filename)
atlas_data = atlas_img.get_fdata().astype(int)
affine = atlas_img.affine

coords = []
max_label = int(atlas_data.max())
for lab in range(1, max_label + 1):
    mask = (atlas_data == lab)
    if mask.sum() == 0:
        coords.append([0.0, 0.0, 0.0])
        continue
    com_vox = ndimage.center_of_mass(mask)      # voxel-space centroid
    com_mni = nib.affines.apply_affine(affine, com_vox)  # convert to world/MNI coords
    coords.append(list(com_mni))
coords = np.array(coords, dtype=np.float32)   # shape (n_rois, 3)
# expose atlas centroids for process_one
atlas_centroids = coords.copy()

if __name__ == '__main__':
    # batch input root (change if needed)
    input_root = Path(r'F:\项目存档\home\brain\raw_data')
    pattern = '**/NIfTI/*rest*.nii*'
    out_base = Path(r'E:\brain_data\adhd_5_0.5')  # will create Pittsburgh/<session> under this

    # collect files and build a canonical mapping (subject, session) -> file
    files = sorted(input_root.glob(pattern))
    mapping = {}  # (subject, session) -> Path
    for f in files:
        try:
            rel = f.relative_to(input_root)
            rel_parts = list(rel.parts)
        except Exception:
            rel_parts = list(f.parts)
        # session detection
        session = next((p for p in rel_parts if p in ('0', '1')), None)
        if session is None:
            session = next((p.name for p in f.parents if p.name in ('0', '1')), '0')
        # subject detection
        subject = next((p for p in rel_parts if re.match(r'^[A-Za-z]+_\d+_\d+$', p)), None)
        if subject is None:
            subject = next((p.name for p in f.parents if re.match(r'^[A-Za-z]+_\d+_\d+$', p)), f.stem)
        key = (subject, session)
        # keep first (lexicographically smallest) path for each (subject,session)
        if key not in mapping or str(f) < str(mapping[key]):
            mapping[key] = f

    # ensure output directories exist and process one file per (subject,session)
    for (subject, session), fpath in sorted(mapping.items()):
        out_dir = out_base / '~' / session
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = str(out_dir / (subject + '.pt'))
        # skip if already exists to preserve one-to-one mapping and avoid accidental overwrite
        if os.path.exists(out_path):
            print(f"Skipping (exists): {out_path}")
            continue
        print(f"Processing subject={subject}, session={session}, input={fpath}")
        process_one(str(fpath), out_path, window_size=5, global_percentile=50.0)

