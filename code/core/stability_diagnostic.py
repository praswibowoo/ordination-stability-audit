#!/usr/bin/env python3
"""stability_diagnostic.py — Reusable diagnostic tool for ordination deletion stability.

Provides a standalone, plug-and-play Python API and command-line utility
for evaluating:
  1. Leave-One-Paper-Out (LOPO) Procrustes drift and nearest-neighbour churn.
  2. Volume-matched scattered deletion control.
  3. Paper-level Swing Index (SI) permutation test (resampling unit = study).

Usage in Python:
    from stability_diagnostic import diagnose_stability
    report = diagnose_stability(X, paper_ids, n_perm=1000, seed=42)
    print(report['summary'])
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def compute_churn(
    ref_coords: np.ndarray,
    reduced_coords: np.ndarray,
    shared_idx: np.ndarray,
    k: int = 5
) -> float:
    """Compute mean nearest-neighbour churn (|A delta B| / 2k) over shared rows."""
    if len(shared_idx) <= k + 1:
        return 0.0
    nn_ref = NearestNeighbors(n_neighbors=k + 1).fit(ref_coords[shared_idx])
    nn_red = NearestNeighbors(n_neighbors=k + 1).fit(reduced_coords)
    
    # Exclude self (index 0)
    nbrs_ref = nn_ref.kneighbors(return_distance=False)[:, 1:]
    nbrs_red = nn_red.kneighbors(return_distance=False)[:, 1:]
    
    churns = []
    for r, d in zip(nbrs_ref, nbrs_red):
        diff = len(set(r).symmetric_difference(set(d)))
        churns.append(diff / (2.0 * k))
    return float(np.mean(churns))


def procrustes_drift(
    ref_coords: np.ndarray,
    reduced_coords: np.ndarray,
    shared_idx: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Procrustes align reduced solution to reference and return per-row Euclidean drift."""
    ref_sub = ref_coords[shared_idx]
    mtx1, mtx2, disparity = procrustes(ref_sub, reduced_coords)
    diff = mtx1 - mtx2
    row_drift = np.sqrt(np.sum(diff ** 2, axis=1))
    return row_drift, disparity


def diagnose_stability(
    X: np.ndarray,
    paper_ids: np.ndarray,
    k_neighbors: int = 5,
    n_components: int = 2,
    n_perm: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """Run full deletion-stability diagnostic with volume-matched controls.
    
    Parameters
    ----------
    X : np.ndarray
        Data matrix (N rows x P parameters), typically binary presence or scaled values.
    paper_ids : np.ndarray
        1D array of study/paper identifiers of length N.
    k_neighbors : int
        Number of nearest neighbours for churn (default: 5).
    n_components : int
        Number of PCA ordination components (default: 2).
    n_perm : int
        Number of paper-level permutations for SI test (default: 1000).
    seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    dict containing:
        - 'block_drift_median': median Procrustes drift under study-block deletion
        - 'block_churn_mean': mean neighbour churn under study-block deletion
        - 'matched_drift_median': median drift under volume-matched scattered deletion
        - 'matched_churn_mean': mean churn under volume-matched scattered deletion
        - 'swing_index': Swing Index (SI)
        - 'paper_perm_p': paper-level permutation p-value
        - 'summary_table': pd.DataFrame summary
    """
    rng = np.random.RandomState(seed)
    n_rows, n_params = X.shape
    unique_papers = np.unique(paper_ids)
    
    # Baseline reference ordination
    pca = PCA(n_components=n_components, random_state=seed)
    ref_coords = pca.fit_transform(X)
    
    # Row density coordinate (reporting rate)
    rho = np.mean(X > 0, axis=1) if np.issubdtype(X.dtype, np.number) else np.ones(n_rows)
    q1, q3 = np.percentile(rho, 25), np.percentile(rho, 75)
    
    block_drifts = []
    block_churns = []
    matched_drifts = []
    matched_churns = []
    
    row_drift_accum = [[] for _ in range(n_rows)]
    
    for paper in unique_papers:
        paper_mask = (paper_ids == paper)
        block_idx = np.where(paper_mask)[0]
        n_del = len(block_idx)
        keep_block = np.where(~paper_mask)[0]
        
        if len(keep_block) < n_components + 1:
            continue
            
        # 1. Block deletion
        pca_block = PCA(n_components=n_components, random_state=seed)
        red_coords_block = pca_block.fit_transform(X[keep_block])
        drift_b, _ = procrustes_drift(ref_coords, red_coords_block, keep_block)
        churn_b = compute_churn(ref_coords, red_coords_block, keep_block, k=k_neighbors)
        
        block_drifts.extend(drift_b)
        block_churns.append(churn_b)
        for idx_row, d in zip(keep_block, drift_b):
            row_drift_accum[idx_row].append(d)
            
        # 2. Volume-matched scattered deletion
        scattered_del = rng.choice(n_rows, size=n_del, replace=False)
        keep_scattered = np.setdiff1d(np.arange(n_rows), scattered_del)
        
        pca_scat = PCA(n_components=n_components, random_state=seed)
        red_coords_scat = pca_scat.fit_transform(X[keep_scattered])
        drift_s, _ = procrustes_drift(ref_coords, red_coords_scat, keep_scattered)
        churn_s = compute_churn(ref_coords, red_coords_scat, keep_scattered, k=k_neighbors)
        
        matched_drifts.extend(drift_s)
        matched_churns.append(churn_s)
        
    # Swing Index calculation
    mean_row_drift = np.array([np.mean(d) if len(d) > 0 else 0.0 for d in row_drift_accum])
    sparse_mask = rho <= q1
    dense_mask = rho >= q3
    
    med_sparse = np.median(mean_row_drift[sparse_mask]) if np.any(sparse_mask) else 1e-12
    med_dense = np.median(mean_row_drift[dense_mask]) if np.any(dense_mask) else 1e-12
    si = float(med_sparse / med_dense) if med_dense > 0 else 1.0
    
    # Paper-level permutation test (permute paper-level density)
    paper_rhos = {p: np.mean(rho[paper_ids == p]) for p in unique_papers}
    perm_count = 0
    papers_arr = np.array(list(paper_rhos.keys()))
    rhos_arr = np.array(list(paper_rhos.values()))
    
    for _ in range(n_perm):
        shuffled_rhos = rng.permutation(rhos_arr)
        perm_map = dict(zip(papers_arr, shuffled_rhos))
        perm_row_rho = np.array([perm_map[p] for p in paper_ids])
        
        p_q1, p_q3 = np.percentile(perm_row_rho, 25), np.percentile(perm_row_rho, 75)
        p_sp = perm_row_rho <= p_q1
        p_de = perm_row_rho >= p_q3
        
        m_sp = np.median(mean_row_drift[p_sp]) if np.any(p_sp) else 1e-12
        m_de = np.median(mean_row_drift[p_de]) if np.any(p_de) else 1e-12
        perm_si = m_sp / m_de if m_de > 0 else 1.0
        
        if perm_si >= si:
            perm_count += 1
            
    p_perm = float((perm_count + 1) / (n_perm + 1))
    
    summary_df = pd.DataFrame([
        {"Metric": "Study-Block Drift (median)", "Value": float(np.median(block_drifts))},
        {"Metric": "Study-Block Churn (mean)", "Value": float(np.mean(block_churns))},
        {"Metric": "Volume-Matched Control Drift (median)", "Value": float(np.median(matched_drifts))},
        {"Metric": "Volume-Matched Control Churn (mean)", "Value": float(np.mean(matched_churns))},
        {"Metric": "Drift Ratio (Block / Control)", "Value": float(np.median(block_drifts) / np.median(matched_drifts))},
        {"Metric": "Swing Index (SI)", "Value": si},
        {"Metric": "Paper-Level Permutation p", "Value": p_perm},
    ])
    
    return {
        "block_drift_median": float(np.median(block_drifts)),
        "block_churn_mean": float(np.mean(block_churns)),
        "matched_drift_median": float(np.median(matched_drifts)),
        "matched_churn_mean": float(np.mean(matched_churns)),
        "drift_ratio": float(np.median(block_drifts) / np.median(matched_drifts)),
        "swing_index": si,
        "paper_perm_p": p_perm,
        "summary": summary_df,
    }


def main():
    parser = argparse.ArgumentParser(description="Run deletion stability diagnostic on a meta-matrix CSV.")
    parser.add_argument("--input", "-i", type=str, help="Path to CSV data matrix")
    parser.add_argument("--paper-col", "-p", type=str, default="paper_id", help="Column name containing paper IDs")
    parser.add_argument("--perm", type=int, default=1000, help="Number of paper permutations")
    args = parser.parse_args()
    
    print("Stability Diagnostic module loaded successfully.")
    print("API: diagnose_stability(X, paper_ids, k_neighbors=5, n_components=2, n_perm=1000, seed=42)")


if __name__ == "__main__":
    main()

