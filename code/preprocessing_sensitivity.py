#!/usr/bin/env python3
"""preprocessing_sensitivity.py — fixed vs recomputed preprocessing under deletion.

The primary pipeline refits PCA (and, for the value layer, StandardScaler) on each
reduced matrix. That is a legitimate "refit the whole pipeline" protocol, but it
conflates standardization instability with ordination instability, especially for
an extreme row such as B154. This script repeats the paired block/matched design
with preprocessing fixed at the full-data centering (and, for the value layer,
full-data z-scoring) while still refitting the loadings on the reduced rows.

Reads: frozen layers via experiment_C1.load(); reproduces the first scattered draw
with the same seed rule as experiment_H_volumematch*.py.
Writes: results/preprocessing_sensitivity.csv

Run: python3 code/preprocessing_sensitivity.py
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import pandas as pd
from scipy.spatial import procrustes
from scipy.spatial.distance import pdist

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import load  # noqa: E402


def scores_fixed(M: np.ndarray, mu: np.ndarray, sd: np.ndarray | None,
                 ncomp: int = 2) -> np.ndarray:
    Z = M - mu if sd is None else (M - mu) / sd
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, :ncomp] * S[:ncomp]


def drift_median(S_full: np.ndarray, S_red: np.ndarray, keep: np.ndarray) -> float:
    a, b, _ = procrustes(S_full[keep], S_red)
    return float(np.median(np.linalg.norm(b - a, axis=1)))


def normalise(S: np.ndarray) -> np.ndarray:
    S = np.asarray(S, float)
    S = S - S.mean(axis=0, keepdims=True)
    n = np.linalg.norm(S)
    return S / n if n else S


def run_variant(name: str, M: np.ndarray, papers: np.ndarray, sd: bool):
    mu = M.mean(axis=0)
    scale = M.std(axis=0, ddof=0) if sd else None
    S_full = scores_fixed(M, mu, scale)
    denom = float(np.median(pdist(normalise(S_full))))
    n = len(M)
    uniq = sorted(set(papers))
    rows = []
    for p in uniq:
        mask = papers == p
        n_p = int(mask.sum())
        keep = ~mask
        S_red = scores_fixed(M[keep], mu, scale)
        d_block = drift_median(S_full, S_red, keep)

        rng = np.random.default_rng(SEED + zlib.crc32(str(p).encode()) % 100000)
        drop = rng.choice(n, size=n_p, replace=False)
        keep2 = np.ones(n, bool)
        keep2[drop] = False
        S_red2 = scores_fixed(M[keep2], mu, scale)
        d_match = drift_median(S_full, S_red2, keep2)

        rows.append(dict(variant=name, paper=p, n_rows=n_p,
                         block_drift=d_block, matched_drift=d_match,
                         ratio=d_block / d_match,
                         block_pct_interrow=100.0 * d_block / denom,
                         matched_pct_interrow=100.0 * d_match / denom,
                         denom_interrow=denom))
    return pd.DataFrame(rows)


def main() -> None:
    X, _rho, paper, _params, Xv, keysv, _rho_v = load()
    paper = np.asarray(paper)
    vpaper = np.asarray([k[0] for k in keysv], dtype=object)

    pres = run_variant("presence_pca", X, paper, sd=False)
    val = run_variant("std_value_pca", Xv, vpaper, sd=True)
    out = pd.concat([pres, val], ignore_index=True)

    path = os.path.join(RESULTS, "preprocessing_sensitivity.csv")
    out.to_csv(path, index=False, float_format="%.6g")

    # Layer summary with a 10,000-resample bootstrap CI for the median paired ratio.
    rng = np.random.default_rng(SEED)
    summ = []
    for name, d in (("presence_pca", pres), ("std_value_pca", val)):
        r = d["ratio"].to_numpy(float)
        boot = np.empty(10000)
        for i in range(10000):
            idx = rng.integers(0, len(r), len(r))
            boot[i] = np.median(r[idx])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        b154 = d[d["paper"] == "B154"]
        summ.append(dict(
            variant=name, n_papers=len(d),
            median_paired_ratio=float(np.median(r)),
            paired_ci_lo=float(lo), paired_ci_hi=float(hi),
            ratio_of_medians=float(np.median(d["block_drift"]) / np.median(d["matched_drift"])),
            b154_block_pct=float(b154["block_pct_interrow"].iloc[0]) if len(b154) else np.nan,
        ))
    summ = pd.DataFrame(summ)
    spath = os.path.join(RESULTS, "preprocessing_sensitivity_summary.csv")
    summ.to_csv(spath, index=False, float_format="%.6g")

    print("== fixed preprocessing: layer summary ==")
    print(summ.to_string(index=False))
    for name, d in (("presence_pca", pres), ("std_value_pca", val)):
        for b in ("B154", "B063"):
            r = d[d["paper"] == b]
            if len(r):
                r = r.iloc[0]
                print(f"{name} {b}: block {r['block_drift']:.3e} "
                      f"matched {r['matched_drift']:.3e} ratio {r['ratio']:.3f} "
                      f"block % inter-row {r['block_pct_interrow']:.2f}")
    print(f"\nwrote {os.path.relpath(path, ROOT)} and {os.path.relpath(spath, ROOT)}")


if __name__ == "__main__":
    main()
