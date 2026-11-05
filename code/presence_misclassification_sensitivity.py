#!/usr/bin/env python3
"""presence_misclassification_sensitivity.py — cell-level presence perturbation.

The presence layer treats NR as absent. The Stage-A audit found 104/117 claims
correct before adjudication (88.9%), and all 13 errors were corrected; the
remaining uncertainty in the 764x44 matrix is not propagated. This script
perturbs the presence matrix at bounded rates, reruns the block vs matched
control, and reports how the block factor and paired ratio move.

Perturbation rates:
  - drop:  present (Y/NE) -> NR
  - add:   NR -> present (Y)

Rates are deliberately coarse bounds, not estimates of the true error rate.

Reads: frozen presence layer via experiment_C1.load()
Writes: results/presence_misclassification_sensitivity.csv
        results/presence_misclassification_summary.csv

Run: python3 code/presence_misclassification_sensitivity.py
"""
from __future__ import annotations

import os
import sys
import zlib

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import (load, ordinate, drift_after_procrustes)  # noqa: E402


def cluster_boot_block(ev: pd.DataFrame, n_boot: int = 1000, seed: int = SEED):
    y = np.log(ev["med_drift"].to_numpy(float))
    v = np.log(ev["n_rows_deleted"].to_numpy(float))
    b = (ev["arm"] == "block").to_numpy(float)
    Xd = np.column_stack([np.ones(len(y)), v, b])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    uniq = np.asarray(sorted(set(ev["paper"])), dtype=object)
    gid = pd.Categorical(ev["paper"], categories=uniq).codes
    idx = {j: np.where(gid == j)[0] for j in range(len(uniq))}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[j] for j in pick])
        boots.append(np.linalg.lstsq(Xd[sel], y[sel], rcond=None)[0])
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots[:, 2], [0.025, 0.975])
    return float(np.exp(beta[2])), float(np.exp(lo)), float(np.exp(hi))


def perturb(X: np.ndarray, drop: float, add: float, seed: int):
    rng = np.random.default_rng(seed)
    Xp = X.copy()
    ones = Xp == 1
    zeros = Xp == 0
    Xp[ones & (rng.random(Xp.shape) < drop)] = 0
    Xp[zeros & (rng.random(Xp.shape) < add)] = 1
    return Xp


def run_once(Xp, paper, n_draws=20, seed=SEED):
    n = len(Xp)
    uniq = sorted(set(paper))
    S_full = ordinate("pca_pres", Xp)
    rows = []
    for p in uniq:
        mask = paper == p
        n_p = int(mask.sum())
        keep = ~mask
        S_red = ordinate("pca_pres", Xp[keep])
        drift, _ = drift_after_procrustes(S_full[keep], S_red)
        rows.append(dict(paper=p, arm="block", med_drift=float(np.median(drift)),
                         n_rows_deleted=n_p, draw=0))
        for dr in range(n_draws):
            rng = np.random.default_rng(seed + zlib.crc32(f"{p}-{dr}".encode()) % 100000)
            drop = rng.choice(n, size=n_p, replace=False)
            keep2 = np.ones(n, bool)
            keep2[drop] = False
            S_red2 = ordinate("pca_pres", Xp[keep2])
            drift2, _ = drift_after_procrustes(S_full[keep2], S_red2)
            rows.append(dict(paper=p, arm="matched", med_drift=float(np.median(drift2)),
                             n_rows_deleted=n_p, draw=dr))
    ev = pd.DataFrame(rows)
    # paired first draw
    block = ev[ev.arm == "block"][["paper", "med_drift"]].rename(columns={"med_drift": "block"})
    first = ev[(ev.arm == "matched") & (ev.draw == 0)][["paper", "med_drift"]].rename(
        columns={"med_drift": "matched"})
    pair = block.merge(first, on="paper")
    ratio = pair.block / pair.matched
    bf, lo, hi = cluster_boot_block(ev, n_boot=1000)
    return dict(block_factor=bf, block_ci_lo=lo, block_ci_hi=hi,
                median_paired_ratio=float(np.median(ratio)),
                paired_ci_lo=float(np.quantile(ratio, 0.025)),
                paired_ci_hi=float(np.quantile(ratio, 0.975)))


def main() -> None:
    X, rho, paper, params, *_ = load()
    paper = np.asarray(paper)
    rates = [(0.05, 0.01), (0.10, 0.05), (0.111, 0.0)]
    out = []
    for drop, add in rates:
        for rep in range(3):
            seed = SEED + rep * 101
            Xp = perturb(X, drop, add, seed)
            r = run_once(Xp, paper)
            r.update(drop_rate=drop, add_rate=add, replicate=rep,
                     n_present_before=int((X == 1).sum()),
                     n_present_after=int((Xp == 1).sum()))
            out.append(r)
            print(f"drop={drop:.3f} add={add:.3f} rep={rep}: "
                  f"block {r['block_factor']:.3f} "
                  f"({r['block_ci_lo']:.3f}-{r['block_ci_hi']:.3f}), "
                  f"paired {r['median_paired_ratio']:.3f}")
    df = pd.DataFrame(out)
    df.to_csv(os.path.join(RESULTS, "presence_misclassification_sensitivity.csv"),
              index=False, float_format="%.6g")
    summ = df.groupby(["drop_rate", "add_rate"]).agg(
        block_factor_median=("block_factor", "median"),
        block_factor_min=("block_factor", "min"),
        block_factor_max=("block_factor", "max"),
        paired_median=("median_paired_ratio", "median"),
        paired_min=("median_paired_ratio", "min"),
        paired_max=("median_paired_ratio", "max"),
        n=("block_factor", "size")).reset_index()
    summ.to_csv(os.path.join(RESULTS, "presence_misclassification_summary.csv"),
                index=False, float_format="%.6g")
    print("\n", summ.to_string(index=False))
    print("\nwrote presence_misclassification_sensitivity.csv and summary")


if __name__ == "__main__":
    main()
