#!/usr/bin/env python3
"""experiment_S_ordination_calibration.py — ordination-based calibration (C4).

The original S1 calibration generates a drift vector directly and runs no
ordination. This experiment closes that gap: it simulates binary presence
matrices on the real 122-paper geometry, runs the actual LOPO PCA (and,
optionally, Jaccard MDS) refits, computes Procrustes drift, and applies the
same row-level and paper-level SDR tests.

Design
------
For each synthetic corpus:
  * row probabilities follow the real row densities, with a paper-level
    logit random effect sigma;
  * a density-dependent instability term flips a fraction
    eta_p = gamma * max(0, median(prho) - prho_p) of cells in each paper,
    so sparse papers are perturbed more;
  * a random subset of papers is deleted one at a time; each deletion is
    refit and Procrustes-aligned to the full-data fit;
  * drift is pooled across deletion events and tested with
    (a) a fully exchangeable row-level permutation and
    (b) a paper-level permutation of paper mean densities.

Outputs
-------
results/sim_ordination_calibration.csv          per-replicate outcomes
results/sim_ordination_calibration_summary.csv  rejection rates by setting

Run (pilot):
  python3 code/experiment_S_ordination_calibration.py --reps 10 --deletes 10 \
      --variant pca_pres --settings null,cluster,gamma0.5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import load, ordinate, drift_after_procrustes  # noqa: E402

VARIANT = "pca_pres"


def si_fast(rho, drift):
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    lo = np.median(drift[rho <= q1])
    hi = np.median(drift[rho >= q3])
    return float(lo / hi) if hi > 0 else float("nan")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def logit(p):
    p = np.clip(p, 0.02, 0.95)
    return np.log(p / (1.0 - p))


def paper_mean(rho, paper):
    uniq = np.asarray(sorted(set(paper)))
    code = np.searchsorted(uniq, paper)
    return uniq, code, np.asarray([rho[code == j].mean() for j in range(len(uniq))])


def simulate_matrix(rng, rho, paper, sigma, gamma):
    """Binary presence matrix on the real row/paper geometry."""
    uniq, code, prho = paper_mean(rho, paper)
    med = float(np.median(prho))
    u = rng.standard_normal(len(uniq))
    P = sigmoid(logit(rho) + sigma * u[code])
    X = (rng.random((len(rho), 44)) < P[:, None]).astype(float)
    if gamma > 0:
        # density-dependent instability: sparse papers get more cell flips
        for j, p in enumerate(uniq):
            idx = np.where(code == j)[0]
            eta = gamma * max(0.0, med - prho[j])
            if eta <= 0:
                continue
            mask = rng.random((len(idx), X.shape[1])) < eta
            sub = X[np.ix_(idx, np.arange(X.shape[1]))]
            sub[mask] = 1.0 - sub[mask]
            X[np.ix_(idx, np.arange(X.shape[1]))] = sub
    return X


def lopo_drift(X, paper, rng, n_delete):
    """LOPO refits and pooled Procrustes drift for a random paper subset."""
    uniq = np.asarray(sorted(set(paper)))
    chosen = rng.choice(uniq, size=min(n_delete, len(uniq)), replace=False)
    S_ref = ordinate(VARIANT, X)
    drifts, rhos, papers = [], [], []
    for p in chosen:
        keep = paper != p
        keep_idx = np.where(keep)[0]
        S_red = ordinate(VARIANT, X[keep])
        d, _ = drift_after_procrustes(S_ref[keep_idx], S_red)
        drifts.append(d)
        rhos.append(X[keep].mean(axis=1))
        papers.append(paper[keep])
    return (np.concatenate(drifts), np.concatenate(rhos),
            np.concatenate(papers))


def test_sdr(drift, rho, paper, rng, n_perm=199):
    obs = si_fast(rho, drift)
    null_row = np.empty(n_perm)
    for b in range(n_perm):
        null_row[b] = si_fast(rng.permutation(rho), drift)
    p_row = float((np.sum(null_row >= obs) + 1) / (n_perm + 1))

    uniq, code, prho = paper_mean(rho, paper)
    codes = np.searchsorted(uniq, paper)
    null_blk = np.empty(n_perm)
    for b in range(n_perm):
        null_blk[b] = si_fast(rng.permutation(prho)[codes], drift)
    p_block = float((np.sum(null_blk >= obs) + 1) / (n_perm + 1))
    return obs, p_row, p_block


def main():
    global VARIANT
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--deletes", type=int, default=30)
    ap.add_argument("--variant", default="pca_pres",
                    choices=["pca_pres", "mds_pres"])
    ap.add_argument("--settings", default="null,cluster,gamma0.2,gamma0.5,gamma1.0")
    ap.add_argument("--n-perm", type=int, default=199)
    args = ap.parse_args()
    VARIANT = args.variant

    X, rho, paper, params, *_ = load()
    settings = []
    for s in args.settings.split(","):
        if s == "null":
            settings.append((0.0, 0.0, s))
        elif s == "cluster":
            settings.append((0.8, 0.0, s))
        elif s.startswith("gamma"):
            settings.append((0.8, float(s.replace("gamma", "")), s))
        else:
            raise ValueError(s)

    rows = []
    t0 = time.time()
    for sigma, gamma, label in settings:
        for rep in range(args.reps):
            rng = np.random.default_rng(SEED + 1000 * rep + int(100 * gamma)
                                        + int(10 * sigma))
            Xs = simulate_matrix(rng, rho, paper, sigma, gamma)
            drift, rho_s, paper_s = lopo_drift(Xs, paper, rng, args.deletes)
            obs, p_row, p_block = test_sdr(drift, rho_s, paper_s, rng,
                                           n_perm=args.n_perm)
            rows.append(dict(variant=args.variant, setting=label,
                             sigma=sigma, gamma=gamma, rep=rep,
                             sdr=round(obs, 4), p_row=round(p_row, 4),
                             p_block=round(p_block, 4),
                             rej_row=int(p_row < 0.05),
                             rej_block=int(p_block < 0.05)))
        sub = [r for r in rows if r["setting"] == label]
        print(f"  {args.variant} {label}: "
              f"row {np.mean([r['rej_row'] for r in sub]):.3f} / "
              f"paper {np.mean([r['rej_block'] for r in sub]):.3f} | "
              f"median SDR {np.median([r['sdr'] for r in sub]):.3f} | "
              f"{time.time()-t0:.0f}s")

    out = os.path.join(RESULTS, "sim_ordination_calibration.csv")
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".tmp", out)

    summary = os.path.join(RESULTS, "sim_ordination_calibration_summary.csv")
    with open(summary + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "setting", "reps", "deletes", "rej_row",
                    "rej_block", "median_sdr"])
        for sigma, gamma, label in settings:
            sub = [r for r in rows if r["setting"] == label]
            w.writerow([args.variant, label, len(sub), args.deletes,
                        round(float(np.mean([r["rej_row"] for r in sub])), 4),
                        round(float(np.mean([r["rej_block"] for r in sub])), 4),
                        round(float(np.median([r["sdr"] for r in sub])), 4)])
    os.replace(summary + ".tmp", summary)
    print(f"[ordination-calibration] wrote {os.path.relpath(out, ROOT)} + summary")


if __name__ == "__main__":
    main()
