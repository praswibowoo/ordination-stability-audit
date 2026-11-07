#!/usr/bin/env python3
"""subset_sdr_calibration.py — null calibration of the Table S6 subset SDR test.

The Table S6 density test uses 100 subsets of 30 deletion events.  Within each
subset the rows of the remaining studies are pooled, paper-level mean coverage
density is attached to each row, and the lower-tail permutation p-value is
computed by permuting paper mean densities.  This script calibrates that exact
estimator on the real 122-study / 730-row structure with synthetic drift:

    drift = study multiplier * row noise
    study multiplier ~ lognormal(0, sigma)
    row noise        ~ lognormal(0, 0.40)

No density effect is planted (gamma = 0), so a calibrated test should reject at
about alpha.  The script reports the per-subset false-positive rate and the
distribution of the median p-value across subsets.

Run:
  python3 code/subset_sdr_calibration.py --reps 200 --subsets 100 --perm 199 \
      --sigma 0.87 --out results/subset_sdr_calibration.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import load_primary  # noqa: E402


def si_fast(rho: np.ndarray, drift: np.ndarray) -> float:
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    hi = np.median(drift[rho >= q3])
    if hi <= 0:
        return float("nan")
    return float(np.median(drift[rho <= q1]) / hi)


def subset_pvalue(rho_k, drift_k, groups_k, rng, n_perm):
    uniq = np.asarray(sorted(set(groups_k)), dtype=object)
    code = np.searchsorted(uniq, groups_k)
    prho = np.array([rho_k[code == j].mean() for j in range(len(uniq))])
    obs = si_fast(prho[code], drift_k)

    perm = np.empty((n_perm, len(uniq)))
    for b in range(n_perm):
        perm[b] = rng.permutation(prho)
    rho_rows = perm[:, code]
    q1 = np.quantile(rho_rows, 0.25, axis=1)
    q3 = np.quantile(rho_rows, 0.75, axis=1)
    null = np.empty(n_perm)
    for b in range(n_perm):
        lo = np.median(drift_k[rho_rows[b] <= q1[b]])
        hi = np.median(drift_k[rho_rows[b] >= q3[b]])
        null[b] = lo / hi if hi > 0 else np.nan
    p = float((np.sum(null <= obs) + 1) / (n_perm + 1))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--subsets", type=int, default=100)
    ap.add_argument("--perm", type=int, default=199)
    ap.add_argument("--sigma", type=float, nargs="+", default=[0.0, 0.5, 0.87, 1.0, 1.32])
    ap.add_argument("--row-sigma", type=float, default=0.40)
    ap.add_argument("--out", default=os.path.join(RESULTS, "subset_sdr_calibration.csv"))
    args = ap.parse_args()

    X, rho, paper, *_ = load_primary()
    paper = np.asarray(paper)
    n_rows = len(rho)
    papers = np.asarray(sorted(set(paper)), dtype=object)
    n_papers = len(papers)
    paper_code = np.searchsorted(papers, paper)
    rng = np.random.default_rng(SEED)

    # Fixed subset design across replicates; random subsets of 30 studies.
    subset_idx = np.empty((args.subsets, 30), dtype=int)
    for s in range(args.subsets):
        subset_idx[s] = rng.choice(n_papers, size=30, replace=False)

    rows = []
    for sigma in args.sigma:
        t0 = time.time()
        pvals = []
        for rep in range(args.reps):
            study_mult = rng.lognormal(mean=0.0, sigma=sigma, size=n_papers)
            row_noise = rng.lognormal(mean=0.0, sigma=args.row_sigma, size=n_rows)
            drift = study_mult[paper_code] * row_noise
            for s in range(args.subsets):
                deleted = set(subset_idx[s])
                keep = np.array([c not in deleted for c in paper_code])
                pvals.append(subset_pvalue(
                    rho[keep], drift[keep], paper[keep], rng, args.perm))
        pvals = np.asarray(pvals, dtype=float)
        rows.append(dict(
            sigma=sigma,
            reps=args.reps,
            subsets=args.subsets,
            permutations=args.perm,
            n_tests=len(pvals),
            fpr_05=float(np.mean(pvals < 0.05)),
            fpr_10=float(np.mean(pvals < 0.10)),
            median_p=float(np.median(pvals)),
            p05=float(np.quantile(pvals, 0.05)),
            p25=float(np.quantile(pvals, 0.25)),
            p75=float(np.quantile(pvals, 0.75)),
            seconds=round(time.time() - t0, 1),
        ))
        print(f"  sigma={sigma:.2f}: FPR@.05={rows[-1]['fpr_05']:.3f} "
              f"median p={rows[-1]['median_p']:.3f} ({rows[-1]['seconds']:.0f}s)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    main()
