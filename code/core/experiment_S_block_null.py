#!/usr/bin/env python3
"""experiment_S_block_null.py — paper-weighted density null and study bootstrap.

The old "block-preserving null" permuted whole block labels. Because deletion
is defined by the labels, that permutation could not change the drift values:
the 0.604 observed / 0.607 null comparison was tautological.

This script replaces it with two separate procedures:

  * inference — permute paper-level mean coverage density across papers while
    holding the real rows and blocks fixed, then recompute the SDR. The
    one-sided p-value is for the observed SDR being below the null (dense
    studies drifting more); the two-sided p is also reported.
  * uncertainty — a study-level block bootstrap that resamples whole papers
    with replacement and recomputes the SDR.

The two are not conflated: the permutation supplies the null p-value, the
bootstrap supplies the confidence interval.

Outputs:
  results/sim_block_null_<tag>.csv
  results/sim_block_null_summary_<tag>.csv

Run:
  python3 code/experiment_S_block_null.py --variant pca_pres --reps 100 --deletes 30 --tag pca
  python3 code/experiment_S_block_null.py --variant pcoa_mds --reps 100 --deletes 30 --tag pcoa
  python3 code/experiment_S_block_null.py --variant mds_pres --reps 20 --deletes 10 --tag mds20
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

from experiment_C1 import load, jac_dist  # noqa: E402
from revision_layers import add_layer_args, layer_kwargs  # noqa: E402


def si_fast(rho, drift):
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    lo = np.median(drift[rho <= q1])
    hi = np.median(drift[rho >= q3])
    return float(lo / hi) if hi > 0 else float("nan")


def pcoa_jaccard(X):
    """Classical MDS (PCoA) on Jaccard distances; fast, metric, 2-D."""
    D = jac_dist(X)
    n = len(D)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    vals, vecs = np.linalg.eigh(B)
    idx = np.argsort(vals)[::-1][:2]
    return vecs[:, idx] * np.sqrt(np.maximum(vals[idx], 0.0))


def fit(variant, X):
    if variant == "pca_pres":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=SEED).fit_transform(X)
    if variant == "pcoa_mds":
        return pcoa_jaccard(X)
    if variant == "mds_pres":
        from experiment_C1 import ordinate
        return ordinate("mds_pres", X)
    raise ValueError(variant)


def procrustes_drift(S_ref, S_red):
    from scipy.spatial import procrustes
    a, b, _ = procrustes(S_ref, S_red)
    return np.linalg.norm(b - a, axis=1)


def drift_for_labels(X, rho, groups, labels, variant):
    """Delete each label's block, refit, pool Procrustes drift."""
    S_ref = fit(variant, X)
    drifts, rhos, labs = [], [], []
    for lab in labels:
        idx = np.where(groups == lab)[0]
        keep = np.ones(len(X), bool)
        keep[idx] = False
        keep_idx = np.where(keep)[0]
        S_red = fit(variant, X[keep])
        d = procrustes_drift(S_ref[keep_idx], S_red)
        drifts.append(d)
        rhos.append(rho[keep])
        labs.append(groups[keep])
    return np.concatenate(drifts), np.concatenate(rhos), np.concatenate(labs)


def sdr_inference(drift, rho, groups, rng, n_perm=199, n_boot=1000):
    """Paper-weighted density permutation p-values + study-level bootstrap CI.

    Both the observed statistic and the permutation null use the paper-weighted
    density ``prho[code]``. The row-weighted SDR is retained only descriptively
    so the comparison is not the old row-observed/paper-null hybrid.
    """
    uniq = np.asarray(sorted(set(groups)), dtype=object)
    code = np.searchsorted(uniq, groups)
    prho = np.array([rho[code == j].mean() for j in range(len(uniq))])
    obs_paper = si_fast(prho[code], drift)
    obs_row = si_fast(rho, drift)

    # Inference: permute paper-level mean density across papers. The real rows
    # and block composition stay fixed; observed and null are both paper-weighted.
    null = np.empty(n_perm)
    for b in range(n_perm):
        null[b] = si_fast(rng.permutation(prho)[code], drift)
    p_lower = float((np.sum(null <= obs_paper) + 1) / (n_perm + 1))
    p_upper = float((np.sum(null >= obs_paper) + 1) / (n_perm + 1))
    p_two = float(min(1.0, 2.0 * min(p_lower, p_upper)))

    # Uncertainty: resample whole papers with replacement. The resampled rows
    # get their own paper's mean density, so the bootstrap statistic is also
    # paper-weighted. Duplicates are kept because this is a cluster bootstrap.
    idx_by = {j: np.where(code == j)[0] for j in range(len(uniq))}
    boot = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([idx_by[j] for j in pick])
        rho_boot = np.concatenate([np.full(len(idx_by[j]), prho[j]) for j in pick])
        boot[b] = si_fast(rho_boot, drift[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(
        sdr=obs_paper,
        sdr_row_descriptive=obs_row,
        p_lower=p_lower,
        p_two=p_two,
        ci_lo=float(lo),
        ci_hi=float(hi),
        null_median=float(np.median(null)),
        null_lo=float(np.quantile(null, 0.025)),
        null_hi=float(np.quantile(null, 0.975)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pca_pres",
                    choices=["pca_pres", "pcoa_mds", "mds_pres"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--deletes", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tag", default=None,
                    help="artifact tag; defaults to the variant name")
    ap.add_argument("--exclude-papers", default="",
                    help="comma-separated batch IDs to remove before the null")
    ap.add_argument("--outdir", default=RESULTS,
                    help="output directory (default: results)")
    add_layer_args(ap)
    args = ap.parse_args()
    tag = args.tag or args.variant
    os.makedirs(args.outdir, exist_ok=True)

    X, rho, paper, params, *_ = load(**layer_kwargs(args))
    excluded = {p.strip() for p in args.exclude_papers.split(",") if p.strip()}
    if excluded:
        keep = ~np.isin(paper, sorted(excluded))
        X, rho, paper = X[keep], rho[keep], np.asarray(paper)[keep]
        print(f"excluded {sorted(excluded)}: {X.shape[0]} rows / {len(set(paper))} papers")
    paper = np.asarray(paper)
    labels = np.asarray(sorted(set(paper)))
    rng = np.random.default_rng(SEED)

    rows = []
    t0 = time.time()
    for rep in range(args.reps):
        chosen = rng.choice(labels, size=min(args.deletes, len(labels)),
                            replace=False)
        d, r, g = drift_for_labels(X, rho, paper, chosen, args.variant)
        stats = sdr_inference(d, r, g, rng, args.n_perm, args.n_boot)
        rows.append(dict(variant=args.variant, rep=rep, deletes=len(chosen),
                         **stats))
    print(f"  {args.variant}: median SDR {np.median([r['sdr'] for r in rows]):.3f} | "
          f"p_lower {np.median([r['p_lower'] for r in rows]):.3f} | "
          f"null {np.median([r['null_median'] for r in rows]):.3f} | "
          f"{time.time()-t0:.0f}s")

    out = os.path.join(args.outdir, f"sim_block_null_{tag}.csv")
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".tmp", out)

    summary = os.path.join(args.outdir, f"sim_block_null_summary_{tag}.csv")
    with open(summary + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "reps", "deletes", "median_sdr",
                    "median_sdr_row", "median_p_lower",
                    "median_p_two", "median_ci_lo", "median_ci_hi",
                    "median_null_sdr", "rej_lower_05", "rej_two_05"])
        w.writerow([
            args.variant, len(rows), args.deletes,
            round(float(np.median([r["sdr"] for r in rows])), 4),
            round(float(np.median([r["sdr_row_descriptive"] for r in rows])), 4),
            round(float(np.median([r["p_lower"] for r in rows])), 4),
            round(float(np.median([r["p_two"] for r in rows])), 4),
            round(float(np.median([r["ci_lo"] for r in rows])), 4),
            round(float(np.median([r["ci_hi"] for r in rows])), 4),
            round(float(np.median([r["null_median"] for r in rows])), 4),
            round(float(np.mean([r["p_lower"] < 0.05 for r in rows])), 4),
            round(float(np.mean([r["p_two"] < 0.05 for r in rows])), 4),
        ])
    os.replace(summary + ".tmp", summary)
    print(f"[density-null] wrote {os.path.relpath(out, ROOT)} + "
          f"{os.path.relpath(summary, ROOT)}")


if __name__ == "__main__":
    main()
