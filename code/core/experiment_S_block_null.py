#!/usr/bin/env python3
"""experiment_S_block_null.py — block-preserving ordination null (C4 follow-up).

The independent-Bernoulli check (experiment_S_ordination_calibration.py) did
not preserve the real co-reporting blocks, so its null SDR was far from 1.
This experiment keeps the real presence matrix exactly and shuffles **whole
paper blocks** (all rows of a paper, with their panels and densities). It then
runs actual LOPO PCA or classical-MDS (PCoA) refits and asks:

  * is the observed SDR typical of the block-preserving null?
  * does the paper-level density permutation test reject at ~5% under that null?

Outputs
-------
results/sim_block_null_<tag>.csv          per-subset / per-permutation outcomes
results/sim_block_null_summary_<tag>.csv  observed vs null SDR and rejection rates

The tag names the cited artifact. The committed runs use ``pca`` (100 x 30),
``pcoa`` (100 x 30) and ``mds20`` (20 x 10). Without ``--tag`` the variant name
is used, so repeated runs cannot silently overwrite a cited file.

Run:
  python3 code/core/experiment_S_block_null.py --variant pca_pres --reps 100 --deletes 30 --tag pca
  python3 code/core/experiment_S_block_null.py --variant pcoa_mds --reps 100 --deletes 30 --tag pcoa
  python3 code/core/experiment_S_block_null.py --variant mds_pres --reps 20 --deletes 10 --tag mds20
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import load, jac_dist  # noqa: E402


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


def test_sdr(drift, rho, groups, rng, n_perm=199):
    obs = si_fast(rho, drift)
    null_row = np.empty(n_perm)
    for b in range(n_perm):
        null_row[b] = si_fast(rng.permutation(rho), drift)
    p_row = float((np.sum(null_row >= obs) + 1) / (n_perm + 1))

    uniq = np.asarray(sorted(set(groups)))
    code = np.searchsorted(uniq, groups)
    prho = np.asarray([rho[code == j].mean() for j in range(len(uniq))])
    null_blk = np.empty(n_perm)
    for b in range(n_perm):
        null_blk[b] = si_fast(rng.permutation(prho)[code], drift)
    p_block = float((np.sum(null_blk >= obs) + 1) / (n_perm + 1))
    return obs, p_row, p_block


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="pca_pres",
                    choices=["pca_pres", "pcoa_mds", "mds_pres"])
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--deletes", type=int, default=30)
    ap.add_argument("--n-perm", type=int, default=199)
    ap.add_argument("--tag", default=None,
                    help="artifact tag; defaults to the variant name")
    args = ap.parse_args()
    tag = args.tag or args.variant

    X, rho, paper, params, *_ = load()
    labels = np.asarray(sorted(set(paper)))
    blocks = {p: np.where(paper == p)[0] for p in labels}
    rng = np.random.default_rng(SEED)

    rows = []
    t0 = time.time()

    # observed: random subsets of real paper blocks
    for rep in range(args.reps):
        chosen = rng.choice(labels, size=min(args.deletes, len(labels)),
                            replace=False)
        d, r, g = drift_for_labels(X, rho, paper, chosen, args.variant)
        obs, p_row, p_block = test_sdr(d, r, g, rng, args.n_perm)
        rows.append(dict(variant=args.variant, arm="observed", rep=rep,
                         sdr=round(obs, 4), p_row=round(p_row, 4),
                         p_block=round(p_block, 4),
                         rej_row=int(p_row < 0.05),
                         rej_block=int(p_block < 0.05)))
    obs_rows = [r for r in rows if r["arm"] == "observed"]
    print(f"  observed: median SDR {np.median([r['sdr'] for r in obs_rows]):.3f} | "
          f"row {np.mean([r['rej_row'] for r in obs_rows]):.3f} / "
          f"paper {np.mean([r['rej_block'] for r in obs_rows]):.3f} | "
          f"{time.time()-t0:.0f}s")

    # null: permute whole block labels, then delete blocks
    for rep in range(args.reps):
        # each block gets a random label; the label set is unchanged
        perm = rng.permutation(labels)
        perm_groups = np.empty(len(paper), dtype=object)
        for block_label, new_label in zip(labels, perm):
            perm_groups[blocks[block_label]] = new_label
        chosen = rng.choice(labels, size=min(args.deletes, len(labels)),
                            replace=False)
        d, r, g = drift_for_labels(X, rho, perm_groups, chosen, args.variant)
        obs, p_row, p_block = test_sdr(d, r, g, rng, args.n_perm)
        rows.append(dict(variant=args.variant, arm="block_null", rep=rep,
                         sdr=round(obs, 4), p_row=round(p_row, 4),
                         p_block=round(p_block, 4),
                         rej_row=int(p_row < 0.05),
                         rej_block=int(p_block < 0.05)))
    null_rows = [r for r in rows if r["arm"] == "block_null"]
    print(f"  block null: median SDR {np.median([r['sdr'] for r in null_rows]):.3f} | "
          f"row {np.mean([r['rej_row'] for r in null_rows]):.3f} / "
          f"paper {np.mean([r['rej_block'] for r in null_rows]):.3f} | "
          f"{time.time()-t0:.0f}s")

    out = os.path.join(RESULTS, f"sim_block_null_{tag}.csv")
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".tmp", out)

    summary = os.path.join(RESULTS, f"sim_block_null_summary_{tag}.csv")
    with open(summary + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "arm", "reps", "deletes", "median_sdr",
                    "rej_row", "rej_block"])
        for arm, sub in (("observed", obs_rows), ("block_null", null_rows)):
            w.writerow([args.variant, arm, len(sub), args.deletes,
                        round(float(np.median([r["sdr"] for r in sub])), 4),
                        round(float(np.mean([r["rej_row"] for r in sub])), 4),
                        round(float(np.mean([r["rej_block"] for r in sub])), 4)])
    os.replace(summary + ".tmp", summary)
    print(f"[block-null] wrote {os.path.relpath(out, ROOT)} + "
          f"{os.path.relpath(summary, ROOT)}")


if __name__ == "__main__":
    main()
