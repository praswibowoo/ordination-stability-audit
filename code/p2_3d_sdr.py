#!/usr/bin/env python3
"""p2_3d_sdr.py — three-component sensitivity for the main SDR results.

Refits the presence PCA (44 binary columns, centered) and the standardized
F1 value PCA (four parameters, z-scored) with three components, runs the same
leave-one-paper-out Procrustes drift, and computes the per-event paper-weighted
SDR with the same null and percentile bootstrap as the two-component analysis.

Writes results/p2_3d_sdr.csv.
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42
N_PERM = 1000
N_BOOT = 2000

from experiment_C1 import load  # noqa: E402


def sdr(rho, drift):
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    hi = np.median(drift[rho >= q3])
    if not np.isfinite(hi) or hi <= 0:
        return np.nan
    return float(np.median(drift[rho <= q1]) / hi)


def per_event(specs):
    """specs: list of (rho, drift, paper) for each deletion event."""
    events = []
    for rho, drift, paper in specs:
        by = {}
        for r, d, p in zip(rho, drift, paper):
            by.setdefault(p, []).append((r, d))
        if len(by) < 4:
            continue
        rr = np.asarray([np.mean([x[0] for x in v]) for v in by.values()])
        dd = np.asarray([np.mean([x[1] for x in v]) for v in by.values()])
        events.append((rr, dd))
    obs = float(np.nanmedian([sdr(r, d) for r, d in events]))
    rng = np.random.default_rng(SEED)
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        null[b] = np.nanmedian([sdr(rng.permutation(r), d) for r, d in events])
    p1 = float((np.sum(null >= obs) + 1) / (N_PERM + 1))
    p2 = float(2 * min(p1, (np.sum(null <= obs) + 1) / (N_PERM + 1)))
    boot = np.empty(N_BOOT)
    for b in range(N_BOOT):
        idx = rng.integers(0, len(events), len(events))
        boot[b] = np.nanmedian([sdr(*events[i]) for i in idx])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return dict(n_events=len(events), sdr=round(obs, 4), p1=round(p1, 4),
                p2=round(p2, 4), ci_lo=round(float(lo), 4),
                ci_hi=round(float(hi), 4))


def lopo_specs(M, paper, rho, dims=3):
    """LOPO drift plus the density attached to each surviving row.

    ``rho`` must be the presence density of the row, not a mean of the
    ordination matrix: for standardized value rows ``M.mean(axis=1)`` is a
    mean z-score centred near zero and carries no density information.
    """
    S_full = PCA(n_components=dims, random_state=SEED).fit_transform(M)
    specs = []
    for p in sorted(set(paper)):
        keep = paper != p
        idx = np.where(keep)[0]
        S_red = PCA(n_components=dims, random_state=SEED).fit_transform(M[keep])
        a, b, _ = procrustes(S_full[idx], S_red)
        drift = np.linalg.norm(b - a, axis=1)
        specs.append((rho[keep], drift, paper[keep]))
    return specs


def main() -> None:
    X, rho, paper, _params, Xv, keys, rho_v = load()
    out = []
    out.append(dict(variant="presence_pca_3d", n_rows=len(X),
                    n_papers=len(set(paper)),
                    **per_event(lopo_specs(X, paper, rho, 3))))

    paper_v = np.asarray([k[0] for k in keys])
    Z = StandardScaler().fit_transform(Xv)
    # rho_v is the presence density mapped onto the value-layer rows; the
    # two-component value SDR uses the same density (experiment_C1.run_variant).
    out.append(dict(variant="std_value_pca_3d", n_rows=len(Z),
                    n_papers=len(set(paper_v)),
                    **per_event(lopo_specs(Z, paper_v, rho_v, 3))))

    path = os.path.join(RESULTS, "p2_3d_sdr.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"[p2-3d] wrote {os.path.relpath(path, ROOT)}")
    for r in out:
        print(r)


if __name__ == "__main__":
    main()
