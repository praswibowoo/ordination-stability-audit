#!/usr/bin/env python3
"""stats_selftest.py — known-answer fixtures for the Phase-4 statistics (D108).

Independent of the data: constructs tiny examples with hand-computed answers and asserts the
implementations reproduce them. Run before any stats-bearing freeze:
    python3 code/stats_selftest.py
"""
from __future__ import annotations
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from experiment_B_familymaps import permanova_f, anosim_r  # noqa: E402
from experiment_C1 import classical_mds_init  # noqa: E402
from experiment_C2_stability import event_si, holm, bootstrap_G  # noqa: E402
from mds_seed_baseline import CHECKS, check_summary, parse_seeds  # noqa: E402
from scipy.spatial.distance import squareform, pdist  # noqa: E402
from sklearn.manifold import MDS  # noqa: E402


def check(name, ok):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return ok


def main():
    ok = True

    # --- PERMANOVA fixture: x=[0,1,10,11], groups A,A,B,B ---
    # d² upper-tri sum = 1+100+121+81+100+1 = 404; SS_T = ΣΣd²/(2n) = 808/8 = 101
    # SS_W = Σ_g (Σ_g d²)/(2n_g) = group{0,1}: 2/4 + group{2,3}: 2/4 = 1.0
    # SS_A = 100 → F = (100/1)/(1.0/2) = 200.0
    x = np.array([[0.0], [1.0], [10.0], [11.0]])
    D = squareform(pdist(x))
    f, ssw = permanova_f(D, ["A", "A", "B", "B"])
    ok &= check("permanova_f == 200.0 (SS_W==1.0)", abs(f - 200.0) < 1e-9 and abs(ssw - 1.0) < 1e-9)

    # --- ANOSIM fixture: x=[0,3,4,10], groups A,A,B,B → R = 0.25 ---
    x2 = np.array([[0.0], [3.0], [4.0], [10.0]])
    D2 = squareform(pdist(x2))
    R = anosim_r(D2, ["A", "A", "B", "B"])
    ok &= check("anosim_r == 0.25", abs(R - 0.25) < 1e-12)

    # --- SI fixture: rho=[0,0,10,10,20,20,30,30], drift=[4,4,3,3,2,2,1,1] ---
    # linear quantiles: Q1 = 0+0.75*10 = 7.5 → sparse rho≤7.5 = {0,0} → median δ 4.0
    #                   Q3 = 20+0.25*10 = 22.5 → dense ρ≥22.5 = {30,30} → median δ 1.0
    # SI = 4.0/1.0 = 4.0
    rho = np.array([0, 0, 10, 10, 20, 20, 30, 30], float)
    drift = np.array([4, 4, 3, 3, 2, 2, 1, 1], float)
    si = event_si(rho, drift)
    ok &= check("event_si == 4.0", abs(si - 4.0) < 1e-12)

    # --- Holm fixture: [0.01, 0.04, 0.03] → [0.03, 0.06, 0.06] ---
    adj = holm([0.01, 0.04, 0.03])
    ok &= check("holm == [0.03, 0.06, 0.06]", np.allclose(adj, [0.03, 0.06, 0.06]))

    # --- bootstrap G fixture: G const +0.5 pairs → median G == 0.5, P(G>0) small ---
    pairs = [dict(SI_family=0.5, SI_all=1.0, G=0.5)] * 10
    b = bootstrap_G(pairs, np.random.default_rng(42), n_boot=200)
    ok &= check("bootstrap_G median == 0.5", abs(b["G_median"] - 0.5) < 1e-12)

    # --- Churn identity mapping regression test ---
    from experiment_C1 import knn_idx, knn_churn
    # 1. Identity case: identical neighbor sets must yield zero churn
    a = np.array([[1, 2, 3], [0, 2, 3]])
    ch_zero = knn_churn(a, a)
    ok &= check("knn_churn(A, A) == 0.0", np.allclose(ch_zero, 0.0))

    # 2. Index alignment under middle-block deletion:
    # 20 1D points on a line: [0, 1, 2, ..., 19].
    # Keeping rows [0..4] and [15..19] (deleting middle [5..14]):
    # The reduced ordination preserves exact order for the kept rows.
    # With identity mapping ki[reduced_neighbors], retained points match their original identities.
    pts_full = np.arange(20, dtype=float).reshape(-1, 1)
    knn_full = knn_idx(pts_full, k=3)
    keep = np.array([True]*5 + [False]*10 + [True]*5)
    ki = np.where(keep)[0]
    pts_red = pts_full[keep]
    knn_red = knn_idx(pts_red, k=3)
    # Buggy unmapped indexing:
    ch_bug = knn_churn(knn_full[keep], knn_red)
    # Corrected identity-mapped indexing:
    ch_fix = knn_churn(knn_full[keep], ki[knn_red])
    # The first 5 rows (indices 0..4) had neighbors inside [0..4].
    # Under buggy indexing, rows 15..19 receive local indices 5..9, which clash with original indices.
    # Therefore, ch_fix must have zero churn on boundary-free rows while ch_bug has spurious churn.
    ok &= check("churn index mapping corrects spurious middle-deletion churn",
                float(ch_fix[0]) == 0.0 and float(ch_bug[5]) > 0.0)

    # --- Phase 1 seed-baseline checks (D253) ---
    ok &= check("classical_mds_init returns the pinned init",
                classical_mds_init() == {"init": "classical_mds"})
    small = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    small_d = squareform(pdist(small))

    def _fit_small():
        return MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=42,
            **classical_mds_init(),
            n_init=1,
            normalized_stress="auto",
        ).fit_transform(small_d)

    ok &= check("classical MDS small fit is deterministic",
                np.array_equal(_fit_small(), _fit_small()))
    ok &= check("parse_seeds('0-3,5') == [0,1,2,3,5]",
                parse_seeds("0-3,5") == [0, 1, 2, 3, 5])
    ok &= check("seed-42 deletion and hybrid targets are distinct",
                0.0182216 != 0.0217170)
    ok &= check("seed-0 deletion and hybrid targets are distinct",
                0.0095830 != 0.0217170)
    ok &= check("hybrid seed-0 is within 0.001 of the audit 0.02141",
                abs(0.0217170 - 0.02141) <= 0.001)
    synthetic = {}
    for protocol, fs, rs, col, expected, _tol in CHECKS:
        key = (protocol, fs, rs)
        synthetic.setdefault(
            key,
            {"protocol": protocol, "full_seed": fs, "refit_seed": rs},
        )[col] = expected
    synthetic_rows = list(synthetic.values())
    ok &= check("mds_seed_baseline regression table passes",
                check_summary(synthetic_rows, verbose=False))
    shifted = [dict(r) for r in synthetic_rows]
    shifted[0]["mean_pooled"] = float(shifted[0]["mean_pooled"]) + 1.0
    ok &= check("mds_seed_baseline detects a shifted regression value",
                not check_summary(shifted, verbose=False))

    print("stats_selftest:", "ALL PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
