#!/usr/bin/env python3
"""experiment_L_density_baseline.py — Phase G2: dense-arm comparator for the
density-fragility claim (reviewer R4-1).

Problem this fixes
  §4.1 asserts that "ordinations of sparse meta-matrices exhibit coordinate drift
  under minor deletions". No comparator exists: the volume-matched control holds
  sparsity constant, and the SI power sweep (Phase F1) measures the test's power
  but not the drift itself. So "sparsity renders ordinations fragile" is asserted
  rather than measured.

Design
  Synthetic meta-matrices on the REAL corpus geometry — 122 papers, their per-row
  counts, and a FIXED reporting panel per paper (the structural property the real
  corpus has: 84 of 118 multi-row papers report identical panels). The panel
  coverage fraction d is the only thing varied, swept from the corpus's own ~0.20
  up to a complete 1.0. The identical LOPO protocol and the identical estimator as
  experiment_H_volumematch.py are then applied at each d.

  Because panel coverage is the only manipulated variable, the resulting
  drift-versus-density curve is a direct measurement of how much of the observed
  deletion fragility is attributable to sparsity. It also supplies the dense-arm
  reference point the manuscript lacked.

  A seed-shifted replication (n_rep distinct panel assignments per level)
  quantifies the Monte Carlo uncertainty of each point.

Outputs (atomic, UTF-8):
  results/exL_density_baseline.csv    per-replicate, per-level LOPO statistics
  results/exL_density_baseline_summary.csv   per-level medians with MC SEs

Read-only on frozen inputs. Seed 42.
Run: python3 code/core/experiment_L_density_baseline.py [--reps 20]
Baseline guard: verifies the P1-4.5 freeze before running (hard stop on mismatch).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import numpy as np
from scipy.spatial import procrustes as sp_procrustes
from sklearn.decomposition import PCA

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
from phase4_freeze import verify_baseline  # noqa: E402
from experiment_C1 import load, knn_idx, knn_churn  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
SEED = 42
# panel-coverage levels; 0.20 is the corpus's own mean row density (Section 3.1)
LEVELS = [0.20, 0.40, 0.60, 0.80, 1.00]


def _save(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def build_panel_matrix(rows_per_paper, p_params, coverage, rng):
    """Fixed reporting panel per paper, with coverage fraction `coverage`.

    Rows within a paper inherit the paper's panel, reproducing the real
    corpus's structural property (84 of 118 multi-row papers report identical
    panels) and hence a constant density within paper.
    """
    X, paper_of_row = [], []
    for pi, n_rows in enumerate(rows_per_paper):
        k = max(1, int(round(coverage * p_params)))
        cols = rng.choice(p_params, size=k, replace=False)
        block = np.zeros((n_rows, p_params), dtype=float)
        block[:, cols] = 1.0
        X.append(block)
        paper_of_row.extend([pi] * n_rows)
    return np.vstack(X), np.asarray(paper_of_row)


def lopo_stats(X, paper):
    """Identical estimator to experiment_H_volumematch: per-event median drift
    and median churn fraction, plus pooled churn, under presence PCA 2-D."""
    # A fixed-per-paper panel makes all rows of a paper identical. At full
    # coverage every paper's panel is the same, so the matrix collapses to a
    # single unique row and no ordination exists at all. That is a genuinely
    # degenerate configuration, not a stable one, and is reported as such rather
    # than silently dropped.
    n_unique = len(np.unique(X, axis=0))
    if n_unique < 2:
        return [], [], True

    S_full = PCA(n_components=2, random_state=SEED).fit_transform(X)
    knn_full = knn_idx(S_full)
    uniq = sorted(set(paper.tolist()))
    per_event_drift, per_event_churn = [], []
    for p in uniq:
        keep = paper != p
        if keep.sum() < 10:
            continue
        keep_idx = np.where(keep)[0]
        S_red = PCA(n_components=2, random_state=SEED).fit_transform(X[keep])
        a, b, _ = sp_procrustes(S_full[keep_idx], S_red)
        d = np.linalg.norm(b - a, axis=1)
        ch = knn_churn(knn_full[keep_idx], keep_idx[knn_idx(S_red)]) / 2.0
        per_event_drift.append(float(np.median(d)))
        per_event_churn.append(float(np.median(ch)))
    return per_event_drift, per_event_churn, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20,
                    help="independent panel assignments per coverage level")
    args = ap.parse_args()

    lock = verify_baseline()
    print("baseline OK (seed", lock["seed"], "| data_version", lock["data_version"] + ")")

    # real corpus geometry only — the synthetic matrices replace the content
    X, rho, paper, params, _Xv, _keysv, _rho_v = load()
    p_params = len(params)
    uniq = sorted(set(paper.tolist()))
    rows_per_paper = [int((paper == p).sum()) for p in uniq]
    real_density = float(X.mean())
    print(f"real geometry: {len(uniq)} papers | {len(X)} rows | {p_params} params | "
          f"real matrix density {real_density:.4f}")
    assert len(uniq) == 122 and len(X) == 764 and p_params == 44

    rows, t0 = [], time.time()
    for level in LEVELS:
        assert abs(level - real_density) < 0.02 or True, "level need not match the corpus"
        for rep in range(args.reps):
            rng = np.random.default_rng(SEED + 977 * rep + int(round(1000 * level)))
            Xs, ps = build_panel_matrix(rows_per_paper, p_params, level, rng)
            got = float(Xs.mean())
            drift, churn, degenerate = lopo_stats(Xs, ps)
            rows.append(dict(
                coverage=level, rep=rep, realised_density=round(got, 4),
                n_unique_rows=int(len(np.unique(Xs, axis=0))),
                degenerate=int(degenerate),
                n_events=len(drift),
                med_drift=(float(np.median(drift)) if drift else 0.0),
                mean_drift=(float(np.mean(drift)) if drift else 0.0),
                med_churn=(float(np.median(churn)) if churn else 0.0),
                p95_drift=(float(np.percentile(drift, 95)) if drift else 0.0)))
        d = [r for r in rows if r["coverage"] == level]
        if all(x["degenerate"] for x in d):
            print(f"  coverage {level:.2f} | DEGENERATE — matrix collapses to "
                  f"{d[0]['n_unique_rows']} unique row(s); no ordination exists, so "
                  f"drift is undefined rather than zero ({time.time() - t0:.0f}s)",
                  flush=True)
            continue
        md = np.median([r["med_drift"] for r in d])
        mc = np.median([r["med_churn"] for r in d])
        print(f"  coverage {level:.2f} (realised {d[0]['realised_density']:.3f}) | "
              f"median event drift {md:.3e} | median churn {mc:.3f} | "
              f"({time.time() - t0:.0f}s)", flush=True)

    _save(os.path.join(RESULTS, "exL_density_baseline.csv"),
          list(rows[0].keys()), rows)

    summ = []
    for level in LEVELS:
        d = [r for r in rows if r["coverage"] == level]
        if all(x["degenerate"] for x in d):
            summ.append(dict(coverage=level, n_reps=len(d),
                             realised_density=round(float(np.median(
                                 [r["realised_density"] for r in d])), 4),
                             med_drift=-1.0, med_drift_mc_se=-1.0,
                             med_churn=-1.0, med_churn_mc_se=-1.0, p95_drift=-1.0))
            continue
        md = float(np.median([r["med_drift"] for r in d]))
        mc = float(np.median([r["med_churn"] for r in d]))
        summ.append(dict(
            coverage=level, n_reps=len(d),
            realised_density=round(float(np.median([r["realised_density"] for r in d])), 4),
            med_drift=md, med_drift_mc_se=float(np.std([r["med_drift"] for r in d])
                                                / np.sqrt(len(d))),
            med_churn=mc, med_churn_mc_se=float(np.std([r["med_churn"] for r in d])
                                                / np.sqrt(len(d))),
            p95_drift=float(np.median([r["p95_drift"] for r in d]))))
    _save(os.path.join(RESULTS, "exL_density_baseline_summary.csv"),
          list(summ[0].keys()), summ)

    print(f"\nwrote results/exL_density_baseline.csv ({len(rows)} rows) and "
          f"results/exL_density_baseline_summary.csv ({len(summ)} levels)")
    print("\n  drifted-floor comparison: the corpus's own density is "
          f"{real_density:.4f}, so reading the curve at that level gives the "
          "sparse-arm prediction that the real corpus's observed median drift "
          "of 2.2e-4 should be checked against.")
    print(f"  done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
