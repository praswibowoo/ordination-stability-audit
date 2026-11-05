#!/usr/bin/env python3
"""mds_stress.py — fit-stress report for the primary MDS variant (P8E, D171).

Spec (identical to experiment_C1.ordinate("mds_pres")): metric SMACOF MDS over
precomputed Jaccard dissimilarities, two components, classical-MDS initialization
(init = "classical_mds", sklearn >= 1.8), n_init = 1, normalized_stress = "auto".
random_state = 42 is retained for API stability but is inert under classical init.

Fit quality is reported as Kruskal stress-1 over the upper triangle:
    stress1 = sqrt( sum_{i<j} (D_ij - dhat_ij)^2 / sum_{i<j} D_ij^2 )
computed on the full-data fit (764 rows / 122 papers) and on every
leave-one-paper-out refit (122 events).

Output: results/mds_stress_summary.csv (atomic write; frozen layers untouched).
Run: python3 code/mds/mds_stress.py        (~4 min)
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import MDS

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
from phase4_freeze import verify_baseline  # noqa: E402
from experiment_C1 import classical_mds_init, load, jac_dist  # noqa: E402  (module main is guarded)

SEED = 42
OUT = os.path.join(ROOT, "results", "mds_stress_summary.csv")
FIELDS = ["variant", "scope", "n_rows", "n_events",
          "stress1_med", "stress1_q1", "stress1_q3", "stress1_min", "stress1_max"]


def fit_stress1(D: np.ndarray) -> float:
    """Metric SMACOF fit + Kruskal stress-1 (upper triangle)."""
    emb = MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
              **classical_mds_init(), n_init=1,
              normalized_stress="auto").fit_transform(D)
    iu = np.triu_indices_from(D, k=1)
    d = D[iu]
    dh = squareform(pdist(emb))[iu]
    return float(np.sqrt(np.sum((d - dh) ** 2) / np.sum(d ** 2)))


def main() -> None:
    verify_baseline()
    X, rho, paper, params, Xv, keys, rho_v = load()
    D = jac_dist(X)
    s_full = fit_stress1(D)
    s_again = fit_stress1(D)
    assert s_full == s_again, "non-deterministic fit (seed/n_init drifted)"
    papers = sorted(set(paper))
    stresses = np.asarray([fit_stress1(D[np.ix_(paper != p, paper != p)]) for p in papers])
    rows = [
        dict(variant="mds_pres", scope="full", n_rows=int(X.shape[0]), n_events=1,
             stress1_med=round(s_full, 6), stress1_q1=round(s_full, 6),
             stress1_q3=round(s_full, 6), stress1_min=round(s_full, 6),
             stress1_max=round(s_full, 6)),
        dict(variant="mds_pres", scope="lopo", n_rows=int(X.shape[0]), n_events=len(papers),
             stress1_med=round(float(np.median(stresses)), 6),
             stress1_q1=round(float(np.quantile(stresses, 0.25)), 6),
             stress1_q3=round(float(np.quantile(stresses, 0.75)), 6),
             stress1_min=round(float(stresses.min()), 6),
             stress1_max=round(float(stresses.max()), 6)),
    ]
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)
    print(f"wrote {OUT}")
    print(f"full stress-1 = {s_full:.6f}")
    print(f"LOPO stress-1: median {np.median(stresses):.6f} "
          f"[{np.quantile(stresses, 0.25):.6f}, {np.quantile(stresses, 0.75):.6f}] "
          f"over {len(papers)} events")


if __name__ == "__main__":
    main()
