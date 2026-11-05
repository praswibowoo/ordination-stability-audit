#!/usr/bin/env python3
"""mds_multistart_check.py — 100-start check of the full-data Jaccard MDS.

The manuscript's primary MDS uses one random start. This diagnostic runs 100
single-start fits with different seeds on the same 764 x 764 Jaccard matrix,
records Kruskal stress-1 for each, and Procrustes-aligns each configuration to
the committed single-start fit (seed 42). It answers two questions in one pass:

  1. Is the committed fit the best of 100 starts (stress)?
  2. Is the best configuration the same solution up to rotation/reflection
     (Procrustes disparity near zero), or a different local minimum?

Output: results/mds_multistart_check.csv (seed, stress1, procrustes_disparity)
Run: python3 code/mds/mds_multistart_check.py    (~5 min)
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.spatial import procrustes
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
from experiment_C1 import jac_dist, load  # noqa: E402

SEED = 42
N_STARTS = 100
OUT = os.path.join(ROOT, "results", "mds_multistart_check.csv")


def stress1(D: np.ndarray, emb: np.ndarray) -> float:
    iu = np.triu_indices_from(D, k=1)
    d = D[iu]
    dh = squareform(pdist(emb))[iu]
    return float(np.sqrt(np.sum((d - dh) ** 2) / np.sum(d ** 2)))


def fit(D: np.ndarray, seed: int) -> np.ndarray:
    return MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
               n_init=1, normalized_stress="auto").fit_transform(D)


def main() -> int:
    X, _rho, _paper, _params, _Xv, _keys, _rho_v = load()
    D = jac_dist(X)
    ref = fit(D, SEED)
    s_ref = stress1(D, ref)
    rows = []
    for seed in range(N_STARTS):
        emb = fit(D, seed)
        _a, _b, disp = procrustes(ref, emb)
        rows.append(dict(seed=seed,
                         stress1=round(stress1(D, emb), 8),
                         procrustes_disparity=round(float(disp), 8)))
    stresses = np.asarray([r["stress1"] for r in rows])
    disps = np.asarray([r["procrustes_disparity"] for r in rows])
    best = rows[int(np.argmin(stresses))]
    same = int(np.sum(stresses <= s_ref + 1e-9))
    identical = int(np.sum(disps <= 1e-6))
    print(f"[multistart] committed single-start stress-1 = {s_ref:.8f}")
    print(f"[multistart] 100-start min/median/max = "
          f"{stresses.min():.8f} / {np.median(stresses):.8f} / {stresses.max():.8f}")
    print(f"[multistart] best seed {best['seed']} | stress {best['stress1']:.8f} | "
          f"Procrustes disparity {best['procrustes_disparity']:.8f}")
    print(f"[multistart] starts within 1e-9 of committed stress: {same}/100")
    print(f"[multistart] configurations identical to committed fit (disp <= 1e-6): "
          f"{identical}/100")
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)
    print(f"[multistart] wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
