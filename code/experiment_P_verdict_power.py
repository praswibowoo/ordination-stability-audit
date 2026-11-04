#!/usr/bin/env python3
"""experiment_P_verdict_power.py — positive control for the F1 value-layer verdict tests.

The main text reports a null result for author-assigned classes in the 57-row
F1 physical multi-class subset. A null can mean "no structure" or "no power".
This script plants a known shift in the same 421-row value matrix, assigns
balanced three-group labels at random, and runs the same within-paper
PERMANOVA / ANOSIM tests used in Table S5. The rejection rate at delta = 0 is
the calibration check; the rates at delta > 0 are the power of the test to
detect a known separation.

Design:
  * Z = standardized 421 x 4 F1 physical value matrix (DO, pH, salinity, temperature)
  * labels = random balanced permutation of three groups (140/140/141)
  * group A receives +delta SD on salinity (column 2)
  * 100 replicates per delta, 99 within-paper permutations
  * no silhouette: the permutation loop is the cost driver and PERMANOVA/ANOSIM
    answer the power question; silhouette remains a descriptive diagnostic.

Output: results/exP_verdict_power.csv
Run: python3 code/experiment_P_verdict_power.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_B_familymaps import permanova_test, permute_anosim_p  # noqa: E402
from experiment_C1 import load  # noqa: E402

N_REP = 100
N_PERM = 99
DELTAS = [0.0, 0.25, 0.5, 1.0]
SEED = 42
OUT = os.path.join(ROOT, "results", "exP_verdict_power.csv")


def main() -> int:
    _X, _rho, _paper, _params, Xv, keys, _rho_v = load()
    Z = StandardScaler().fit_transform(Xv)
    papers = np.array([k[0] for k in keys])
    n = len(Z)
    base = np.array(["A"] * (n // 3) + ["B"] * (n // 3) + ["C"] * (n - 2 * (n // 3)))

    rows = []
    for delta in DELTAS:
        rng = np.random.default_rng(SEED + int(1000 * delta))
        rej_f = rej_r = 0
        fs = []
        for _ in range(N_REP):
            labels = rng.permutation(base)
            Zs = Z.copy()
            Zs[labels == "A", 2] += delta
            D = squareform(pdist(Zs))
            f, p, _ = permanova_test(D, labels, papers, rng, n_perm=N_PERM)
            _r, rp = permute_anosim_p(D, labels, papers, rng, n_perm=N_PERM)
            rej_f += int(p < 0.05)
            rej_r += int(rp < 0.05)
            fs.append(float(f))
        rows.append(dict(
            delta=delta,
            n_rep=N_REP,
            n_perm=N_PERM,
            perm_rej=round(rej_f / N_REP, 4),
            anosim_rej=round(rej_r / N_REP, 4),
            median_F=round(float(np.median(fs)), 4),
        ))
        print(f"[positive-control] delta={delta:.2f}  PERMANOVA={rej_f/N_REP:.3f}  "
              f"ANOSIM={rej_r/N_REP:.3f}  median F={np.median(fs):.3f}")

    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)
    print(f"[positive-control] wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
