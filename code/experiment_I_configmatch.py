#!/usr/bin/env python3
"""experiment_I_configmatch.py — configuration-matched view of the random-family control (R6).

The random-family control matches cover-row count (400-900) but not the number of
distinct presence configurations, which drives tie structure in low-dimensional binary
spaces. This script recomputes, for the SAME 100 random families already stored in
results/sim_random_families.csv, the number of distinct configurations realised by
their cover rows, and re-examines where the F6 gain sits.

Reads: results/sim_random_families.csv + frozen presence layer (experiment_C1.load()).
Writes: results/exI_configmatch.csv

Run: python3 code/experiment_I_configmatch.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")

from experiment_C1 import load  # noqa: E402

F6_SHORT = ["DO", "BOD", "COD"]


def main() -> None:
    X, rho, paper, params, _Xv, _keysv, _rho_v = load()
    short = {p.split(" (")[0]: i for i, p in enumerate(params)}

    def n_configs(cols):
        cover = X[:, cols].sum(axis=1) >= 1
        return int(len({tuple(X[i, cols].astype(int)) for i in np.where(cover)[0]}))

    def g_for(members):
        cols = [short[m] for m in members]
        return cols

    rows = list(csv.DictReader(open(os.path.join(RESULTS, "sim_random_families.csv"),
                                    encoding="utf-8")))
    out = []
    for r in rows:
        members = [m.strip() for m in r["members"].split("|")]
        try:
            cols = g_for(members)
        except KeyError:
            continue
        out.append(dict(members=r["members"], n_cover=int(r["n_cover"]),
                        n_papers=int(r["n_papers"]), G=float(r["G"]),
                        n_configs=n_configs(cols)))

    f6_cols = [short[m] for m in F6_SHORT]
    f6_cfg = n_configs(f6_cols)
    f6_G = 0.9634  # observed F6 presence-PCA gain (results/exC2_summary.csv)
    G = np.asarray([r["G"] for r in out])
    C = np.asarray([r["n_configs"] for r in out])
    rho_s = spearmanr(C, G)
    print(f"random families parsed: {len(out)} | F6 distinct configurations: {f6_cfg}")
    print(f"configuration count: min={C.min()} median={np.median(C):.0f} max={C.max()}")
    print(f"Spearman(n_configs, G) = {rho_s.statistic:.3f} (p = {rho_s.pvalue:.4g})")
    for thr in (5, 6, 8, 12):
        sub = G[C <= thr]
        if len(sub) < 3:
            print(f"  G | n_configs <= {thr:2d}: n={len(sub)} (too few)")
            continue
        p = (1 + int(np.sum(sub >= f6_G))) / (1 + len(sub))
        print(f"  G | n_configs <= {thr:2d}: n={len(sub):3d} median={np.median(sub):+.3f} "
              f"max={sub.max():+.3f}  F6 percentile={100*(1-p):.1f}  one-sided p={p:.3f}")
    exact = G[C == f6_cfg]
    print(f"  G | n_configs == {f6_cfg}: n={len(exact)} "
          f"(F6 G={f6_G}; exceedances={int(np.sum(exact >= f6_G))})")

    path = os.path.join(RESULTS, "exI_configmatch.csv")
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader(); w.writerows(out)
    os.replace(path + ".tmp", path)
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
