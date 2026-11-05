#!/usr/bin/env python3
"""churn_corrected.py — corrections pass: churn units and mode separation.

experiment_C1.knn_churn returns |A ^ B| / k (symmetric difference over k), which
ranges 0..2 and equals twice the fraction of replaced neighbours. Figure 2c also
pooled LOPO and LORO rows while titled "LOPO". This script recomputes the churn
summaries with the stated-units correction (fraction = chi/2) separately for
LOPO and LORO, overall and by row-density quartile.

Reads results/exC1_rows.csv.gz only. Writes results/exC1_churn_corrected.csv.

Run: python3 code/core/churn_corrected.py
"""
from __future__ import annotations

import csv
import gzip
import os

import numpy as np

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "exC1_churn_corrected.csv")


def main() -> None:
    data = {}
    with gzip.open(os.path.join(RESULTS, "exC1_rows.csv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            d = data.setdefault((r["variant"], r["mode"]), ([], []))
            d[0].append(float(r["rho"]))
            d[1].append(float(r["churn"]))
    rows = []
    for (variant, mode), (rho, ch) in sorted(data.items()):
        rho = np.asarray(rho)
        ch = np.asarray(ch)
        q1, q2, q3 = np.quantile(rho, [0.25, 0.5, 0.75])
        masks = [rho <= q1, (rho > q1) & (rho <= q2),
                 (rho > q2) & (rho <= q3), rho > q3]
        row = dict(variant=variant, mode=mode, n=len(ch),
                   mean_chi=round(float(ch.mean()), 4),
                   median_chi=round(float(np.median(ch)), 4),
                   mean_frac=round(float((ch / 2).mean()), 4),
                   median_frac=round(float(np.median(ch / 2)), 4),
                   q1=round(float(q1), 4), q3=round(float(q3), 4))
        for i, m in enumerate(masks):
            row[f"median_frac_Q{i+1}"] = round(float(np.median(ch[m] / 2)), 4)
            row[f"n_Q{i+1}"] = int(m.sum())
        rows.append(row)
    with open(OUT + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(OUT + ".tmp", OUT)
    print(f"[churn] wrote {os.path.relpath(OUT, ROOT)}")
    for r in rows:
        if r["variant"] in ("pca_pres", "mds_pres", "tsne_pres"):
            print(f"  {r['variant']:9s} {r['mode']:4s} n={r['n']:>6d} "
                  f"median chi={r['median_chi']:.2f} -> fraction={r['median_frac']:.2f} "
                  f"| LOPO quartile fractions "
                  f"{r['median_frac_Q1']:.2f}/{r['median_frac_Q2']:.2f}/"
                  f"{r['median_frac_Q3']:.2f}/{r['median_frac_Q4']:.2f}")


if __name__ == "__main__":
    main()
