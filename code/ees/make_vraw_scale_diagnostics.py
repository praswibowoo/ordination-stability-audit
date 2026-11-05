#!/usr/bin/env python3
"""make_vraw_scale_diagnostics.py — unstandardized value-PCA scale diagnostics (D199).

Backs three manuscript numbers that no artifact previously produced:
  - Section 3.2: "salinity has variance 78,900 against 18 for the next highest
    parameter and accounts for 99.97% of total variance, pinning the leading
    axis with a loading of 1.000"
  - Table S3 note: "salinity accounts for 99.97% of sample variance"

The unstandardized value PCA (`pca_vraw` in Experiment A) is fitted on the
value-complete F1-physical rows — exactly the matrix `experiment_C1.load`
returns as `Xv`. This script computes per-parameter variance, each parameter's
share of total variance, and the leading-axis loadings, then asserts the four
numbers the manuscript quotes so they cannot drift silently.

Reads (via the shipped loader): docs/extraction/stageA/matrix_presence_v2.csv,
docs/extraction/stageB/matrix_values_v2.csv, docs/family_map.csv (frozen layers).
Writes: results/exC1_vraw_scale.csv  (atomic)

Deterministic (no randomness). Run: python3 code/ees/make_vraw_scale_diagnostics.py
"""
from __future__ import annotations
import csv
import os
import sys

import numpy as np

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
from phase4_freeze import verify_baseline  # noqa: E402
from experiment_C1 import load, F1_PARAMS  # noqa: E402

OUT = os.path.join(ROOT, "results", "exC1_vraw_scale.csv")
FIELDS = ["rank", "param", "variance", "share_pct", "pc1_loading"]


def main() -> int:
    verify_baseline()
    _, _, _, _, Xv, _, _ = load()
    assert Xv.shape[1] == len(F1_PARAMS), (Xv.shape, F1_PARAMS)
    n = Xv.shape[0]

    var = Xv.var(axis=0)
    total = float(var.sum())
    share = var / total * 100.0
    # leading component of the unstandardized fit (sklearn PCA == SVD of the centred matrix)
    Xc = Xv - Xv.mean(axis=0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    load1 = vt[0]

    order = np.argsort(-var)
    rows = [dict(rank=i + 1, param=F1_PARAMS[j], variance=float(var[j]),
                 share_pct=round(float(share[j]), 4), pc1_loading=round(float(load1[j]), 4))
            for i, j in enumerate(order)]

    top, second = rows[0], rows[1]
    # --- the four numbers the manuscript quotes (Section 3.2, Table S3 note) ---
    assert top["param"] == "Salinity (ppt/‰)", top
    assert second["param"] == "Temperature (°C)", second
    assert abs(top["variance"] / 78900.0 - 1.0) < 1e-3, top["variance"]        # quoted "78,900"
    assert abs(second["variance"] - 18.0) < 0.05, second["variance"]          # quoted "18"
    assert 99.96 <= top["share_pct"] <= 99.98, top["share_pct"]               # quoted "99.97%"
    assert abs(abs(top["pc1_loading"]) - 1.0) < 1e-3, top["pc1_loading"]      # quoted "1.000"

    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)

    print(f"unstandardized value PCA: n={n} rows x {len(F1_PARAMS)} parameters")
    for r in rows:
        print(f"  {r['rank']}. {r['param']:22s} variance={r['variance']:12.1f} "
              f"share={r['share_pct']:7.2f}%  PC1 loading={r['pc1_loading']:+.3f}")
    print(f"top-2 ratio = {top['variance'] / second['variance']:.0f}x  "
          f"(\u2248{top['variance'] / 1000:.1f}e3 vs {second['variance']:.1f})")
    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
