#!/usr/bin/env python3
"""extraction_confidence_sensitivity.py — F1 value-layer sensitivity.

Drops F1 value rows with any low-confidence F1 cell or any stage-B-incomplete
F1 cell, then reruns the F1 standardized value PCA and LOPO SDR.

Writes results/extraction_confidence_sensitivity.csv
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
F1 = ["DO (mg/L)", "pH", "Salinity (ppt/‰)", "Temperature (°C)"]

from experiment_C1 import load  # noqa: E402


def sdr(rho, drift):
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    return float(np.median(drift[rho <= q1]) / np.median(drift[rho >= q3]))


def main():
    X, rho, paper_all, params, Xv, keys, rho_v = load()
    meta = {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "docs/extraction/stageB/matrix_values_v2.csv"),
            encoding="utf-8-sig")):
        if r["param"] in F1:
            meta[(r["batch_id"], r["row_id"], r["param"])] = (
                r["extraction_confidence"], r["stageB_done"])
    keep = []
    for i, (b, rid) in enumerate(keys):
        ok = all(meta.get((b, rid, p), ("", ""))[0] != "low" and
                 meta.get((b, rid, p), ("", ""))[1] == "TRUE" for p in F1)
        if ok:
            keep.append(i)
    keep = np.asarray(keep)
    Xf = Xv[keep]
    paper = np.asarray([k[0] for k in keys])[keep]
    Z = StandardScaler().fit_transform(Xf)
    S = PCA(n_components=2).fit_transform(Z)
    drifts, rhos = [], []
    for p in sorted(set(paper)):
        m = paper != p
        idx = np.where(m)[0]
        Sr = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Xf[m]))
        a, b, _ = procrustes(S[idx], Sr)
        drifts.append(np.linalg.norm(b - a, axis=1))
        rhos.append(rho_v[keep][m])
    d = np.concatenate(drifts)
    r = np.concatenate(rhos)
    out = os.path.join(RESULTS, "extraction_confidence_sensitivity.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rows", "papers", "sdr", "median_drift"])
        w.writerow([len(Xf), len(set(paper)), round(sdr(r, d), 4),
                    f"{np.median(d):.6g}"])
    print(f"rows={len(Xf)} papers={len(set(paper))} SDR={sdr(r, d):.4f} "
          f"median_drift={np.median(d):.3g}")
    print(f"wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
