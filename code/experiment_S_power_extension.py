#!/usr/bin/env python3
"""experiment_S_power_extension.py — extend the power sweep beyond gamma=0.6.

The main sweep stops at a 5.94% mean-drift contrast and never reaches 80%
paper-level power. This script runs larger planted effects to locate the
minimum detectable effect, or to show that the test remains underpowered even
at large contrasts.

Writes results/sim_calibration_extended.csv
Run: python3 code/experiment_S_power_extension.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")

from experiment_C1 import load  # noqa: E402
from experiment_S_validation import calibration, drift_contrast_pct  # noqa: E402


def main():
    X, rho, paper, *_ = load()
    cal, meta = calibration(
        X, rho, paper,
        scenarios=((0.8, 1.0), (0.8, 1.5), (0.8, 2.0)),
        n_rep=500, n_perm=199)
    out = os.path.join(RESULTS, "sim_calibration_extended.csv")
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cal[0].keys()))
        w.writeheader()
        w.writerows(cal)
    os.replace(out + ".tmp", out)
    summ = []
    for g in (1.0, 1.5, 2.0):
        sub = [r for r in cal if float(r["gamma"]) == g]
        summ.append(dict(sigma_paper=0.8, gamma=g, n_rep=len(sub),
                         rej_row=float(np.mean([r["rej_row"] for r in sub])),
                         rej_block=float(np.mean([r["rej_block"] for r in sub])),
                         contrast_pct=float(drift_contrast_pct(g, meta["delta_rho"]))))
    sm = os.path.join(RESULTS, "sim_calibration_extended_summary.csv")
    with open(sm + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
    os.replace(sm + ".tmp", sm)
    for g in (1.0, 1.5, 2.0):
        sub = [r for r in cal if float(r["gamma"]) == g]
        print(f"gamma={g}: row {np.mean([r['rej_row'] for r in sub]):.3f} / "
              f"paper {np.mean([r['rej_block'] for r in sub]):.3f} | "
              f"contrast {drift_contrast_pct(g, meta['delta_rho']):.2f}%")
    print(f"[power-extension] wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
