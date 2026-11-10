#!/usr/bin/env python3
"""row_count_sensitivity.py — row-count threshold sensitivity for the paired ratio.

Computes the median per-study block/matched ratio after excluding studies above
a fixed row-count threshold (not selected on the outcome). This replaces the
outcome-selected "exclude the four largest events" check.

Reads: results/wp3_clean/coverage_events.csv
Writes: results/wp3_clean/row_count_sensitivity.csv
Run: python3 code/row_count_sensitivity.py
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "wp3_clean")
SEED = 42


def main() -> None:
    df = pd.read_csv(os.path.join(RESULTS, "coverage_events.csv"))
    rows = []
    for paper, g in df.groupby("paper"):
        block = g[g["arm"] == "block"]
        matched = g[g["arm"] == "matched"]
        if len(block) and len(matched):
            b = float(block["med_drift"].median())
            m = float(matched["med_drift"].median())
            rows.append((paper, b / m, int(block["n_rows_deleted"].iloc[0])))
    out = pd.DataFrame(rows, columns=["paper", "ratio", "rows"])
    rng = np.random.default_rng(SEED)
    records = []
    for threshold in (10, 12, 14, 20):
        sub = out[out["rows"] <= threshold]
        boot = [np.median(rng.choice(sub["ratio"].to_numpy(), len(sub), replace=True))
                for _ in range(10000)]
        records.append(dict(threshold_rows=threshold, n_studies=len(sub),
                            median_ratio=float(sub["ratio"].median()),
                            ci_lo=float(np.quantile(boot, 0.025)),
                            ci_hi=float(np.quantile(boot, 0.975))))
    path = os.path.join(RESULTS, "row_count_sensitivity.csv")
    pd.DataFrame(records).to_csv(path, index=False, float_format="%.6g")
    print(f"wrote {os.path.relpath(path, ROOT)}")
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
