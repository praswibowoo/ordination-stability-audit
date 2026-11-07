#!/usr/bin/env python3
"""coverage_paired_ratio_rules.py — one paired-ratio rule for all coverage arms.

The manuscript uses a first-draw paired ratio for coverage and a median-of-draws
ratio for composition/density/value.  This script reports both rules for the
volume-matched coverage arm so the main text can use one rule and report the
other as a sensitivity.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
R = ROOT / "results"
SEED = 42


def load():
    block = {}
    with (R / "exH_volumematch.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["arm"] == "lopo":
                block[r["paper"]] = float(r["med_drift"])
    draws = {}
    with (R / "exH_volumematch_draws.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["arm"] == "matched_scattered":
                draws.setdefault(r["paper"], []).append(float(r["med_drift"]))
    return block, draws


def boot_ci(values, n_boot=10000, seed=SEED):
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(values), len(values))
        boot[i] = np.median(values[idx])
    return np.quantile(boot, [0.025, 0.975])


def main():
    block, draws = load()
    papers = sorted(block)
    first = np.array([block[p] / draws[p][0] for p in papers])
    med = np.array([block[p] / np.median(draws[p]) for p in papers])
    rows = []
    for label, values in (("first_draw", first), ("median_of_draws", med)):
        lo, hi = boot_ci(values)
        rows.append(dict(
            rule=label,
            n_papers=len(values),
            median_paired_ratio=float(np.median(values)),
            ci_lo=float(lo),
            ci_hi=float(hi),
        ))
    out = R / "coverage_paired_ratio_rules.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(out.relative_to(ROOT))
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
