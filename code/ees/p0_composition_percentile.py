#!/usr/bin/env python3
"""p0_composition_percentile.py — composition-matched block percentiles (P0-5).

Reads the curated block events and the curated composition/density-matched
draws, and reports the per-study block percentile and 95th/5th exceedance for
the volume-matched, composition-matched, density-only, and exact-match arms.

Run: python3 code/p0_composition_percentile.py
Out: results/wp3_clean/p0_composition_percentile.csv
"""
from __future__ import annotations
import csv, os
from collections import defaultdict
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENTS = os.path.join(ROOT, "results", "wp3_clean", "coverage_events.csv")
DRAWS = os.path.join(ROOT, "results", "wp3_clean", "composition_matched_control.csv")
OUT = os.path.join(ROOT, "results", "wp3_clean", "p0_composition_percentile.csv")


def read_block():
    block = {}
    with open(EVENTS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "block":
                block[r["paper"]] = float(r["med_drift"])
    return block


def read_draws():
    comp, dens, exact = defaultdict(list), defaultdict(list), set()
    with open(DRAWS, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            p = r["paper"]
            if r["arm"] == "composition_matched":
                comp[p].append(float(r["med_drift"]))
                if float(r["exact_fraction"]) > 0:
                    exact.add(p)
            elif r["arm"] == "density_only":
                dens[p].append(float(r["med_drift"]))
    return comp, dens, exact


def summarise(name, block, draws, papers):
    pct, above, below = [], 0, 0
    for p in papers:
        b = block.get(p)
        if b is None or not draws.get(p):
            continue
        arr = np.asarray(draws[p], float)
        pct.append(float((arr < b).mean()))
        if b > np.quantile(arr, 0.95):
            above += 1
        if b < np.quantile(arr, 0.05):
            below += 1
    n = len(pct)
    return dict(arm=name, n_papers=n,
                median_percentile=float(np.median(pct)) if n else "",
                above95_n=above, above95_pct=100.0 * above / n if n else "",
                below5_n=below, below5_pct=100.0 * below / n if n else "")


def main():
    block = read_block()
    comp, dens, exact = read_draws()
    all_papers = sorted(block)
    # Volume-matched percentile needs the volume-matched draws, which are not
    # in this file; use the manuscript's existing value for the summary row.
    rows = [
        dict(arm="volume_matched (manuscript)", n_papers=117,
             median_percentile=76.0, above95_n=34, above95_pct=29.06,
             below5_n=7, below5_pct=5.98),
        summarise("composition_matched", block, comp, all_papers),
        summarise("density_only", block, dens, all_papers),
        summarise("exact_match_subset", block, comp, sorted(exact)),
    ]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
