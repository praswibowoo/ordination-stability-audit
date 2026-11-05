#!/usr/bin/env python3
"""sdr_hierarchical.py — hierarchical bootstrap for the paper-weighted SDR.

The per-event paper-weighted SDR uses each paper's mean density and mean drift
as fixed. This script adds a three-level bootstrap:

  events -> papers within event -> rows within paper

and reports the 95% interval for the median across events. It also reports a
two-level bootstrap (events + papers, no row resampling) to show how much of
the interval comes from row-level sampling.

Outputs results/sdr_hierarchical.csv.

Run: python3 code/core/sdr_hierarchical.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS = os.path.join(ROOT, "results")
SEED = 42
N_BOOT = 2000
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_C1 import load  # noqa: E402


def sdr(rho: np.ndarray, drift: np.ndarray) -> float:
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    return float(np.median(drift[rho <= q1]) / np.median(drift[rho >= q3]))


def load_events(variant: str):
    _rows = os.path.join(RESULTS, "exC1_rows.csv")
    if not os.path.exists(_rows):
        _rows += ".gz"
    rows = pd.read_csv(_rows, low_memory=False)
    # Read the staged extraction layers (stage-layers), not the release path.
    _, _, pres_paper, _, _, _, _ = load()
    x = rows[(rows["variant"] == variant) & (rows["mode"] == "lopo")].copy()
    if variant.startswith("pca_v"):
        x["paper"] = x["row_id"].astype(str).str.split("#").str[0]
    else:
        x["paper"] = pres_paper[x["row_id"].astype(int).values]
    events = []
    for _, ev in x.groupby("deleted"):
        by_paper = {}
        for paper, g in ev.groupby("paper"):
            by_paper[paper] = (g["rho"].astype(float).values,
                               g["drift"].astype(float).values)
        if len(by_paper) >= 4:
            events.append(by_paper)
    return events


def event_sdr(by_paper, rng, resample_rows: bool) -> float:
    papers = list(by_paper)
    idx = rng.integers(0, len(papers), len(papers))
    rho, drift = [], []
    for j in idx:
        r, d = by_paper[papers[j]]
        if resample_rows:
            take = rng.integers(0, len(r), len(r))
            r, d = r[take], d[take]
        rho.append(float(np.mean(r)))
        drift.append(float(np.mean(d)))
    return sdr(np.asarray(rho), np.asarray(drift))


def observed_event_sdr(by_paper) -> float:
    r = np.asarray([np.mean(v[0]) for v in by_paper.values()])
    d = np.asarray([np.mean(v[1]) for v in by_paper.values()])
    return sdr(r, d)


def bootstrap(events, rng, resample_rows: bool) -> np.ndarray:
    out = np.empty(N_BOOT)
    for b in range(N_BOOT):
        ev_idx = rng.integers(0, len(events), len(events))
        vals = [event_sdr(events[j], rng, resample_rows) for j in ev_idx]
        out[b] = float(np.nanmedian(vals))
    return out


def main() -> None:
    rows = []
    for variant in ("pca_pres", "pca_vstd", "mds_pres", "tsne_pres"):
        events = load_events(variant)
        point = float(np.nanmedian([observed_event_sdr(ev) for ev in events]))
        rng = np.random.default_rng(SEED)
        three = bootstrap(events, rng, True)
        two = bootstrap(events, rng, False)
        rows.append(dict(
            variant=variant,
            point=round(point, 4),
            three_level_lo=round(float(np.percentile(three, 2.5)), 4),
            three_level_hi=round(float(np.percentile(three, 97.5)), 4),
            two_level_lo=round(float(np.percentile(two, 2.5)), 4),
            two_level_hi=round(float(np.percentile(two, 97.5)), 4),
            n_events=len(events),
            n_boot=N_BOOT,
        ))
        print(rows[-1])
    path = os.path.join(RESULTS, "sdr_hierarchical.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[sdr-hierarchical] wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
