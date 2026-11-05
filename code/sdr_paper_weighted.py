#!/usr/bin/env python3
"""sdr_paper_weighted.py — row-weighted vs per-event paper-weighted SDR.

The manuscript's SDR is defined on row density. A row-weighted observed
statistic cannot be tested against a paper-weighted null. This script computes
three quantities for each ordination variant:

1. the row-weighted SDR (descriptive);
2. a pooled paper-mean SDR (each paper contributes its mean density and mean
   drift; rows are pooled across papers);
3. a per-event paper-weighted SDR: within each leave-one-paper-out event every
   remaining paper contributes one mean density and one mean drift; the SDR is
   computed across papers and the statistic is the median across events. The
   null permutes paper mean densities within each event.

Outputs results/sdr_weighting.csv.

Run: python3 code/sdr_paper_weighted.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEED = 42
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_C1 import load  # noqa: E402


def sdr(rho: np.ndarray, drift: np.ndarray) -> float:
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    return float(np.median(drift[rho <= q1]) / np.median(drift[rho >= q3]))


def paper_labels(variant: str, rows: pd.DataFrame, presence_paper: np.ndarray) -> np.ndarray:
    if variant.startswith("pca_v"):
        return rows["row_id"].astype(str).str.split("#").str[0].values
    return presence_paper[rows["row_id"].astype(int).values]


def main() -> None:
    _rows = os.path.join(RESULTS, "exC1_rows.csv")
    if not os.path.exists(_rows):
        _rows += ".gz"
    rows = pd.read_csv(_rows, low_memory=False)
    # Read the staged extraction layers (stage-layers), not the release path:
    # the GitHub code mirrors the data record but does not keep a v1.1 copy.
    _, _, presence_paper, _, _, _, _ = load()

    out = []
    for variant in ("pca_pres", "pca_vraw", "pca_vstd", "mds_pres", "tsne_pres"):
        x = rows[(rows["variant"] == variant) & (rows["mode"] == "lopo")].copy()
        if x.empty:
            continue
        x["paper"] = paper_labels(variant, x, presence_paper)
        rho = x["rho"].astype(float).values
        drift = x["drift"].astype(float).values

        row_sdr = sdr(rho, drift)
        pooled = x.groupby("paper").agg(rho=("rho", "mean"), drift=("drift", "mean"))
        pooled_sdr = sdr(pooled["rho"].values, pooled["drift"].values)

        events = []
        for _, event in x.groupby("deleted"):
            by_paper = event.groupby("paper").agg(
                rho=("rho", "mean"), drift=("drift", "mean")
            )
            if len(by_paper) >= 4:
                events.append((by_paper["rho"].values, by_paper["drift"].values))
        event_sdr = float(np.nanmedian([sdr(r, d) for r, d in events]))

        rng = np.random.default_rng(SEED)
        null = np.empty(1000)
        for b in range(1000):
            null[b] = np.nanmedian(
                [sdr(rng.permutation(r), d) for r, d in events]
            )
        p1 = float((np.sum(null >= event_sdr) + 1) / 1001)
        p2 = float(2 * min(p1, (np.sum(null <= event_sdr) + 1) / 1001))

        boot = []
        for _ in range(2000):
            idx = rng.integers(0, len(events), len(events))
            boot.append(np.nanmedian([sdr(*events[i]) for i in idx]))
        ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

        out.append(dict(
            variant=variant,
            n_rows=len(x),
            n_papers=len(pooled),
            n_events=len(events),
            row_sdr=round(row_sdr, 4),
            pooled_paper_sdr=round(pooled_sdr, 4),
            event_paper_sdr=round(event_sdr, 4),
            event_paper_p=round(p1, 4),
            event_paper_p_two=round(p2, 4),
            event_paper_ci_lo=round(float(ci_lo), 4),
            event_paper_ci_hi=round(float(ci_hi), 4),
            null_mean=round(float(null.mean()), 4),
            null_sd=round(float(null.std()), 4),
        ))

    path = os.path.join(RESULTS, "sdr_weighting.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        writer.writeheader()
        writer.writerows(out)
    print(f"[sdr-weighting] wrote {os.path.relpath(path, ROOT)}")
    for row in out:
        print(row)


if __name__ == "__main__":
    main()
