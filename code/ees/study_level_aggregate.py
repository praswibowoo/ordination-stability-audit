#!/usr/bin/env python3
"""study_level_aggregate.py — one-observation-per-study density association.

The row-weighted and per-event SDRs in the main text answer different
questions. This script computes the study-level aggregate used in the
manuscript: one mean density and one mean (or median) drift per study, an SDR
across studies, and a study-level permutation test.

Presence layer
  - drift: per-block-event mean/median Procrustes drift over the retained rows
    (from results/exC1_rows.csv[.gz])
  - density: study mean of the 44-parameter row density

Value layer
  - drift: per-block-event mean/median drift over the retained value-complete
    rows (pca_vstd, from results/exC1_rows.csv[.gz])
  - density: study mean 44-parameter density of the value-complete rows
    (not all rows of the study)

Output: results/study_level_aggregate.csv

Run: python3 code/study_level_aggregate.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_C1 import load  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
SEED = 42
N_PERM = 10_000


def sdr(rho: np.ndarray, drift: np.ndarray) -> float:
    """Sparse/dense ratio of quartile medians."""
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    lo = np.median(drift[rho <= q1])
    hi = np.median(drift[rho >= q3])
    return float(lo / hi) if hi > 0 else float("nan")


def permutation_p(rho: np.ndarray, drift: np.ndarray, obs: float, rng) -> float:
    """Two-sided study-level permutation p for SDR != 1."""
    null = np.empty(N_PERM)
    work = rho.copy()
    for b in range(N_PERM):
        rng.shuffle(work)
        null[b] = sdr(work, drift)
    p1 = (np.sum(null <= obs) + 1.0) / (N_PERM + 1.0)
    p2 = (np.sum(null >= obs) + 1.0) / (N_PERM + 1.0)
    return float(min(1.0, 2.0 * min(p1, p2)))


def presence_events() -> pd.DataFrame:
    """Per-block-event mean/median drift and study mean presence density."""
    rows = pd.read_csv(os.path.join(RESULTS, "exC1_rows.csv.gz"), low_memory=False)
    rows = rows[(rows["variant"] == "pca_pres") & (rows["mode"] == "lopo")].copy()
    events = rows.groupby("deleted").agg(
        mean_drift=("drift", "mean"),
        med_drift=("drift", "median"),
    )
    events.index.name = "paper"

    _, rho, paper, *_ = load()
    density = pd.Series(rho).groupby(np.asarray(paper)).mean().rename("rho")
    return events.join(density, how="inner")


def value_events() -> pd.DataFrame:
    """Per-block-event mean/median drift and value-row study density."""
    rows = pd.read_csv(os.path.join(RESULTS, "exC1_rows.csv.gz"), low_memory=False)
    rows = rows[(rows["variant"] == "pca_vstd") & (rows["mode"] == "lopo")].copy()
    rows["paper"] = rows["row_id"].astype(str).str.split("#").str[0]

    events = rows.groupby("deleted").agg(
        mean_drift=("drift", "mean"),
        med_drift=("drift", "median"),
    )
    events.index.name = "paper"

    # Value-layer density is the 44-parameter coverage density of the
    # value-complete rows, not of every row in the study.
    row_density = rows.drop_duplicates("row_id")[["paper", "rho"]]
    density = row_density.groupby("paper")["rho"].mean().rename("rho")
    return events.join(density, how="inner")


def summarise(events: pd.DataFrame, label: str) -> list[dict]:
    out = []
    for outcome, col in (("mean_drift", "mean_drift"), ("med_drift", "med_drift")):
        rho = events["rho"].to_numpy(float)
        drift = events[col].to_numpy(float)
        obs = sdr(rho, drift)
        p2 = permutation_p(rho, drift, obs, np.random.default_rng(SEED))
        rho_s = float(spearmanr(rho, drift).statistic)
        rho_p = float(spearmanr(rho, drift).pvalue)
        out.append(dict(
            label=label,
            outcome=outcome,
            n=len(events),
            sdr=round(obs, 6),
            p2=round(p2, 6),
            spearman=round(rho_s, 6),
            spearman_p=rho_p,
        ))
    return out


def main() -> None:
    rows = []
    rows += summarise(presence_events(), "presence")
    rows += summarise(value_events(), "value")
    out = pd.DataFrame(rows)
    path = os.path.join(RESULTS, "study_level_aggregate.csv")
    out.to_csv(path, index=False, float_format="%.6g")
    print(out.to_string(index=False))
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
