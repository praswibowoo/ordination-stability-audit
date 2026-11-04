#!/usr/bin/env python3
"""r2_per_event_calibration.py — R2: size and power of the per-event paper-weighted SDR.

The manuscript's headline density test is the per-event paper-weighted SDR
(median across deletion events of the sparse/dense paper-quartile drift ratio),
with paper mean densities permuted within each event. The existing simulation
calibrates the row-level and pooled paper-level tests but not this estimator.

This script reuses the real 122-paper structure and the drift generator of
experiment_S_validation.py, then runs the per-event estimator and its
within-event permutation null.

Reports rejection rates at alpha = 0.05:
  size   at gamma = 0 (no planted density effect), sigma = 0.8
  power  at gamma = 0.2 and 0.6 (the contrasts used in the paper)

Writes results/r2_per_event_calibration.csv.

Run: python3 code/r2_per_event_calibration.py [--rep 500] [--events 30] [--perm 199]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_C1 import load  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
SEED = 42


def sdr(rho, drift):
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    lo = drift[rho <= q1]
    hi = drift[rho >= q3]
    if len(lo) == 0 or len(hi) == 0:
        return np.nan
    a, b = np.median(lo), np.median(hi)
    return float(a / b) if b > 0 else np.nan


def run(rep: int, n_events: int, n_perm: int):
    X, rho, paper, params, Xv, keys, rho_v = load()
    uniq = np.asarray(sorted(set(paper)))
    code = np.searchsorted(uniq, paper)
    prho = np.array([rho[code == j].mean() for j in range(len(uniq))])
    med = float(np.median(prho))
    idx = {p: np.where(paper == p)[0] for p in uniq}

    rows = []
    for sigma, gamma in ((0.8, 0.0), (0.8, 0.2), (0.8, 0.6)):
        rej = 0
        for r in range(rep):
            rng = np.random.default_rng(SEED + 10000 * r + int(1000 * gamma))
            lam = np.exp(gamma * (med - prho))[code]
            u = rng.standard_normal(len(uniq))
            drift = lam * np.exp(sigma * u[code] + 0.4 * rng.standard_normal(len(rho)))

            ev = rng.choice(len(uniq), size=min(n_events, len(uniq)), replace=False)
            obs_ev, null_ev = [], []
            for j in ev:
                keep = np.where(code != j)[0]
                # one mean per remaining paper
                papers = np.unique(code[keep])
                rp = np.array([rho[code == p].mean() for p in papers])
                dp = np.array([drift[code == p].mean() for p in papers])
                obs_ev.append(sdr(rp, dp))
                null_ev.append((rp, dp))
            obs = float(np.nanmedian(obs_ev))

            null = np.empty(n_perm)
            for b in range(n_perm):
                vals = [sdr(rng.permutation(rp), dp) for rp, dp in null_ev]
                null[b] = float(np.nanmedian(vals))
            p = float((np.sum(null >= obs) + 1) / (n_perm + 1))
            rej += int(p < 0.05)
        rate = rej / rep
        se = float(np.sqrt(rate * (1 - rate) / rep))
        rows.append(dict(sigma_paper=sigma, gamma=gamma, n_rep=rep,
                         n_events=n_events, n_perm=n_perm,
                         rejection_rate=round(rate, 4), mc_se=round(se, 4)))
        print(f"[r2] sigma={sigma} gamma={gamma} | per-event rejection "
              f"{rate:.3f} (+-{se:.3f}) n={rep} events={n_events} perm={n_perm}")

    path = os.path.join(RESULTS, "r2_per_event_calibration.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[r2] wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", type=int, default=500)
    ap.add_argument("--events", type=int, default=30)
    ap.add_argument("--perm", type=int, default=199)
    a = ap.parse_args()
    run(a.rep, a.events, a.perm)
