#!/usr/bin/env python3
"""ees_rev3_splits.py — EES rev3 audit splits.

Adds three descriptive splits that the rev3 manuscript needs but did not report:
  1. repeated-panel vs varying-panel studies, presence layer;
  2. single-row vs multi-row studies, presence layer;
  3. the repeated-panel fraction of the standardized value layer.

It also fits the same log-linear block model within each split and runs a
drop-substitutes sensitivity (B238--B240) for the presence layer.

Reads:
  results/exH_volumematch.csv
  results/exH_volumematch_draws.csv
  results/exH_value_volumematch.csv
  frozen presence layer via experiment_C1.load()

Writes:
  results/ees_rev3_splits.csv

Run: python3 code/ees/ees_rev3_splits.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_C1 import load  # noqa: E402


def cluster_boot_block(ev: pd.DataFrame, n_boot: int = 2000, seed: int = SEED):
    """Fit log(drift) ~ 1 + log(V) + block with a paper-level cluster bootstrap."""
    y = np.log(ev["med_drift"].to_numpy(float))
    v = np.log(ev["n_rows_deleted"].to_numpy(float))
    b = (ev["arm"] == "lopo").to_numpy(float)
    X = np.column_stack([np.ones(len(y)), v, b])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)

    uniq = np.asarray(sorted(set(ev["paper"])), dtype=object)
    gid = pd.Categorical(ev["paper"], categories=uniq).codes
    idx = {j: np.where(gid == j)[0] for j in range(len(uniq))}
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[j] for j in pick])
        try:
            boots.append(np.linalg.lstsq(X[sel], y[sel], rcond=None)[0])
        except np.linalg.LinAlgError:
            continue
    boots = np.asarray(boots)
    lo, hi = np.quantile(boots[:, 2], [0.025, 0.975])
    return beta[2], lo, hi, len(boots), len(y)


def paired_summary(df: pd.DataFrame):
    """Block vs matched-first descriptive summary and paired bootstrap CI."""
    d_l = df["lopo_drift"].to_numpy(float)
    d_m = df["matched_drift"].to_numpy(float)
    ratio = d_l / d_m
    rng = np.random.default_rng(SEED)
    boot = np.empty(10000)
    for i in range(10000):
        idx = rng.integers(0, len(ratio), len(ratio))
        boot[i] = np.median(ratio[idx])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = stats.wilcoxon(d_l, d_m).pvalue if len(ratio) >= 5 else np.nan
    return dict(
        n_papers=len(df),
        block_med=float(np.median(d_l)),
        matched_med=float(np.median(d_m)),
        ratio_of_medians=float(np.median(d_l) / np.median(d_m)),
        median_paired_ratio=float(np.median(ratio)),
        paired_ci_lo=float(lo),
        paired_ci_hi=float(hi),
        wilcoxon_p=float(p),
    )


def main() -> None:
    X, rho, paper, params, *_ = load()
    paper = np.asarray(paper)
    papers = sorted(set(paper))

    # Per-paper presence-pattern repetition and row count.
    meta = []
    for p in papers:
        rows = X[paper == p]
        repeated = bool(np.all(rows == rows[0]))
        meta.append(dict(paper=p, n_rows=len(rows), repeated=repeated,
                         single_row=len(rows) == 1))
    meta = pd.DataFrame(meta)

    lopo = pd.read_csv(os.path.join(RESULTS, "exH_volumematch.csv"))
    lopo = lopo[lopo["arm"] == "lopo"][["paper", "med_drift", "n_rows_deleted"]]
    matched = pd.read_csv(os.path.join(RESULTS, "exH_volumematch.csv"))
    # exH_volumematch.csv keeps exactly one first-draw matched row per paper;
    # the full draw set is in exH_volumematch_draws.csv.
    matched = matched[matched["arm"] == "matched_scattered"]
    matched = matched[["paper", "med_drift"]].rename(columns={"med_drift": "matched_drift"})
    lopo = lopo.rename(columns={"med_drift": "lopo_drift"})
    df = meta.merge(lopo, on="paper", how="inner").merge(matched, on="paper", how="inner")
    assert len(df) == 122, len(df)

    draws = pd.read_csv(os.path.join(RESULTS, "exH_volumematch_draws.csv"))
    lopo_ev = pd.read_csv(os.path.join(RESULTS, "exH_volumematch.csv"))
    lopo_ev = lopo_ev[lopo_ev["arm"] == "lopo"][["paper", "med_drift", "n_rows_deleted", "arm"]]
    ev = pd.concat([lopo_ev, draws], ignore_index=True)

    out = []

    def add_split(name, group, sub):
        s = paired_summary(sub)
        s["n_rows"] = int(sub["n_rows"].sum())
        s["repeated_papers"] = int(sub["repeated"].sum())
        s["repeated_fraction"] = float(sub["repeated"].mean())
        ev_sub = ev[ev["paper"].isin(sub["paper"])]
        b, lo, hi, n_ok, n_obs = cluster_boot_block(ev_sub)
        out.append(dict(analysis=name, group=group, **s,
                        block_factor=float(np.exp(b)),
                        block_ci_lo=float(np.exp(lo)),
                        block_ci_hi=float(np.exp(hi)),
                        n_events=n_obs,
                        n_boot=n_ok))

    add_split("panel_structure", "repeated", df[df["repeated"]])
    add_split("panel_structure", "varying", df[~df["repeated"]])
    add_split("row_count", "single_row", df[df["single_row"]])
    add_split("row_count", "multi_row", df[~df["single_row"]])

    # Drop the three substitute papers from the presence block model.
    keep = ~ev["paper"].isin(["B238", "B239", "B240"])
    b, lo, hi, n_ok, n_obs = cluster_boot_block(ev[keep])
    sub = df[~df["paper"].isin(["B238", "B239", "B240"])]
    s = paired_summary(sub)
    s["n_rows"] = int(sub["n_rows"].sum())
    s["repeated_papers"] = int(sub["repeated"].sum())
    s["repeated_fraction"] = float(sub["repeated"].mean())
    out.append(dict(analysis="drop_substitutes", group="B238-B240 removed", **s,
                    block_factor=float(np.exp(b)),
                    block_ci_lo=float(np.exp(lo)),
                    block_ci_hi=float(np.exp(hi)),
                    n_events=n_obs, n_boot=n_ok))

    # Value-layer repeated-panel fraction: use the 70 value-complete papers and
    # the 44-parameter presence patterns that define a "panel".
    vdf = pd.read_csv(os.path.join(RESULTS, "exH_value_volumematch.csv"))
    vpapers = sorted(set(vdf["paper"]))
    vmeta = []
    for p in vpapers:
        rows = X[paper == p]
        if len(rows) == 0:
            continue
        vmeta.append(dict(paper=p, n_rows=len(rows),
                          repeated=bool(np.all(rows == rows[0]))))
    vmeta = pd.DataFrame(vmeta)
    out.append(dict(analysis="value_layer_panel", group="value_complete_papers",
                    n_papers=len(vmeta), n_rows=int(vmeta["n_rows"].sum()),
                    repeated_fraction=float(vmeta["repeated"].mean()),
                    repeated_papers=int(vmeta["repeated"].sum()),
                    block_med=np.nan, matched_med=np.nan,
                    ratio_of_medians=np.nan, median_paired_ratio=np.nan,
                    paired_ci_lo=np.nan, paired_ci_hi=np.nan,
                    wilcoxon_p=np.nan, block_factor=np.nan,
                    block_ci_lo=np.nan, block_ci_hi=np.nan,
                    n_events=np.nan, n_boot=np.nan))

    cols = ["analysis", "group", "n_papers", "n_rows", "repeated_papers",
            "repeated_fraction", "block_med", "matched_med", "ratio_of_medians",
            "median_paired_ratio", "paired_ci_lo", "paired_ci_hi", "wilcoxon_p",
            "block_factor", "block_ci_lo", "block_ci_hi", "n_events", "n_boot"]
    out_df = pd.DataFrame(out).reindex(columns=cols)
    path = os.path.join(RESULTS, "ees_rev3_splits.csv")
    out_df.to_csv(path, index=False, float_format="%.6g")
    print(out_df.to_string(index=False))
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
