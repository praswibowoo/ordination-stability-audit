#!/usr/bin/env python3
"""stability_blockperm.py — corrections pass (D-corrections): block-aware inference for the Swing Index.

Problem addressed
-----------------
The original C1 permutation (experiment_C1.py) shuffled rho within *deletion-event*
strata. Because row density is constant within 118/122 papers, that null treats
rows as independent units and is not a valid test for a paper-level density
contrast. This script recomputes the C1 Swing Index inference with the paper as
the resampling unit:

  * paper-level permutation p (one- and two-sided), 1000 draws, seed 42
  * cluster bootstrap 95% CI for SI (papers resampled with replacement)
  * carries the original row-level p from results/exC1_summary.csv for contrast

Reads only frozen event-layer records (results/exC1_rows.csv.gz) + presence matrix
for row->paper mapping. Writes results/exC1_blockperm.csv. No analysis input is
modified.

Run: python3 code/stability_blockperm.py
"""
from __future__ import annotations

import csv
import gzip
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEED = 42
N_PERM = 1000
N_BOOT = 1000
PRESENCE = os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv")


def load_row_to_paper() -> np.ndarray:
    """Presence-variant row_id i (0-based data-row index) -> paper (batch_id)."""
    own = ""
    papers = []
    with open(PRESENCE, encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            b = (r["batch_id"] or "").strip()
            if b:
                own = b
            if r["row_kind"] != "data":
                continue
            papers.append(own)
    return np.asarray(papers)


def si(rho: np.ndarray, drift: np.ndarray) -> float:
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    lo = drift[rho <= q1]
    hi = drift[rho >= q3]
    if len(hi) == 0 or np.median(hi) <= 0:
        return float("nan")
    return float(np.median(lo) / np.median(hi))


def load_event_rows():
    """variant -> (rho, drift, paper_code) for LOPO rows."""
    r2p = load_row_to_paper()
    per_variant = {}
    with gzip.open(os.path.join(RESULTS, "exC1_rows.csv.gz"), "rt", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["mode"] != "lopo":
                continue
            if r["variant"].startswith("pca_v"):
                paper = r["row_id"].split("#")[0]
            else:
                paper = r2p[int(r["row_id"])]
            d = per_variant.setdefault(r["variant"], ([], [], []))
            d[0].append(float(r["rho"]))
            d[1].append(float(r["drift"]))
            d[2].append(paper)
    out = {}
    for v, (rho, drift, papers) in per_variant.items():
        rho = np.asarray(rho, float)
        drift = np.asarray(drift, float)
        papers = np.asarray(papers)
        uniq = np.asarray(sorted(set(papers)))
        code = np.searchsorted(uniq, papers)
        # paper-level density (rows within paper are near-constant)
        prho = np.array([rho[code == j].mean() for j in range(len(uniq))])
        idx_by_paper = [np.where(code == j)[0] for j in range(len(uniq))]
        out[v] = dict(rho=rho, drift=drift, code=code, uniq=uniq, prho=prho,
                      idx_by_paper=idx_by_paper, n_papers=len(uniq))
    return out


def block_tests(rho, drift, code, prho, idx_by_paper, rng):
    obs = si(rho, drift)
    # ---- paper-level permutation: reassign paper densities across papers ----
    null = np.empty(N_PERM)
    for b in range(N_PERM):
        rr = rng.permutation(prho)[code]
        null[b] = si(rr, drift)
    p_gt = float((np.sum(null > obs) + 1) / (N_PERM + 1))
    p_lt = float((np.sum(null < obs) + 1) / (N_PERM + 1))
    p_two = float(min(1.0, 2 * min(p_gt, p_lt)))
    # ---- cluster bootstrap over papers ----
    n_p = len(prho)
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        draws = rng.integers(0, n_p, n_p)
        idx = np.concatenate([idx_by_paper[j] for j in draws])
        boots[b] = si(rho[idx], drift[idx])
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return dict(SI=obs, p_block=p_gt, p_block_lt=p_lt, p_block_two=p_two,
                ci_lo=float(lo), ci_hi=float(hi),
                null_mean=float(np.nanmean(null)), null_sd=float(np.nanstd(null)))


def main() -> None:
    variants = load_event_rows()
    rowlevel = {}
    with open(os.path.join(RESULTS, "exC1_summary.csv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["mode"] == "lopo":
                rowlevel[r["variant"]] = r["perm_p"]
    rows = []
    for v in ("pca_pres", "pca_vraw", "pca_vstd", "mds_pres", "tsne_pres"):
        d = variants[v]
        rng = np.random.default_rng(SEED)
        res = block_tests(d["rho"], d["drift"], d["code"], d["prho"],
                          d["idx_by_paper"], rng)
        rows.append(dict(variant=v, n_rows=len(d["rho"]), n_papers=d["n_papers"],
                         SI=round(res["SI"], 4),
                         p_block=round(res["p_block"], 4),
                         p_block_two=round(res["p_block_two"], 4),
                         ci_lo=round(res["ci_lo"], 4), ci_hi=round(res["ci_hi"], 4),
                         null_mean=round(res["null_mean"], 4),
                         null_sd=round(res["null_sd"], 4),
                         p_rowlevel_carried=rowlevel.get(v, "")))
    # Holm across the four defined variants (unstandardized is degenerate)
    defined = [r for r in rows if r["variant"] != "pca_vraw"]
    order = sorted(range(len(defined)), key=lambda i: defined[i]["p_block"])
    m = len(defined)
    prev = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * defined[i]["p_block"])
        prev = max(prev, val)
        defined[i]["holm_block"] = round(prev, 4)
    for r in rows:
        r.setdefault("holm_block", "")
    out = os.path.join(RESULTS, "exC1_blockperm.csv")
    tmp = out + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)
    print(f"[blockperm] wrote {os.path.relpath(out, ROOT)}")
    for r in rows:
        print(f"  {r['variant']:9s} SI={r['SI']:.4f} p_block={r['p_block']:.3f} "
              f"CI=({r['ci_lo']:.3f},{r['ci_hi']:.3f}) p_row={r['p_rowlevel_carried']}")


if __name__ == "__main__":
    main()
