#!/usr/bin/env python3
"""experiment_structural_generality.py — structural factorial for the audit.

The manuscript's calibration simulations reuse the real 122-paper geometry.
This script asks a different question: does the block-versus-scattered drift
contrast survive changes in the synthetic meta-matrix structure?

Design
  Presence matrices with P = 30 parameters. Factors:
    papers        30, 60, 122, 250
    rows/paper    3, 6, 12
    panel         repeated (one panel per paper) or varying (row-specific)
    coverage      0.20, 0.50, 0.80
  Parameter co-reporting is generated from a global Dirichlet weight vector,
  so some parameters co-occur across papers (the correlated default). A small
  independent-panel sensitivity is run at the centre of the grid.

  For each cell, fit 2-D presence PCA, run LOPO block deletion and one
  volume-matched scattered deletion per paper, and report the ratio of median
  block drift to median scattered drift. This is an ordination-based structural
  check, not a re-use of the real corpus geometry.

Outputs
  results/structural_generality.csv          per-cell/per-replicate rows
  results/structural_generality_summary.csv  marginal summaries

Run: python3 code/experiment_structural_generality.py [--reps 1]
"""
from __future__ import annotations

import argparse
import csv
import os
import time

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import numpy as np
from scipy.spatial import procrustes
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SEED = 42
P = 30
PAPERS = [30, 60, 122, 250]
ROWS = [3, 6, 12]
PANELS = ["repeated", "varying"]
COVERAGE = [0.20, 0.50, 0.80]


def _save(path: str, header: list[str], rows: list[dict]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def parameter_weights(rng: np.random.Generator, correlated: bool) -> np.ndarray:
    """Global inclusion weights; Dirichlet induces cross-paper co-reporting."""
    if correlated:
        return rng.dirichlet(np.full(P, 0.35))
    return np.full(P, 1.0 / P)


def build_matrix(papers: int, rows: int, panel: str, coverage: float,
                 weights: np.ndarray, rng: np.random.Generator):
    """Return binary presence matrix and paper labels."""
    k = max(1, int(round(coverage * P)))
    X, labels = [], []
    for pi in range(papers):
        block = np.zeros((rows, P), dtype=float)
        if panel == "repeated":
            cols = rng.choice(P, size=k, replace=False, p=weights / weights.sum())
            block[:, cols] = 1.0
        else:
            for ri in range(rows):
                cols = rng.choice(P, size=k, replace=False, p=weights / weights.sum())
                block[ri, cols] = 1.0
        X.append(block)
        labels.extend([pi] * rows)
    return np.vstack(X), np.asarray(labels)


def lopo_ratio(X: np.ndarray, paper: np.ndarray, rng: np.random.Generator):
    """Median block drift / median volume-matched scattered drift."""
    n = len(X)
    if len(np.unique(X, axis=0)) < 3:
        return None
    S_full = PCA(n_components=2, random_state=SEED).fit_transform(X)
    block, scattered = [], []
    for p in range(int(paper.max()) + 1):
        keep = paper != p
        n_del = int((~keep).sum())
        keep_idx = np.where(keep)[0]
        if len(keep_idx) < 4:
            continue
        S_red = PCA(n_components=2, random_state=SEED).fit_transform(X[keep])
        a, b, _ = procrustes(S_full[keep_idx], S_red)
        block.append(float(np.median(np.linalg.norm(b - a, axis=1))))

        scat = rng.choice(n, size=n_del, replace=False)
        keep_s = np.setdiff1d(np.arange(n), scat)
        S_scat = PCA(n_components=2, random_state=SEED).fit_transform(X[keep_s])
        a, b, _ = procrustes(S_full[keep_s], S_scat)
        scattered.append(float(np.median(np.linalg.norm(b - a, axis=1))))

    if not block or not scattered:
        return None
    mb, ms = float(np.median(block)), float(np.median(scattered))
    if ms <= 0:
        return None
    return mb, ms, mb / ms, len(block)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time()
    rows_out: list[dict] = []

    for papers in PAPERS:
        for n_rows in ROWS:
            for panel in PANELS:
                for cov in COVERAGE:
                    for rep in range(args.reps):
                        rng = np.random.default_rng(SEED + 1000 * rep + papers + n_rows)
                        weights = parameter_weights(rng, correlated=True)
                        X, labels = build_matrix(papers, n_rows, panel, cov, weights, rng)
                        out = lopo_ratio(X, labels, rng)
                        rec = dict(papers=papers, rows_per_paper=n_rows, panel=panel,
                                   coverage=cov, correlated=1, rep=rep,
                                   n_rows=int(X.shape[0]), n_events=0,
                                   block_med="", scattered_med="", ratio="",
                                   degenerate=int(out is None))
                        if out is not None:
                            rec.update(n_events=out[3], block_med=f"{out[0]:.8g}",
                                       scattered_med=f"{out[1]:.8g}",
                                       ratio=f"{out[2]:.8g}")
                        rows_out.append(rec)

    # Independent-panel sensitivity at the centre of the grid.
    for panel in PANELS:
        for rep in range(args.reps):
            rng = np.random.default_rng(SEED + 9000 + rep)
            weights = parameter_weights(rng, correlated=False)
            X, labels = build_matrix(122, 6, panel, 0.50, weights, rng)
            out = lopo_ratio(X, labels, rng)
            rec = dict(papers=122, rows_per_paper=6, panel=panel, coverage=0.50,
                       correlated=0, rep=rep, n_rows=int(X.shape[0]), n_events=0,
                       block_med="", scattered_med="", ratio="",
                       degenerate=int(out is None))
            if out is not None:
                rec.update(n_events=out[3], block_med=f"{out[0]:.8g}",
                           scattered_med=f"{out[1]:.8g}", ratio=f"{out[2]:.8g}")
            rows_out.append(rec)

    header = ["papers", "rows_per_paper", "panel", "coverage", "correlated",
              "rep", "n_rows", "n_events", "block_med", "scattered_med",
              "ratio", "degenerate"]
    _save(os.path.join(RESULTS, "structural_generality.csv"), header, rows_out)

    # Marginal summaries over non-degenerate cells.
    valid = [r for r in rows_out if r["degenerate"] == 0 and r["ratio"] != ""]
    summary: list[dict] = []
    for factor, levels in (("papers", PAPERS), ("rows_per_paper", ROWS),
                           ("panel", PANELS), ("coverage", COVERAGE)):
        for level in levels:
            vals = np.asarray([float(r["ratio"]) for r in valid if r[factor] == level])
            if len(vals) == 0:
                continue
            summary.append(dict(factor=factor, level=str(level), n=len(vals),
                                median=f"{np.median(vals):.4f}",
                                q1=f"{np.quantile(vals, .25):.4f}",
                                q3=f"{np.quantile(vals, .75):.4f}",
                                pct_gt1=f"{100 * np.mean(vals > 1):.1f}"))
    vals = np.asarray([float(r["ratio"]) for r in valid])
    summary.append(dict(factor="overall", level="all", n=len(vals),
                        median=f"{np.median(vals):.4f}",
                        q1=f"{np.quantile(vals, .25):.4f}",
                        q3=f"{np.quantile(vals, .75):.4f}",
                        pct_gt1=f"{100 * np.mean(vals > 1):.1f}"))
    _save(os.path.join(RESULTS, "structural_generality_summary.csv"),
          ["factor", "level", "n", "median", "q1", "q3", "pct_gt1"], summary)
    print(f"[structural] {len(rows_out)} cells, {len(valid)} valid, "
          f"{time.time() - t0:.1f}s")
    print(f"[structural] overall ratio median {np.median(vals):.3f} "
          f"(IQR {np.quantile(vals,.25):.3f}-{np.quantile(vals,.75):.3f}), "
          f"{100*np.mean(vals>1):.1f}% > 1")


if __name__ == "__main__":
    main()
