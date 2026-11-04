#!/usr/bin/env python3
"""f6_completeness_sensitivity.py — F6 cover-row vs completeness sensitivities.

The F6 family-restriction result in the main text uses cover rows (at least one
of DO, BOD, COD present). This script asks whether the 60,900-fold presence-PCA
ratio survives when the fit is restricted to rows that report the full family:

  * presence-complete rows: all three parameters present. The 3-column binary
    matrix is constant, so the family ordination and the ratio are undefined.
  * value-complete rows: all three parameters carry real numbers. Fit a
    standardized 3-parameter PCA, run LOPO, and report median Procrustes drift
    and median kNN churn (descriptive; no all-parameter ratio is defined because
    no row reports all 44 parameters).

Reads frozen layers only. Writes results/f6_completeness_sensitivity.csv.
Run: python3 code/f6_completeness_sensitivity.py
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.spatial import procrustes as sp_procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42
F6 = ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)"]

from experiment_C1 import load, knn_idx, knn_churn  # noqa: E402


def load_f6_values() -> tuple[np.ndarray, np.ndarray]:
    """Return (104 x 3) value matrix and paper labels for F6 value-complete rows."""
    own = ""
    ordinal_to_paper: dict[int, str] = {}
    ordinal = 0
    with open(os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv"),
              encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            batch = (row["batch_id"] or "").strip()
            if batch:
                own = batch
            ordinal += 1
            if row["row_kind"] == "data":
                ordinal_to_paper[ordinal] = own

    cells: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    with open(os.path.join(ROOT, "docs/extraction/stageB/matrix_values_v2.csv"),
              encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if row["is_real_value"].strip() != "TRUE" or not (row["value"] or "").strip():
                continue
            try:
                value = float(row["value"])
            except ValueError:
                continue
            cells[(row["batch_id"], int(row["row_id"]))][row["param"]] = value

    values, papers = [], []
    for (batch, ordinal), cell in cells.items():
        if all(param in cell for param in F6) and ordinal in ordinal_to_paper:
            values.append([cell[param] for param in F6])
            papers.append(ordinal_to_paper[ordinal])
    return np.asarray(values, float), np.asarray(papers)


def fit(M: np.ndarray) -> np.ndarray:
    return PCA(n_components=2, random_state=SEED).fit_transform(
        StandardScaler().fit_transform(M)
    )


def value_sensitivity() -> dict:
    Xv, papers = load_f6_values()
    scores = fit(Xv)
    knn = knn_idx(scores)
    rows = []
    for paper in sorted(set(papers)):
        keep = papers != paper
        if keep.sum() < 8:
            continue
        keep_idx = np.where(keep)[0]
        reduced = fit(Xv[keep])
        a, b, _ = sp_procrustes(scores[keep], reduced)
        drift = np.linalg.norm(b - a, axis=1)
        churn = knn_churn(knn[keep_idx], keep_idx[knn_idx(reduced)]) / 2.0
        rows.append((float(np.median(drift)), float(np.median(churn))))

    drift = np.asarray([r[0] for r in rows], float)
    churn = np.asarray([r[1] for r in rows], float)
    rng = np.random.default_rng(SEED)
    boot_drift, boot_churn = [], []
    for _ in range(2000):
        idx = rng.integers(0, len(rows), len(rows))
        boot_drift.append(float(np.median(drift[idx])))
        boot_churn.append(float(np.median(churn[idx])))
    return dict(
        value_rows=int(len(Xv)),
        value_papers=int(len(set(papers))),
        value_events=int(len(rows)),
        median_drift=float(np.median(drift)),
        drift_ci_lo=float(np.percentile(boot_drift, 2.5)),
        drift_ci_hi=float(np.percentile(boot_drift, 97.5)),
        median_churn=float(np.median(churn)),
        churn_ci_lo=float(np.percentile(boot_churn, 2.5)),
        churn_ci_hi=float(np.percentile(boot_churn, 97.5)),
    )


def main() -> None:
    X, _rho, paper, params, *_ = load()
    cols = [params.index(param) for param in F6]
    presence = X[:, cols]
    complete = presence.all(axis=1)
    n_configs = int(len(np.unique(presence[complete], axis=0)))
    row = {
        "family": "F6_aeration",
        "cover_rows": int((presence.sum(axis=1) >= 1).sum()),
        "presence_complete_rows": int(complete.sum()),
        "presence_complete_papers": int(len(set(paper[complete]))),
        "presence_complete_configs": n_configs,
        "presence_complete_ratio": "" if n_configs == 1 else "computed",
        "partial_rows": int((presence.sum(axis=1) >= 1).sum() - complete.sum()),
        "do_only_rows": int(((presence[:, 0] == 1) & (presence[:, 1] == 0)
                             & (presence[:, 2] == 0)).sum()),
        **value_sensitivity(),
    }
    path = os.path.join(RESULTS, "f6_completeness_sensitivity.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"[f6-completeness] {row}")
    print(f"[f6-completeness] wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
