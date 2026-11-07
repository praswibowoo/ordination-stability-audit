#!/usr/bin/env python3
"""decision_consequences.py — translate LOPO drift into decision-level changes.

The drift audit reports normalized Procrustes distances. This script adds the
question a reader actually asks: does deleting one paper change a ranking or a
cluster assignment?

For presence PCA (730 primary rows, 122 papers) and standardized F1 value PCA
(410 primary rows, 69 papers), each leave-one-paper-out fit is compared with the
full fit; the legacy 764/421 set is available via experiment_C1.load_legacy():

  * top-10% and top-25% PC1 rank overlap (sign-aligned PC1);
  * top-10 priority-set overlap;
  * k-means (k = 4) adjusted Rand index (ARI);
  * PC1 Spearman correlation.

Outputs
  results/decision_consequences.csv

Run: python3 code/core/decision_consequences.py
"""
from __future__ import annotations

import csv
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from scipy.stats import spearmanr

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS = os.path.join(ROOT, "results")
SEED = 42


def _pca(X):
    return PCA(n_components=2, random_state=SEED).fit_transform(X)


def _top_overlap(full_scores, red_scores, keep, frac):
    n = int(round(frac * len(keep)))
    n = max(5, min(n, len(keep) - 1))
    full_top = set(keep[np.argsort(full_scores[keep])[-n:]])
    red_top = set(keep[np.argsort(red_scores)[-n:]])
    return len(full_top & red_top) / n


def _priority_overlap(full_scores, red_scores, keep, k=10):
    k = min(k, len(keep) - 1)
    full_top = set(keep[np.argsort(full_scores[keep])[-k:]])
    red_top = set(keep[np.argsort(red_scores)[-k:]])
    return len(full_top & red_top) / k


def analyse(X, paper, variant):
    S_full = _pca(X)
    km_full = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit_predict(S_full)
    rows = []
    for p in sorted(set(paper.tolist())):
        keep = np.where(paper != p)[0]
        if len(keep) < 10:
            continue
        S_red = _pca(X[keep])
        # align PC1 sign to the full fit on shared rows
        rho = np.corrcoef(S_full[keep, 0], S_red[:, 0])[0, 1]
        if np.isfinite(rho) and rho < 0:
            S_red[:, 0] *= -1.0
        km_red = KMeans(n_clusters=4, random_state=SEED, n_init=10).fit_predict(S_red)
        rows.append(dict(
            variant=variant, paper=str(p), n_shared=len(keep),
            top10=_top_overlap(S_full[:, 0], S_red[:, 0], keep, 0.10),
            top25=_top_overlap(S_full[:, 0], S_red[:, 0], keep, 0.25),
            priority10=_priority_overlap(S_full[:, 0], S_red[:, 0], keep, 10),
            ari=adjusted_rand_score(km_full[keep], km_red),
            pc1_spearman=float(spearmanr(S_full[keep, 0], S_red[:, 0]).statistic),
        ))
    return rows


def summarise(rows, variant):
    out = []
    for metric in ("top10", "top25", "priority10", "ari", "pc1_spearman"):
        vals = np.asarray([r[metric] for r in rows], float)
        out.append(dict(variant=variant, metric=metric, n_events=len(vals),
                        median=f"{np.median(vals):.4f}",
                        q1=f"{np.quantile(vals, .25):.4f}",
                        q3=f"{np.quantile(vals, .75):.4f}",
                        min=f"{np.min(vals):.4f}",
                        max=f"{np.max(vals):.4f}"))
    # descriptive decision-change rates at declared thresholds
    for metric, threshold in (("top10", 0.90), ("top25", 0.90),
                              ("priority10", 0.90), ("ari", 0.80)):
        vals = np.asarray([r[metric] for r in rows], float)
        out.append(dict(variant=variant, metric=f"{metric}_below_{threshold:.2f}",
                        n_events=len(vals),
                        median=f"{100 * np.mean(vals < threshold):.1f}",
                        q1="", q3="", min="", max=""))
    return out


def main():
    import sys
    sys.path.insert(0, os.path.join(ROOT, "code"))
    from experiment_C1 import load  # noqa: E402

    X, rho, paper, params, Xv, keys, rho_v = load()
    paper_v = np.asarray([k[0] for k in keys])
    rows = []
    rows.extend(analyse(X, paper, "presence PCA"))
    rows.extend(analyse(Xv, paper_v, "std-value PCA"))

    header = ["variant", "paper", "n_shared", "top10", "top25", "priority10",
              "ari", "pc1_spearman"]
    tmp = os.path.join(RESULTS, "decision_consequences.csv") + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})
    os.replace(tmp, os.path.join(RESULTS, "decision_consequences.csv"))

    summary = []
    for variant in ("presence PCA", "std-value PCA"):
        summary.extend(summarise([r for r in rows if r["variant"] == variant], variant))
    tmp = os.path.join(RESULTS, "decision_consequences_summary.csv") + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["variant", "metric", "n_events",
                                           "median", "q1", "q3", "min", "max"])
        w.writeheader()
        w.writerows(summary)
    os.replace(tmp, os.path.join(RESULTS, "decision_consequences_summary.csv"))

    print(f"[decision] {len(rows)} events")
    for r in summary:
        if r["metric"] in ("top10", "top25", "priority10", "ari", "pc1_spearman",
                           "top10_below_0.90", "ari_below_0.80"):
            print(f"  {r['variant']:<14} {r['metric']:<18} {r['median']}")


if __name__ == "__main__":
    main()
