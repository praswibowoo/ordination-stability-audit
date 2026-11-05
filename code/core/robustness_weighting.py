#!/usr/bin/env python3
"""robustness_weighting.py — three small robustness checks (B/C/D).

B  Row-weighted vs paper-weighted SDR for the confirmatory variants.
C  Paired block/scattered ratio using the first draw vs all 100 draws.
D  Absolute drift percentile of block deletion in the volume-matched null.

Reads existing result layers only; runs no ordination.
Writes results/robustness_weighting.csv (long format).
"""
from __future__ import annotations

import csv
import gzip
import os
from collections import defaultdict

import numpy as np

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
RESULTS = os.path.join(ROOT, "results")
PRES = os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv")


def read_csv(path):
    with open(path, encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def row_to_paper():
    """Map presence ordinal -> paper label for data rows."""
    out = {}
    own = ""
    ordinal = 0
    for r in read_csv(PRES):
        ordinal += 1
        b = (r.get("batch_id") or "").strip()
        if b:
            own = b
        if r.get("row_kind") == "data":
            out[str(ordinal)] = own
    return out


def sdr(rho, drift):
    rho = np.asarray(rho, float)
    drift = np.asarray(drift, float)
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    lo = np.median(drift[rho <= q1])
    hi = np.median(drift[rho >= q3])
    return float(lo / hi) if hi > 0 else float("nan")


def main():
    r2p = row_to_paper()
    def paper_of(variant, rid):
        if variant.startswith("pca_v"):
            return str(rid).split("#")[0]
        return r2p.get(rid, "?")
    rows = []
    with gzip.open(os.path.join(RESULTS, "exC1_rows.csv.gz"), "rt") as fh:
        rd = csv.DictReader(fh)
        by_event = defaultdict(list)
        for r in rd:
            if r["mode"] != "lopo":
                continue
            if r["variant"] not in ("mds_pres", "pca_vstd"):
                continue
            by_event[(r["variant"], r["deleted"])].append(
                (r["row_id"], float(r["rho"]), float(r["drift"])))
    for variant in ("mds_pres", "pca_vstd"):
        events = [(k[1], v) for k, v in by_event.items() if k[0] == variant]
        row_rho = np.concatenate([np.array([x[1] for x in v]) for _, v in events])
        row_drift = np.concatenate([np.array([x[2] for x in v]) for _, v in events])
        rows.append(dict(check="B_row_weighted", variant=variant,
                         value=round(sdr(row_rho, row_drift), 4)))
        per_event = []
        pooled_rho, pooled_drift = [], []
        for _, ev in events:
            by_paper = defaultdict(list)
            for rid, rho, drift in ev:
                by_paper[paper_of(variant, rid)].append((rho, drift))
            pr = [np.mean([x[0] for x in v]) for v in by_paper.values()]
            pd = [np.mean([x[1] for x in v]) for v in by_paper.values()]
            per_event.append(sdr(pr, pd))
            pooled_rho.extend(pr)
            pooled_drift.extend(pd)
        rows.append(dict(check="B_paper_event_median", variant=variant,
                         value=round(float(np.nanmedian(per_event)), 4)))
        # paper-weighted permutation p for the pooled statistic
        paper_labels = []
        for _, ev in events:
            by_paper = defaultdict(list)
            for rid, rho, drift in ev:
                by_paper[paper_of(variant, rid)].append((rho, drift))
            paper_labels.extend(list(by_paper.keys()))
        paper_labels = np.asarray(paper_labels)
        uniq = np.asarray(sorted(set(paper_labels)))
        code = np.searchsorted(uniq, paper_labels)
        prho = np.asarray([np.mean(np.asarray(pooled_rho)[code == j])
                           for j in range(len(uniq))])
        rng = np.random.default_rng(42)
        obs = sdr(pooled_rho, pooled_drift)
        null = np.empty(1000)
        for b in range(1000):
            null[b] = sdr(rng.permutation(prho)[code], pooled_drift)
        # proper permutation: observed and null both use paper mean densities
        obs_proper = sdr(prho[code], pooled_drift)
        null_proper = np.empty(1000)
        for b in range(1000):
            null_proper[b] = sdr(rng.permutation(prho)[code], pooled_drift)
        p_proper = float((np.sum(null_proper >= obs_proper) + 1) / 1001)
        rows.append(dict(check="B_paper_pooled_obs_proper", variant=variant,
                         value=round(obs_proper, 4)))
        rows.append(dict(check="B_paper_pooled_p_proper", variant=variant,
                         value=round(p_proper, 4)))

    # C — first draw vs all draws
    lopo = {r["paper"]: float(r["med_drift"])
            for r in read_csv(os.path.join(RESULTS, "exH_volumematch.csv"))
            if r["arm"] == "lopo"}
    draws = defaultdict(dict)
    for r in read_csv(os.path.join(RESULTS, "exH_volumematch_draws.csv")):
        draws[r["paper"]][int(r["draw"])] = float(r["med_drift"])
    papers = sorted(set(lopo) & set(draws))
    first = [lopo[p] / draws[p][0] for p in papers if 0 in draws[p]]
    rows.append(dict(check="C_first_draw_median_ratio", variant="presence PCA",
                     value=round(float(np.median(first)), 4)))
    all_draw = []
    for d in range(100):
        vals = [lopo[p] / draws[p][d] for p in papers if d in draws[p]]
        if vals:
            all_draw.append(float(np.median(vals)))
    rows.append(dict(check="C_all_draw_median_ratio", variant="presence PCA",
                     value=round(float(np.median(all_draw)), 4)))
    rows.append(dict(check="C_all_draw_ratio_ci_lo", variant="presence PCA",
                     value=round(float(np.quantile(all_draw, 0.025)), 4)))
    rows.append(dict(check="C_all_draw_ratio_ci_hi", variant="presence PCA",
                     value=round(float(np.quantile(all_draw, 0.975)), 4)))

    # D — absolute drift percentile of block events in their matched draws
    pct = []
    for p in papers:
        b = lopo[p]
        vals = np.array(list(draws[p].values()))
        pct.append(float(np.mean(vals < b) * 100.0))
    rows.append(dict(check="D_block_percentile_median", variant="presence PCA",
                     value=round(float(np.median(pct)), 1)))
    rows.append(dict(check="D_block_percentile_q1", variant="presence PCA",
                     value=round(float(np.quantile(pct, 0.25)), 1)))
    rows.append(dict(check="D_block_percentile_q3", variant="presence PCA",
                     value=round(float(np.quantile(pct, 0.75)), 1)))
    above95 = [lopo[p] > float(np.quantile(list(draws[p].values()), 0.95))
               for p in papers]
    rows.append(dict(check="D_block_above_q95_pct", variant="presence PCA",
                     value=round(100.0 * float(np.mean(above95)), 1)))

    out = os.path.join(RESULTS, "robustness_weighting.csv")
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "variant", "value"])
        w.writeheader()
        w.writerows(rows)
    os.replace(out + ".tmp", out)
    for r in rows:
        print(f"  {r['check']:28s} {r['variant']:14s} {r['value']}")


if __name__ == "__main__":
    main()
