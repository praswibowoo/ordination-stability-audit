#!/usr/bin/env python3
"""clean_primary_audit.py — deletion audit on the curated primary set.

The manuscript's 730/122 primary set contains one known re-publication pair
(B111/B175) and four single-row I4 exceptions (B095, B111, B182, B239). This
script reruns the coverage and value deletion audits after removing those five
papers, using the same estimators as the manuscript:

  * binary PCA on the 44 coverage columns (unscaled, centred);
  * standardized PCA on the four F1 physical parameters;
  * leave-one-study-out (LOSO) block deletion;
  * 100 volume-matched scattered draws per study, donors excluding the deleted
    study;
  * Procrustes drift on shared rows;
  * log-linear block model with a study-level cluster bootstrap;
  * paired block/matched ratio, ratio of medians, block percentile, and
    quartile-grade stability.

The script writes only to results/wp3_clean/. It does not overwrite the
manuscript's committed primary results.

Run with the pinned environment:
    python3 code/clean_primary_audit.py
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import zlib
from collections import defaultdict

import numpy as np
from scipy import stats
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from revision_layers import load_layers  # noqa: E402
from ees_revision_checks import coverage_rows, classify_coverage, classify_value  # noqa: E402

SEED = 42
N_DRAWS = 100
N_BOOT = 2000
EXCLUDE = ("B175", "B111", "B095", "B182", "B239")
OUT = os.path.join(ROOT, "results", "wp3_clean")


def fit_presence(X, dims=2):
    return PCA(n_components=dims, random_state=SEED).fit_transform(X)


def fit_presence_3d(X):
    return PCA(n_components=3, random_state=SEED).fit_transform(X)


def fit_value(X, dims=2):
    Z = StandardScaler().fit_transform(X)
    return PCA(n_components=dims).fit_transform(Z)


def fit_value_3d(X):
    Z = StandardScaler().fit_transform(X)
    return PCA(n_components=3).fit_transform(Z)


def fit_value_fixed(mu, sd, dims=2):
    """Fixed full-data centring and z-scoring, with loadings re-fit per event."""
    def _fit(X):
        Z = (X - mu) / sd
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        return U[:, :dims] * S[:dims]
    return _fit


def drift(S_ref, S_red):
    a, b, _ = procrustes(S_ref, S_red)
    return np.linalg.norm(b - a, axis=1)


def rank_grade(S_full, S_red, idx):
    """Quartile-grade and rank metrics used in the manuscript."""
    a = S_full[idx, 0].copy()
    b = S_red[:, 0].copy()
    c = np.corrcoef(a, b)[0, 1]
    if np.isfinite(c) and c < 0:
        b = -b
    b = (b - b.mean()) / b.std() * a.std() + a.mean()
    cuts = np.quantile(a, [0.25, 0.50, 0.75])
    qa = np.searchsorted(cuts, a, side="right")
    qb = np.searchsorted(cuts, b, side="right")
    n = len(a)
    k10 = max(5, min(int(round(0.10 * n)), n - 1))
    k25 = max(5, min(int(round(0.25 * n)), n - 1))
    top10 = len(set(np.argsort(a)[-k10:]) & set(np.argsort(b)[-k10:])) / k10
    top25 = len(set(np.argsort(a)[-k25:]) & set(np.argsort(b)[-k25:])) / k25
    return dict(
        quartile_flip_pct=100.0 * float(np.mean(qa != qb)),
        top10=float(top10),
        top25=float(top25),
        pc1_spearman=float(stats.spearmanr(a, b).statistic),
    )


def event_stats(S_full, X, keep, fit):
    S_red = fit(X[keep])
    d = drift(S_full[keep], S_red)
    m = rank_grade(S_full, S_red, np.where(keep)[0])
    return dict(med_drift=float(np.median(d)), **m)


def cluster_boot_model(y, lv, block, paper, n_boot=N_BOOT, design_interaction=True):
    """Fit log drift ~ block * centred log volume, cluster bootstrap by paper."""
    if design_interaction:
        Xd = np.column_stack([np.ones(len(y)), lv, block, block * lv])
        names = ["intercept", "log_volume_centred", "block_at_mean_volume",
                 "block_x_log_volume"]
    else:
        Xd = np.column_stack([np.ones(len(y)), lv, block])
        names = ["intercept", "log_volume_centred", "block"]
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    uniq = np.asarray(sorted(set(paper)), dtype=object)
    gid = np.searchsorted(uniq, np.asarray(paper, dtype=object))
    idx = {j: np.where(gid == j)[0] for j in range(len(uniq))}
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx[j] for j in pick])
        boots.append(np.linalg.lstsq(Xd[sel], y[sel], rcond=None)[0])
    boots = np.vstack(boots)
    if boots.ndim != 2:
        raise RuntimeError(f"bootstrap coefficient array has shape {boots.shape}")
    out = []
    for j, name in enumerate(names):
        lo, hi = np.quantile(boots[:, j], [0.025, 0.975])
        out.append(dict(term=name, coef=float(beta[j]), ci_lo=float(lo),
                        ci_hi=float(hi), mult_factor=float(np.exp(beta[j])),
                        mult_lo=float(np.exp(lo)), mult_hi=float(np.exp(hi))))
    return out, beta, boots, float(np.mean(lv))


def paired_summary(block_by_paper, matched_by_paper):
    papers = sorted(block_by_paper)
    d_l = np.asarray([block_by_paper[p] for p in papers], float)
    d_m = np.asarray([np.median(matched_by_paper[p]) for p in papers], float)
    ratio = d_l / d_m
    rng = np.random.default_rng(SEED)
    boot_ratio = np.empty(10000)
    boot_rom = np.empty(10000)
    for i in range(10000):
        ix = rng.integers(0, len(ratio), len(ratio))
        boot_ratio[i] = np.median(ratio[ix])
        boot_rom[i] = np.median(d_l[ix]) / np.median(d_m[ix])
    lo, hi = np.quantile(boot_ratio, [0.025, 0.975])
    lo_r, hi_r = np.quantile(boot_rom, [0.025, 0.975])
    p = stats.wilcoxon(d_l, d_m).pvalue
    return dict(
        n_papers=len(papers),
        block_median=float(np.median(d_l)),
        matched_median=float(np.median(d_m)),
        ratio_of_medians=float(np.median(d_l) / np.median(d_m)),
        ratio_of_medians_ci_lo=float(lo_r),
        ratio_of_medians_ci_hi=float(hi_r),
        paired_ratio=float(np.median(ratio)),
        paired_ratio_ci_lo=float(lo),
        paired_ratio_ci_hi=float(hi),
        wilcoxon_p=float(p),
        n_block_gt_matched=int(np.sum(d_l > d_m)),
    )


def percentile_summary(block_by_paper, matched_by_paper):
    pct = []
    above95 = 0
    below5 = 0
    for p in sorted(block_by_paper):
        m = np.asarray(matched_by_paper[p], float)
        b = float(block_by_paper[p])
        # Mid-rank percentile of the block drift within its matched draws.
        pct.append(100.0 * (np.sum(m < b) + 0.5 * np.sum(m == b)) / len(m))
        if b > np.percentile(m, 95):
            above95 += 1
        if b < np.percentile(m, 5):
            below5 += 1
    pct = np.asarray(pct)
    return dict(
        median_percentile=float(np.median(pct)),
        pct_q1=float(np.quantile(pct, 0.25)),
        pct_q3=float(np.quantile(pct, 0.75)),
        above95_n=int(above95),
        above95_pct=100.0 * above95 / len(pct),
        below5_n=int(below5),
        below5_pct=100.0 * below5 / len(pct),
    )


def grade_summary(block_events, matched_events):
    """Paired study-level quartile-flip difference and bootstrap CI."""
    by_block = {e["paper"]: e["quartile_flip_pct"] for e in block_events}
    by_match = defaultdict(list)
    for e in matched_events:
        by_match[e["paper"]].append(e["quartile_flip_pct"])
    papers = sorted(by_block)
    b = np.asarray([by_block[p] for p in papers], float)
    m = np.asarray([np.median(by_match[p]) for p in papers], float)
    d = b - m
    rng = np.random.default_rng(SEED)
    boot = np.empty(10000)
    for i in range(10000):
        ix = rng.integers(0, len(d), len(d))
        boot[i] = np.median(d[ix])
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return dict(
        n_papers=len(papers),
        block_median=float(np.median(b)),
        matched_median=float(np.median(m)),
        median_diff=float(np.median(d)),
        diff_ci_lo=float(lo),
        diff_ci_hi=float(hi),
        wilcoxon_p=float(stats.wilcoxon(b, m).pvalue),
    )


def volume_bins(block_events, matched_events):
    """Observed per-volume block/matched ratios, with within-bin bootstrap CIs."""
    block_by = {e["paper"]: e for e in block_events}
    matched_by = defaultdict(list)
    for e in matched_events:
        matched_by[e["paper"]].append(e["med_drift"])
    ratios = {p: block_by[p]["med_drift"] / np.median(matched_by[p])
              for p in block_by}
    bins = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6),
            (7, 8), (9, 10), (11, 10**9)]
    out = []
    for lo_v, hi_v in bins:
        papers = [p for p, e in block_by.items()
                  if lo_v <= e["n_rows_deleted"] <= hi_v]
        if not papers:
            continue
        vals = np.asarray([ratios[p] for p in papers], float)
        rng = np.random.default_rng(SEED)
        boot = np.asarray([np.median(rng.choice(vals, len(vals), replace=True))
                           for _ in range(10000)])
        ci = np.quantile(boot, [0.025, 0.975])
        label = (f"{lo_v}" if lo_v == hi_v else
                 f"{lo_v}+" if hi_v == 10**9 else f"{lo_v}-{hi_v}")
        out.append(dict(rows_deleted=label, studies=len(vals),
                        median_ratio=float(np.median(vals)),
                        ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                        median_block_drift=float(np.median(
                            [block_by[p]["med_drift"] for p in papers])),
                        median_matched_drift=float(np.median(
                            [np.median(matched_by[p]) for p in papers]))))
    return out


def top_events(block_events, matched_events, label, k=8):
    matched_by = defaultdict(list)
    for e in matched_events:
        matched_by[e["paper"]].append(e["med_drift"])
    rows = []
    for e in block_events:
        m = np.asarray(matched_by[e["paper"]], float)
        pct = 100.0 * (np.sum(m < e["med_drift"]) + 0.5 * np.sum(m == e["med_drift"])) / len(m)
        rows.append(dict(layer=label, paper=e["paper"],
                         rows=e["n_rows_deleted"],
                         median_drift=e["med_drift"],
                         quartile_flip_pct=e["quartile_flip_pct"],
                         block_percentile=pct))
    rows.sort(key=lambda r: r["median_drift"], reverse=True)
    return rows[:k]


def run_layer(label, X, paper, fit, n_draws=N_DRAWS):
    paper = np.asarray(paper, dtype=object)
    X = np.asarray(X, float)
    S_full = fit(X)
    block_events, matched_events = [], []
    for p in sorted(set(paper)):
        mask = paper == p
        n_p = int(mask.sum())
        keep = ~mask
        st = event_stats(S_full, X, keep, fit)
        st.update(layer=label, paper=p, arm="block", draw=-1,
                  n_rows_deleted=n_p)
        block_events.append(st)

        donors = np.where(~mask)[0]
        for dr in range(n_draws):
            rng = np.random.default_rng(
                SEED + zlib.crc32(str(p).encode()) % 100000 + 7919 * dr)
            drop = rng.choice(donors, size=n_p, replace=False)
            keep2 = np.ones(len(X), bool)
            keep2[drop] = False
            st2 = event_stats(S_full, X, keep2, fit)
            st2.update(layer=label, paper=p, arm="matched", draw=dr,
                       n_rows_deleted=n_p)
            matched_events.append(st2)
    return block_events, matched_events


def build_model(block_events, matched_events, interaction=True):
    events = block_events + matched_events
    y = np.log(np.asarray([e["med_drift"] for e in events], float))
    v = np.log(np.asarray([e["n_rows_deleted"] for e in events], float))
    centre = float(v.mean())
    lv = v - centre
    block = np.asarray([1.0 if e["arm"] == "block" else 0.0 for e in events])
    paper = [e["paper"] for e in events]
    model, beta, boots, _ = cluster_boot_model(
        y, lv, block, paper, design_interaction=interaction)
    return model, beta, boots, centre


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_events(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        for k in ("med_drift", "quartile_flip_pct", "top10", "top25",
                  "pc1_spearman"):
            r[k] = float(r[k])
        r["draw"] = int(r["draw"])
        r["n_rows_deleted"] = int(r["n_rows_deleted"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true",
                    help="reuse existing coverage/value event CSVs")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    L = load_layers()
    X, rho, paper, params = L.X, L.rho, L.paper, L.params
    Xv, keysv, rho_v = L.Xv, L.keysv, L.rho_v
    paper = np.asarray(paper, dtype=object)
    vpaper = np.asarray([k[0] for k in keysv], dtype=object)

    # Guard the manuscript's full primary set before filtering.
    assert X.shape == (730, 44), X.shape
    assert len(set(paper)) == 122, len(set(paper))
    assert Xv.shape == (410, 4), Xv.shape
    assert len(set(vpaper)) == 69, len(set(vpaper))

    keep_c = ~np.isin(paper, EXCLUDE)
    keep_v = ~np.isin(vpaper, EXCLUDE)
    Xc, rhoc, paperc = X[keep_c], rho[keep_c], paper[keep_c]
    Xvc = Xv[keep_v]
    vpaperc = vpaper[keep_v]
    print(f"curated coverage: {Xc.shape[0]} rows / {len(set(paperc))} studies")
    print(f"curated value:    {Xvc.shape[0]} rows / {len(set(vpaperc))} studies")

    # Water-column-only sensitivity, using the same row classification as the
    # manuscript's water-only audit.
    row_types = [classify_coverage(r) for r in coverage_rows()]
    assert len(row_types) == len(X), (len(row_types), len(X))
    water = np.asarray([t == "water" for t in row_types])
    keep_w = keep_c & water
    Xw, rhow, paperw = X[keep_w], rho[keep_w], paper[keep_w]
    print(f"curated water-only: {Xw.shape[0]} rows / {len(set(paperw))} studies")

    # Value-layer clean subset: water-confidence rows only (excludes
    # low-confidence, summary and derived rows), then the same five-paper
    # exclusion. The previous 284/54 sensitivity included B175/B111/B095.
    vmap = {}
    with open(os.path.join(ROOT, "docs/extraction/stageB/matrix_values_v2.csv"),
              encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            vmap[(str(r["batch_id"]), str(r["row_id"]))] = (
                str(r.get("scenario", "") or ""),
                str(r.get("notes", "") or ""),
                str(r.get("extraction_confidence", "") or ""),
            )
    vtypes = []
    for batch, row_id in keysv:
        scen, notes, conf = vmap.get((str(batch), str(row_id)), ("", "", ""))
        vtypes.append(classify_value(batch, scen, notes, conf))
    value_clean = np.asarray([t == "water" for t in vtypes])
    keep_vc = keep_v & value_clean
    Xvc_clean = Xv[keep_vc]
    vpaperc_clean = vpaper[keep_vc]
    print(f"curated value-clean: {Xvc_clean.shape[0]} rows / "
          f"{len(set(vpaperc_clean))} studies")

    summary = []

    layers = [("coverage", Xc, paperc, fit_presence),
              ("coverage_3d", Xc, paperc, fit_presence_3d),
              ("coverage_water", Xw, paperw, fit_presence),
              ("value", Xvc, vpaperc, fit_value)]
    mu_v = Xvc.mean(axis=0)
    sd_v = Xvc.std(axis=0, ddof=0)
    layers.append(("value_fixedz", Xvc, vpaperc,
                   fit_value_fixed(mu_v, sd_v, 2)))
    layers.append(("value_3d", Xvc, vpaperc, fit_value_3d))
    layers.append(("value_fixedz_3d", Xvc, vpaperc,
                   fit_value_fixed(mu_v, sd_v, 3)))
    layers.append(("value_clean", Xvc_clean, vpaperc_clean, fit_value))
    layers.append(("value_clean_3d", Xvc_clean, vpaperc_clean, fit_value_3d))
    if len(Xvc_clean) > 3:
        mu_vc = Xvc_clean.mean(axis=0)
        sd_vc = Xvc_clean.std(axis=0, ddof=0)
        layers.append(("value_clean_fixedz", Xvc_clean, vpaperc_clean,
                       fit_value_fixed(mu_vc, sd_vc, 2)))
        layers.append(("value_clean_fixedz_3d", Xvc_clean, vpaperc_clean,
                       fit_value_fixed(mu_vc, sd_vc, 3)))
    for label, Xl, pl, fit in layers:
        print(f"\n[{label}] running {len(set(pl))} block events x "
              f"{N_DRAWS} matched draws ...")
        events_path = os.path.join(OUT, f"{label}_events.csv")
        if args.reuse and os.path.exists(events_path):
            rows = read_events(events_path)
            blk = [e for e in rows if e["arm"] == "block"]
            mat = [e for e in rows if e["arm"] == "matched"]
            print(f"[{label}] reused {len(blk)} block + {len(mat)} matched events")
        else:
            blk, mat = run_layer(label, Xl, pl, fit)
            write_csv(events_path, blk + mat)

        block_by = {e["paper"]: e["med_drift"] for e in blk}
        matched_by = defaultdict(list)
        for e in mat:
            matched_by[e["paper"]].append(e["med_drift"])

        pair = paired_summary(block_by, matched_by)
        pct = percentile_summary(block_by, matched_by)
        grade = grade_summary(blk, mat)
        bins = volume_bins(blk, mat)
        tops = top_events(blk, mat, label)
        write_csv(os.path.join(OUT, f"{label}_volume_bins.csv"), bins)
        write_csv(os.path.join(OUT, f"{label}_top_events.csv"), tops)
        model, beta, boots, centre = build_model(
            blk, mat, interaction=label.startswith("coverage"))
        write_csv(os.path.join(OUT, f"{label}_model.csv"), model)

        row = dict(layer=label, rows=int(Xl.shape[0]), studies=len(set(pl)),
                   draws=N_DRAWS, **pair, **pct,
                   grade_block_median=grade["block_median"],
                   grade_matched_median=grade["matched_median"],
                   grade_median_diff=grade["median_diff"],
                   grade_diff_ci_lo=grade["diff_ci_lo"],
                   grade_diff_ci_hi=grade["diff_ci_hi"],
                   grade_wilcoxon_p=grade["wilcoxon_p"],
                   model_block_factor=next(
                       (m["mult_factor"] for m in model
                        if m["term"] in ("block_at_mean_volume", "block")), None),
                   model_block_lo=next(
                       (m["mult_lo"] for m in model
                        if m["term"] in ("block_at_mean_volume", "block")), None),
                   model_block_hi=next(
                       (m["mult_hi"] for m in model
                        if m["term"] in ("block_at_mean_volume", "block")), None),
                   mean_log_volume=centre)
        if label.startswith("coverage"):
            inter = next((m for m in model if m["term"] == "block_x_log_volume"),
                         None)
            if inter:
                row.update(interaction=inter["coef"],
                           interaction_ci_lo=inter["ci_lo"],
                           interaction_ci_hi=inter["ci_hi"])
            # Volume-specific model predictions at 1, 5 and 14 rows.
            bB = next(m["coef"] for m in model
                      if m["term"] == "block_at_mean_volume")
            bI = next(m["coef"] for m in model
                      if m["term"] == "block_x_log_volume")
            for v in (1, 5, 14):
                row[f"model_factor_v{v}"] = float(
                    np.exp(bB + bI * (math.log(v) - centre)))
        summary.append(row)

    write_csv(os.path.join(OUT, "clean_summary.csv"), summary)
    for r in summary:
        print("\n", r)

    # A plain-text audit block for the manuscript revision.
    with open(os.path.join(OUT, "clean_audit.txt"), "w", encoding="utf-8") as fh:
        for r in summary:
            fh.write(f"{r['layer']}: {r['rows']} rows / {r['studies']} studies; "
                     f"model block {r['model_block_factor']:.3f} "
                     f"({r['model_block_lo']:.3f}-{r['model_block_hi']:.3f}); "
                     f"paired {r['paired_ratio']:.3f} "
                     f"({r['paired_ratio_ci_lo']:.3f}-{r['paired_ratio_ci_hi']:.3f}); "
                     f"ratio-of-medians {r['ratio_of_medians']:.3f}; "
                     f">95th {r['above95_pct']:.1f}%; "
                     f"grade {r['grade_block_median']:.2f} vs "
                     f"{r['grade_matched_median']:.2f} "
                     f"(diff {r['grade_median_diff']:.2f}, "
                     f"{r['grade_diff_ci_lo']:.2f}-{r['grade_diff_ci_hi']:.2f})\n")


if __name__ == "__main__":
    main()
