#!/usr/bin/env python3
"""ees_revision_checks.py — pre-submission sensitivity checks for the EES manuscript.

Purpose
-------
The primary manuscript matrices mix water-column rows with sediment, soil,
bioassay, biota, experimental, summary and derived rows.  This script adds the
sensitivities requested in the pre-submission audit:

  1. coverage block model on water-column rows only (2-D, 100 draws);
  2. coverage row-weighted and paper-weighted SDR on water-column rows only;
  3. coverage rank/grade stability on water-column rows only (20 draws);
  4. value block model with fixed full-data z-scoring and 100 draws (2-D and 3-D),
     including a derived-row-excluded variant;
  5. value rank/grade stability with derived rows excluded (20 draws);
  6. the status of the two showcase events, B063 and B156.

Writes machine-readable outputs to results/ees_revision/.  The manuscript is not
modified by this script.

Run: python3 code/ees_revision_checks.py
"""
from __future__ import annotations

import csv
import json
import os
import sys
import zlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import procrustes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results", "ees_revision")
SEED = 42

from revision_layers import load_layers, PRESENCE_PATH  # noqa: E402
from experiment_H_volumematch import ols_cluster_boot  # noqa: E402
from ees_rank_matched import metrics  # noqa: E402


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------
def coverage_rows():
    """Return the 730 primary rows with scenario text, in the exact X order."""
    rows = list(csv.DictReader(open(PRESENCE_PATH, encoding="utf-8-sig", newline="")))
    own = ""
    ordinal = 0
    standard_keys = set()
    kept = []
    for row in rows:
        ordinal += 1
        batch = (row.get("batch_id") or "").strip()
        if batch:
            own = batch
        if row.get("row_kind") != "data":
            continue
        key = (own, str(ordinal))
        if (row.get("scenario_type") or "").strip() == "standard":
            standard_keys.add(key)
        if key in standard_keys:
            continue
        kept.append(dict(
            paper=own,
            scenario=(row.get("scenario") or "").strip(),
            source=(row.get("source_table_ref") or "").strip(),
            note=(row.get("cause_effect_note") or "").strip(),
        ))
    return kept


def classify_coverage(rec):
    s = rec["scenario"].lower()
    if "sediment" in s:
        return "sediment"
    if "soil" in s:
        return "soil"
    if "bioassay" in s or "lc50" in s:
        return "bioassay"
    if "fish" in s and ("treated" in s or "tissue" in s):
        return "biota"
    if "tank" in s or "acclimation" in s:
        return "experimental"
    if (("all " in s and any(k in s for k in ("station", "site", "creek", "transect")))
            or "whole " in s or "pooled" in s or "study-wide" in s
            or "weighted mean" in s or "secondary" in s):
        return "summary"
    if rec["paper"] == "B063":
        # B063 is a crude-oil strandline study: only the two water-surface
        # samples are water-column observations.
        if "water" in s and not any(k in s for k in
                                    ("sediment", "mud", "sand", "sea food", "vegetation")):
            return "water"
        return "strandline"
    return "water"


def classify_value(batch, scenario, notes, confidence):
    s = (scenario or "").lower()
    n = (notes or "").lower()
    if "derived" in n:
        return "derived"
    if (confidence or "").strip().lower() == "low":
        return "low_confidence"
    if (("all " in s and any(k in s for k in ("station", "site", "creek", "transect")))
            or "whole " in s or "pooled" in s or "study-wide" in s
            or "weighted mean" in s or "secondary" in s):
        return "summary"
    return "water"


# ---------------------------------------------------------------------------
# Fitting and drift
# ---------------------------------------------------------------------------
def fit_presence(X, dims=2):
    return PCA(n_components=dims, random_state=SEED).fit_transform(X)


def fit_value_recomputed(X, dims=2):
    Z = StandardScaler().fit_transform(X)
    return PCA(n_components=dims).fit_transform(Z)


def fit_value_fixed(mu, sd, dims=2):
    def _fit(X):
        Z = (X - mu) / sd
        U, S, _ = np.linalg.svd(Z, full_matrices=False)
        return U[:, :dims] * S[:dims]
    return _fit


def procrustes_drift(S_full, S_red, keep):
    a, b, _ = procrustes(S_full[keep], S_red)
    return np.linalg.norm(b - a, axis=1)


def sdr(rho, drift):
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    hi = float(np.median(drift[rho >= q3]))
    lo = float(np.median(drift[rho <= q1]))
    return lo / hi if hi > 0 else float("nan")


def paper_sdr(rho, drift, paper):
    frame = pd.DataFrame({"rho": rho, "drift": drift, "paper": paper})
    agg = frame.groupby("paper").agg(rho=("rho", "mean"), drift=("drift", "mean"))
    return sdr(agg["rho"].to_numpy(float), agg["drift"].to_numpy(float))


# ---------------------------------------------------------------------------
# Block model
# ---------------------------------------------------------------------------
def run_block(label, X, paper, rho, fit, dims=2, draws=100, interaction=True):
    S_full = fit(X)
    n = len(X)
    uniq = sorted(set(paper))
    lopo, matched = [], []
    event_sdr_row, event_sdr_paper = [], []
    for p in uniq:
        mask = paper == p
        v = int(mask.sum())
        keep = ~mask
        drift = procrustes_drift(S_full, fit(X[keep]), keep)
        lopo.append(dict(paper=p, arm="block", V=v, drift=float(np.median(drift))))
        event_sdr_row.append(sdr(rho[keep], drift))
        event_sdr_paper.append(paper_sdr(rho[keep], drift, paper[keep]))

        donors = np.where(keep)[0]
        base = SEED + zlib.crc32(str(p).encode()) % 100000
        for dr in range(draws):
            rng = np.random.default_rng(base + 7919 * dr)
            drop = rng.choice(donors, size=v, replace=False)
            keep2 = np.ones(n, bool)
            keep2[drop] = False
            drift2 = procrustes_drift(S_full, fit(X[keep2]), keep2)
            matched.append(dict(paper=p, arm="scattered", V=v,
                                drift=float(np.median(drift2)), draw=dr))

    ev = lopo + matched
    y = np.log(np.asarray([e["drift"] for e in ev], float))
    raw_lv = np.log(np.asarray([e["V"] for e in ev], float))
    mean_logV = float(raw_lv.mean())
    lv = raw_lv - mean_logV
    blk = np.asarray([1.0 if e["arm"] == "block" else 0.0 for e in ev])
    grp = np.asarray([e["paper"] for e in ev], dtype=object)
    cols = [np.ones(len(ev)), lv, blk]
    names = ["intercept", "log_volume", "block"]
    if interaction:
        cols.append(blk * lv)
        names.append("block_x_log_volume")
    Xd = np.column_stack(cols)
    beta, lo, hi, n_ok, boots = ols_cluster_boot(Xd, y, grp, n_boot=1000, seed=SEED)

    out = {
        "label": label,
        "n_papers": len(uniq),
        "n_rows": n,
        "dims": dims,
        "draws": draws,
        "interaction": bool(interaction),
        "n_events": len(ev),
        "n_boot": n_ok,
        "sdr_row_median": float(np.nanmedian(event_sdr_row)),
        "sdr_paper_median": float(np.nanmedian(event_sdr_paper)),
    }
    for j, name in enumerate(names):
        out[f"{name}_beta"] = float(beta[j])
        out[f"{name}_lo"] = float(lo[j])
        out[f"{name}_hi"] = float(hi[j])
        out[f"{name}_factor"] = float(np.exp(beta[j]))
        out[f"{name}_factor_lo"] = float(np.exp(lo[j]))
        out[f"{name}_factor_hi"] = float(np.exp(hi[j]))
    # factor at 5 rows
    gm = float(np.exp(np.mean(np.log(np.asarray([e["V"] for e in ev], float)))))
    out["geom_mean_V"] = gm
    if interaction:
        b5 = beta[2] + beta[3] * (np.log(5.0) - mean_logV)
        se5 = np.sqrt(boots[:, 2].var() + (np.log(5.0) - mean_logV) ** 2 * boots[:, 3].var()
                      + 2 * (np.log(5.0) - mean_logV) * np.cov(boots[:, 2], boots[:, 3])[0, 1])
        out["block_factor_5rows"] = float(np.exp(b5))
        out["block_factor_5rows_lo"] = float(np.exp(b5 - 1.96 * se5))
        out["block_factor_5rows_hi"] = float(np.exp(b5 + 1.96 * se5))

    # per-paper paired ratio using each paper's median scatter
    by_paper = defaultdict(list)
    block_by_paper = {}
    for e in ev:
        if e["arm"] == "scattered":
            by_paper[e["paper"]].append(e["drift"])
        else:
            block_by_paper[e["paper"]] = e["drift"]
    ratios = np.asarray([block_by_paper[p] / np.median(by_paper[p]) for p in uniq])
    out["paired_ratio_median"] = float(np.median(ratios))
    out["paired_ratio_iqr_lo"] = float(np.percentile(ratios, 25))
    out["paired_ratio_iqr_hi"] = float(np.percentile(ratios, 75))
    out["ratio_of_medians"] = float(np.median(list(block_by_paper.values()))
                                    / np.median([np.median(by_paper[p]) for p in uniq]))
    return out, pd.DataFrame(lopo), pd.DataFrame(matched), ratios


# ---------------------------------------------------------------------------
# Rank/grade
# ---------------------------------------------------------------------------
def run_rank(label, X, paper, fit, draws=20):
    S_full = fit(X)
    rows = []
    for p in sorted(set(paper)):
        mask = paper == p
        v = int(mask.sum())
        keep = ~mask
        idx = np.where(keep)[0]
        mb = metrics(S_full, fit(X[keep]), idx)
        rows.append(dict(label=label, paper=p, arm="block", draw=-1,
                         n_rows=v, **mb))
        donors = np.where(keep)[0]
        base = SEED + zlib.crc32(str(p).encode()) % 100000
        for dr in range(draws):
            rng = np.random.default_rng(base + 7919 * dr)
            drop = rng.choice(donors, size=v, replace=False)
            keep2 = np.ones(len(X), bool)
            keep2[drop] = False
            idx2 = np.where(keep2)[0]
            mm = metrics(S_full, fit(X[keep2]), idx2)
            rows.append(dict(label=label, paper=p, arm="matched", draw=dr,
                             n_rows=v, **mm))
    df = pd.DataFrame(rows)
    summary = []
    for metric in ("quartile_flip_pct", "top10", "top25", "pc1_spearman"):
        block = df[df.arm == "block"].set_index("paper")[metric]
        matched = df[df.arm == "matched"].groupby("paper")[metric].median()
        pair = (block - matched).dropna()
        d = pair.to_numpy(float)
        b = block.loc[pair.index].to_numpy(float)
        m = matched.loc[pair.index].to_numpy(float)
        rng = np.random.default_rng(SEED)
        boot = np.asarray([np.median(rng.choice(d, len(d), replace=True))
                           for _ in range(10000)])
        lo, hi = np.quantile(boot, [0.025, 0.975])
        summary.append(dict(label=label, metric=metric, n=len(d),
                            block_median=float(np.median(b)),
                            matched_median=float(np.median(m)),
                            median_diff=float(np.median(d)),
                            ci_lo=float(lo), ci_hi=float(hi),
                            wilcoxon_p=float(stats.wilcoxon(b, m).pvalue)))
    worst = (df[df.arm == "block"].sort_values("quartile_flip_pct", ascending=False)
             .head(8)[["label", "paper", "n_rows", "quartile_flip_pct",
                       "top10", "pc1_spearman"]])
    return pd.DataFrame(summary), worst


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(RESULTS, exist_ok=True)
    L = load_layers()
    X = L.X
    rho = L.rho
    paper = np.asarray(L.paper)

    kept = coverage_rows()
    assert len(kept) == len(X), (len(kept), len(X))
    types = [classify_coverage(r) for r in kept]
    cov_type = Counter(types)
    print("coverage row types:", dict(cov_type))

    water_mask = np.asarray([t == "water" for t in types])
    print(f"water coverage rows: {water_mask.sum()} / {len(water_mask)} "
          f"({water_mask.mean():.1%}); papers {len(set(paper[water_mask]))}")

    # value row classification
    val = pd.read_csv(os.path.join(ROOT, "docs/extraction/stageB/matrix_values_v2.csv"))
    vmap = {}
    for _, r in val.iterrows():
        vmap[(str(r["batch_id"]), str(r["row_id"]))] = (
            str(r.get("scenario", "") or ""),
            str(r.get("notes", "") or ""),
            str(r.get("extraction_confidence", "") or ""),
        )
    vtypes = []
    for batch, row_id in L.keysv:
        scen, notes, conf = vmap.get((str(batch), str(row_id)), ("", "", ""))
        vtypes.append(classify_value(batch, scen, notes, conf))
    vtype = Counter(vtypes)
    print("value row types:", dict(vtype))
    value_clean = np.asarray([t == "water" for t in vtypes])
    value_no_derived = np.asarray([t != "derived" for t in vtypes])
    print(f"value clean rows: {value_clean.sum()} / {len(value_clean)}; "
          f"papers {len(set(np.asarray([k[0] for k in L.keysv])[value_clean]))}")

    # ---------- coverage block models ----------
    block_rows = []
    r, lopo, matched, ratios = run_block(
        "coverage_all", X, paper, rho, fit_presence, dims=2, draws=100, interaction=True)
    block_rows.append(r)
    r, *_ = run_block(
        "coverage_water", X[water_mask], paper[water_mask], rho[water_mask],
        fit_presence, dims=2, draws=100, interaction=True)
    block_rows.append(r)

    # ---------- value block models ----------
    vpaper = np.asarray([k[0] for k in L.keysv], dtype=object)
    Xv = L.Xv
    rho_v = L.rho_v
    for label, mask in (("value_all", np.ones(len(Xv), bool)),
                        ("value_no_derived", value_no_derived),
                        ("value_clean", value_clean)):
        Xm, pm, rm = Xv[mask], vpaper[mask], rho_v[mask]
        r, *_ = run_block(label + "_recomputed", Xm, pm, rm,
                          fit_value_recomputed, dims=2, draws=100, interaction=False)
        block_rows.append(r)
        mu = Xm.mean(axis=0)
        sd = Xm.std(axis=0, ddof=0)
        r, *_ = run_block(label + "_fixedz", Xm, pm, rm,
                          fit_value_fixed(mu, sd, 2), dims=2, draws=100, interaction=False)
        block_rows.append(r)
        # 3-D fixed z
        r, *_ = run_block(label + "_fixedz_3d", Xm, pm, rm,
                          fit_value_fixed(mu, sd, 3), dims=3, draws=100, interaction=False)
        block_rows.append(r)

    bdf = pd.DataFrame(block_rows)
    bdf.to_csv(os.path.join(RESULTS, "block_models.csv"), index=False, float_format="%.6g")
    print("\nBLOCK MODELS")
    cols = ["label", "n_papers", "n_rows", "block_factor", "block_factor_lo",
            "block_factor_hi", "block_x_log_volume", "block_x_log_volume_lo",
            "block_x_log_volume_hi", "block_factor_5rows", "paired_ratio_median",
            "sdr_row_median", "sdr_paper_median"]
    print(bdf[[c for c in cols if c in bdf.columns]].to_string(index=False))

    # ---------- rank/grade ----------
    rank_all, worst_all = run_rank("coverage_all", X, paper, fit_presence, draws=20)
    rank_water, worst_water = run_rank(
        "coverage_water", X[water_mask], paper[water_mask], fit_presence, draws=20)
    rank_value, worst_value = run_rank(
        "value_all", Xv, vpaper, fit_value_recomputed, draws=20)
    rank_vclean, worst_vclean = run_rank(
        "value_clean", Xv[value_clean], vpaper[value_clean],
        fit_value_recomputed, draws=20)
    rdf = pd.concat([rank_all, rank_water, rank_value, rank_vclean], ignore_index=True)
    rdf.to_csv(os.path.join(RESULTS, "rank_grade.csv"), index=False, float_format="%.6g")
    wdf = pd.concat([worst_all, worst_water, worst_value, worst_vclean], ignore_index=True)
    wdf.to_csv(os.path.join(RESULTS, "worst_events.csv"), index=False, float_format="%.6g")
    print("\nRANK/GRADE")
    print(rdf[rdf.metric == "quartile_flip_pct"].to_string(index=False))
    print("\nWORST EVENTS")
    print(wdf.groupby("label").head(3).to_string(index=False))

    # ---------- status of showcase events ----------
    status = []
    for p in ("B063", "B156", "B154", "B239"):
        idx = [i for i, r in enumerate(kept) if r["paper"] == p]
        status.append(dict(paper=p, layer="coverage", rows=len(idx),
                           types=";".join(sorted(set(types[i] for i in idx)))))
        vidx = [i for i, k in enumerate(L.keysv) if k[0] == p]
        vstat = [vtypes[i] for i in vidx]
        status.append(dict(paper=p, layer="value", rows=len(vidx),
                           types=";".join(sorted(set(vstat)))))
    sdf = pd.DataFrame(status)
    sdf.to_csv(os.path.join(RESULTS, "showcase_status.csv"), index=False)
    print("\nSHOWCASE STATUS")
    print(sdf.to_string(index=False))

    # ---------- JSON summary ----------
    with open(os.path.join(RESULTS, "summary.json"), "w") as fh:
        json.dump(dict(
            coverage_row_types=dict(cov_type),
            value_row_types=dict(vtype),
            block_models=bdf.to_dict(orient="records"),
            rank_grade=rdf.to_dict(orient="records"),
            showcase=sdf.to_dict(orient="records"),
        ), fh, indent=2)


if __name__ == "__main__":
    main()
