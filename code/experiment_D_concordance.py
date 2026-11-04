#!/usr/bin/env python3
"""experiment_D_concordance.py — Experiment D: Cross-Paper Comparability (P1-4.4 · Analysis_Design §ExD).

D1 literal partition attempt: same-site cross-paper pairs ("site if identifiable") — evidence-logged;
   corpus reality: ZERO identifiable shared sites in F1 (token probes: generic-only overlaps) → documented.
D2 fallback concordance operationalizations (pre-registered "if identifiable" clause):
   (i)  paper-clustering counter-tests: paper-label silhouette + within- vs between-paper MRPP-style Δ
        (permutation 999) — tests "within-paper edges only-similar ⇒ map fails its purpose";
   (ii) condition-concordance: cross-paper pairs sharing normalized CONDITION tokens
        (season / tide / depth) → Δcond = mean D(shared) − mean D(unshared) (permutation p),
        Concordance Q = 1 − mean D(shared)/mean D(unshared) (adapted from pre-reg z_b/z_c form);
   Both layers: F1 presence (Jaccard, covers rows) + F1 value (standardized Euclidean, complete numeric rows).
D3 outputs: results/exD_*.csv · figures/figD_distances.png

Baseline guard verified first. Run: python3 code/experiment_D_concordance.py
"""
from __future__ import annotations
import csv, os, re, sys
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117)
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from phase4_freeze import verify_baseline  # noqa: E402

DPI = int(os.environ.get("FIG_DPI", "180"))  # finals pass: FIG_DPI=600 (P1-6.1)

def _save(fig, path):
    """PNG at DPI + vector PDF twin, tight bbox (D117)."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")



SEED, PERM_N = 42, 999
F1 = ["DO (mg/L)", "pH", "Salinity (ppt/\u2030)", "Temperature (\u00b0C)"]
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")

COND_MAP = {
    "monsoon": ("monsoon",), "premonsoon": ("premonsoon", "pre-monsoon", "predry", "pre dry"),
    "postmonsoon": ("postmonsoon", "post-monsoon", "postmonsoon", "post monsoon"),
    "dry": ("dry", "dryseason", "dry season", "summer"), "wet": ("wet", "rainy", "rain", "winter"),
    "hightide": ("high tide", "high-tide", "hightide", "ht"), "lowtide": ("low tide", "low-tide", "lowtide", "lt"),
    "surface": ("surface", "0 m", "0m", "-0.5", "0.5 m"), "depth": ("bottom", "subsurface", "deep", "10 m", "15 m", "20 m", "30 m"),
}


def condition_tokens(scenario: str):
    s = scenario.lower()
    out = set()
    for canon, pats in COND_MAP.items():
        if any(p in s for p in pats):
            out.add(canon)
    # month tokens → season-light split
    months = re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", s)
    if months:
        out.add("month:" + months[0])
    return out


def load_f1():
    pres = list(csv.DictReader(open(os.path.join(
        ROOT, "docs/extraction/stageA/matrix_presence_v2.csv"), encoding="utf-8-sig")))
    params = list(pres[0].keys())[7:51]
    f1_idx = [params.index(p) for p in F1]
    own, rows = "", []
    for r in pres:
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        if r["row_kind"] != "data":
            continue
        Xf = [1.0 if r[p].strip() in ("Y", "NE") else 0.0 for p in F1]
        if any(Xf):
            rows.append(dict(batch=own, scenario=r["scenario"], X=Xf,
                             cond=condition_tokens(r["scenario"])))
    # value-complete F1 rows
    cells = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(
            ROOT, "docs/extraction/stageB/matrix_values_v2.csv"), encoding="utf-8-sig")):
        if r["is_real_value"].strip() == "TRUE" and (r["value"] or "").strip():
            try:
                cells[(r["batch_id"], r["row_id"])][r["param"]] = float(r["value"])
            except ValueError:
                pass
    vrows = []
    scen_by_key = {}
    own, ordn = "", 0
    for r in pres:
        ordn += 1
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        if r["row_kind"] != "data":
            continue
        scen_by_key[(own, str(ordn))] = r["scenario"]
    for k, d in cells.items():
        if all(p in d for p in F1):
            sc = scen_by_key.get(k, "")
            vrows.append(dict(batch=k[0], scenario=sc, X=[d[p] for p in F1],
                              cond=condition_tokens(sc)))
    return rows, vrows


def pair_classes(rows):
    """Pair-type table: within-paper (a-a'), cross-paper shared-conditions (b*), cross-paper other (c)."""
    n = len(rows)
    ii, jj = np.triu_indices(n, 1)
    cls, dist_flag = [], []
    for i, j in zip(ii, jj):
        same_paper = rows[i]["batch"] == rows[j]["batch"]
        shared = bool(rows[i]["cond"] & rows[j]["cond"])
        if same_paper:
            cls.append("a_within")
        elif shared:
            cls.append("b_shared_cond")
        else:
            cls.append("c_cross_other")
    return ii, jj, np.asarray(cls)


def jac_dist(X):
    A = (np.asarray(X) > 0).astype(float)
    inter = A @ A.T
    rs = A.sum(1)
    union = rs[:, None] + rs[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, 1.0)
    return 1.0 - J


def run_layer(rows, tag):
    X = np.asarray([r["X"] for r in rows], float)
    if tag == "value":
        X = StandardScaler().fit_transform(X)
        D = squareform(pdist(X))
    else:
        D = jac_dist(X)
    ii, jj, cls = pair_classes(rows)
    d = D[ii, jj]
    out = {}
    for c in ("a_within", "b_shared_cond", "c_cross_other"):
        sel = cls == c
        out[c] = dict(n=int(sel.sum()), mean=float(d[sel].mean()) if sel.sum() else np.nan,
                      median=float(np.median(d[sel])) if sel.sum() else np.nan)
    # (i) paper-clustering counter-test
    paper = np.asarray([r["batch"] for r in rows])
    papers = sorted(set(paper))
    if len(papers) > 1 and len(set(paper)) < len(paper):
        try:
            sil_paper = float(silhouette_score(D, paper, metric="precomputed"))
        except ValueError:
            sil_paper = np.nan
    else:
        sil_paper = np.nan
    # within vs between Δ (permutation over rows)
    rng = np.random.default_rng(SEED)
    within = d[cls == "a_within"]
    between = d[cls != "a_within"]
    delta_obs = float(within.mean() - between.mean()) if len(within) and len(between) else np.nan
    cnt = 0
    for _ in range(PERM_N):
        perm = rng.permutation(len(rows))
        Xp = X[perm]
        Dp = jac_dist(Xp) if tag != "value" else squareform(pdist(Xp))
        dp = Dp[ii, jj]
        dw = dp[cls == "a_within"].mean()
        db = dp[cls != "a_within"].mean()
        cnt += (dw - db) <= delta_obs
    p_paper = (cnt + 1) / (PERM_N + 1)
    # (ii) condition concordance
    d_shared = d[cls == "b_shared_cond"]
    d_other = d[cls == "c_cross_other"]
    dc_obs = float(d_shared.mean() - d_other.mean()) if len(d_shared) and len(d_other) else np.nan
    cnt2 = 0
    for _ in range(PERM_N):
        perm = rng.permutation(len(rows))
        Xp = X[perm]
        Dp = jac_dist(Xp) if tag != "value" else squareform(pdist(Xp))
        dp = Dp[ii, jj]
        cnt2 += (dp[cls == "b_shared_cond"].mean() - dp[cls == "c_cross_other"].mean()) <= dc_obs
    p_cond = (cnt2 + 1) / (PERM_N + 1)
    Q = 1.0 - (d_shared.mean() / d_other.mean()) if len(d_shared) and len(d_other) and d_other.mean() > 0 else np.nan
    return dict(layer=tag, n_rows=len(rows), n_pairs=len(ii),
                **{f"{c}_n": out[c]["n"] for c in out},
                **{f"{c}_mean": round(out[c]["mean"], 4) for c in out},
                paper_silhouette=round(sil_paper, 4),
                within_minus_between=round(delta_obs, 4), p_within_lt_between=round(float(p_paper), 4),
                shared_minus_other=round(dc_obs, 4), p_shared_lt_other=round(float(p_cond), 4),
                concordance_Q=round(float(Q), 4) if not np.isnan(Q) else np.nan)


def figD(res_pres, res_val, path):
    """Fig D — F1 pair-distance structure, BOTH layers (presence + value)."""
    labels = ["Within-paper", "Cross-paper\n(shared cond.)", "Cross-paper\n(other)"]
    x = np.arange(3)
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.7))
    for ax, res, lab, c, ptag in ((axes[0], res_pres, "Presence layer (Jaccard)", "#4c72b0", "(a)"),
                                  (axes[1], res_val, "Value layer (std. Euclidean)", "#00897b", "(b)")):
        means = [res["a_within_mean"], res["b_shared_cond_mean"], res["c_cross_other_mean"]]
        ax.bar(x, means, 0.58, color=c)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.0)
        ax.set_ylabel("mean pair distance", fontsize=8.2)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.set_title(f"{ptag} {lab}\n$\\Delta$ = {res['shared_minus_other']:+.3f},  "
                     f"p = {res['p_shared_lt_other']:.3f},  Q = {res['concordance_Q']:+.3f}",
                     fontsize=7.5, pad=5)
        head = 0.055 * max(means)
        for xi, mi in zip(x, means):
            ax.annotate(f"{mi:.3f}", (xi, mi + head), fontsize=7.5, ha="center", va="bottom")
        ax.set_ylim(0, max(means) * 1.22)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def outputs(test_rows, probe):
    hdr = ["layer", "n_rows", "n_pairs", "a_within_n",
           "a_within_mean", "b_shared_cond_n", "b_shared_cond_mean",
           "c_cross_other_n", "c_cross_other_mean", "paper_silhouette",
           "within_minus_between", "p_within_lt_between",
           "shared_minus_other", "p_shared_lt_other", "concordance_Q"]
    for name, rows, hdr in (
            ("exD_tests.csv", test_rows, hdr),
            ("exD_bpairs_site_probe.csv", probe, ["note", "n_shared_site_tokens"])):
        path = os.path.join(RESULTS, name)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(rows)
        os.replace(tmp, path)
    print("outputs: exD_tests.csv, exD_bpairs_site_probe.csv")


if __name__ == "__main__":
    verify_baseline()
    rows, vrows = load_f1()
    print(f"F1 covers rows: {len(rows)} · value-complete rows: {len(vrows)}")
    res_pres = run_layer(rows, "presence")
    res_val = run_layer(vrows, "value") if len(vrows) > 20 else None
    probe = [dict(note="same-site cross-paper pairs (literal D1 b-type): token probes found ZERO "
                       "identifiable shared sites in F1 scenarios (capitalized-token scan: 14 "
                       "shared tokens, all generic) — literal partition infeasible; "
                       "fallback = shared-condition concordance (b*)",
                  n_shared_site_tokens=0)]
    for k, v in res_pres.items():
        print("presence:", k, v)
    if res_val:
        for k, v in res_val.items():
            print("value:", k, v)
    outputs([res_pres] + ([res_val] if res_val else []), probe)
    figD(res_pres, res_val, os.path.join(FIGS, "figD_distances.png"))
    print("figD saved")
