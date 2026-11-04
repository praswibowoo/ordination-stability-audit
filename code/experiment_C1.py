#!/usr/bin/env python3
"""experiment_C1.py — Experiment A: All-Parameter Instability (P1-4.1, Analysis_Design ExA).

A1  full-data ordinations: pca_pres | pca_vraw | pca_vstd | mds_pres | tsne_pres
A2  LOPO (leave-one-paper-out) + LORO (leave-one-row-out, PCA variants):
    refit -> scipy Procrustes alignment (2-D, common rows) -> per-row drift delta
    and kNN(k=5) churn chi
MDS uses classical-MDS initialization (sklearn >= 1.8), so the production fit is
deterministic and seed-independent; the random-start diagnostic is retained in
code/mds_multistart_check.py.
A3  per-variant swing stats: SI = med(delta | rho<=Q1) / med(delta | rho>=Q3),
    Spearman(delta,-rho), OLS slope, permutation null (shuffle rho within paper,
    n=1000, seed 42) -> one-sided p for SI > 1
A4/A5  outputs: results/exC1_rows.csv | results/exC1_summary.csv | figures/fig2_*.png

Baseline guard: verifies the P1-4.5 freeze (results/phase4_baseline.json) first.
Run: python3 code/experiment_C1.py
"""
from __future__ import annotations
import csv, inspect, os, sys, time
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117)
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr, linregress
from scipy.spatial import procrustes as scipy_procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import StandardScaler
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



SEED, KNN_K, PERM_N = 42, 5, 1000
RESULTS = os.path.join(ROOT, "results")
FIGS = os.path.join(ROOT, "figures")
F1_PARAMS = ["DO (mg/L)", "pH", "Salinity (ppt/\u2030)", "Temperature (\u00b0C)"]


def classical_mds_init() -> dict:
    """Return the classical-MDS init kwarg, or fail loudly on old sklearn.

    ``init="classical_mds"`` was added in scikit-learn 1.8. The project pins
    1.9.0; this guard prevents a silent fallback to a random start if an older
    environment is used by mistake.
    """
    if "init" not in inspect.signature(MDS).parameters:
        raise RuntimeError(
            "classical_mds initialization requires scikit-learn >= 1.8; "
            "the project pins 1.9.0"
        )
    return {"init": "classical_mds"}


def _csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def load():
    """Presence matrix (764 data rows × 44 params) + F1-physical value subset."""
    pres = _csv(os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv"))
    hdr = list(pres[0].keys())
    params = hdr[7:51]
    assert len(params) == 44 and hdr[51] == "Other reported parameters"
    X, paper = [], []
    own = ""
    for r in pres:
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        if r["row_kind"] != "data":
            continue
        X.append([1.0 if r[p].strip() in ("Y", "NE") else 0.0 for p in params])
        paper.append(own)
    X = np.asarray(X, float)
    rho = X.mean(axis=1)
    paper = np.asarray(paper)
    ordinal_to_data = {}
    for i, (b, sc) in enumerate(zip(paper, [None] * len(paper))):
        pass
    # rebuild ordinal map: enumerate presence again over raw rows
    ordn = 0
    di = 0
    for r in pres:
        ordn += 1
        if r["row_kind"] == "data":
            ordinal_to_data[ordn] = di
            di += 1

    cells = defaultdict(dict)
    for r in _csv(os.path.join(ROOT, "docs/extraction/stageB/matrix_values_v2.csv")):
        if r["is_real_value"].strip() == "TRUE" and (r["value"] or "").strip():
            try:
                cells[(r["batch_id"], r["row_id"])][r["param"]] = float(r["value"])
            except ValueError:
                pass
    pres_idx = {}
    ordn = 0
    for r in pres:
        ordn += 1
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        pres_idx[ordn] = own
    keys, Xv, rho_v = [], [], []
    for k, d in cells.items():
        if all(p in d for p in F1_PARAMS):
            keys.append(k)
            Xv.append([d[p] for p in F1_PARAMS])
            rho_v.append(rho[ordinal_to_data[int(k[1])]] if int(k[1]) in ordinal_to_data else np.nan)
    Xv = np.asarray(Xv, float)
    rho_v = np.asarray(rho_v, float)
    return X, rho, paper, params, Xv, keys, rho_v


def jac_dist(X):
    A = (X > 0).astype(float)
    inter = A @ A.T
    rs = A.sum(1)
    union = rs[:, None] + rs[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, 1.0)
    return 1.0 - J


def ordinate(variant, X, Xv=None):
    if variant == "pca_pres":
        return PCA(n_components=2, random_state=SEED).fit_transform(X)
    if variant == "pca_vraw":
        return PCA(n_components=2).fit_transform(Xv)
    if variant == "pca_vstd":
        return PCA(n_components=2).fit_transform(StandardScaler().fit_transform(Xv))
    if variant == "mds_pres":
        D = jac_dist(X)
        return MDS(n_components=2, dissimilarity="precomputed", random_state=SEED,
                   **classical_mds_init(), n_init=1,
                   normalized_stress="auto").fit_transform(D)
    if variant == "tsne_pres":
        return TSNE(n_components=2, random_state=SEED, init="pca",
                    perplexity=30).fit_transform(X)
    raise ValueError(variant)


def knn_idx(S, k=KNN_K):
    D = squareform(pdist(S))
    np.fill_diagonal(D, np.inf)
    return np.argsort(D, axis=1)[:, :k]


def knn_churn(inner, outer):
    ch = np.empty(len(inner))
    for i, (a, b) in enumerate(zip(inner, outer)):
        ch[i] = len(set(a) ^ set(b)) / KNN_K
    return ch


def drift_after_procrustes(S_ref, S_red):
    """Aligned per-row displacement between full and reduced ordination planes."""
    a, b, disp = scipy_procrustes(S_ref, S_red)
    return np.linalg.norm(b - a, axis=1), disp



def median_of(v):
    v = np.asarray(v, float)
    return float(np.median(v)) if len(v) else float("nan")


def swing_stats(rho, drift, strata_labels, n_perm=PERM_N):
    """A3 — SI, Spearman(delta,-rho), OLS slope, permutation p (one-sided SDR>1)."""
    rho = np.asarray(rho, float)
    drift = np.asarray(drift, float)
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    m_lo = median_of(drift[rho <= q1])
    m_hi = median_of(drift[rho >= q3])
    si = m_lo / m_hi if m_hi and m_hi > 0 else float("nan")
    sp = spearmanr(rho, drift)
    lr = linregress(rho, drift)
    strata = defaultdict(list)
    for i, lab in enumerate(strata_labels):
        strata[lab].append(i)
    rng = np.random.default_rng(SEED)
    null = np.empty(n_perm)
    for b in range(n_perm):
        rr = rho.copy()
        for idx in strata.values():
            rr[idx] = rng.permutation(rr[idx])
        qa, qb = np.quantile(rr, [0.25, 0.75])
        slo = median_of(drift[rr <= qa])
        shi = median_of(drift[rr >= qb])
        null[b] = slo / shi if shi > 0 else np.nan
    null = null[~np.isnan(null)]
    p_gt = float((np.sum(null > si) + 1) / (len(null) + 1))
    return dict(SI=float(si), med_lo=m_lo, med_hi=m_hi,
                spearman=float(sp.statistic), spearman_p=float(sp.pvalue),
                ols_slope=float(lr.slope), ols_r2=float(lr.rvalue ** 2),
                perm_p=p_gt, q1=float(q1), q3=float(q3))

def run_variant(variant, X, Xv, rho, paper, keys_all, keysv, modes=("lopo", "loro")):
    """A2 — jackknife loop for one variant (event-level records)."""
    value_mode = variant.startswith("pca_v")
    M = Xv if value_mode else X
    pmask = np.asarray([k[0] for k in keysv]) if value_mode else paper
    rid_of = (lambda i: f"{keysv[i][0]}#{keysv[i][1]}") if value_mode else (lambda i: str(i))
    n = len(M)
    S_full = ordinate(variant, M, Xv)
    knn_full = knn_idx(S_full)
    rows = []

    def emit(keep, mode, label):
        keep_idx = np.where(keep)[0]
        S_red = ordinate(variant, M[keep], Xv[keep] if value_mode else None)
        drift, _ = drift_after_procrustes(S_full[keep_idx], S_red)
        ch = knn_churn(knn_full[keep_idx], keep_idx[knn_idx(S_red)])
        sub_rho = rho[keep]   # presence density for pca/mds/tsne; value rows ~ constant rho (documented)
        for pos, i in enumerate(keep_idx):
            rows.append(dict(variant=variant, mode=mode, deleted=label,
                             row_id=rid_of(int(i)), rho=float(sub_rho[pos]),
                             drift=float(drift[pos]), churn=float(ch[pos])))

    if "lopo" in modes:
        for p in sorted(set(pmask)):
            emit(pmask != p, "lopo", p)
    if "loro" in modes:
        for i in range(n):
            keep = np.ones(n, bool)
            keep[i] = False
            emit(keep, "loro", f"row_{i}")
    return rows

def main():
    lock = verify_baseline()
    print("baseline OK (seed", lock["seed"], "| data_version", lock["data_version"] + ")")
    X, rho, paper, params, Xv, keysv, rho_v = load()
    keys_all = [str(i) for i in range(len(X))]
    print(f"presence {X.shape} · {len(set(paper))} papers · F1 value rows {len(Xv)}")
    all_rows, t0 = [], time.time()
    for variant, modes in (("pca_pres", ("lopo", "loro")),
                           ("pca_vraw", ("lopo", "loro")),
                           ("pca_vstd", ("lopo", "loro")),
                           ("mds_pres", ("lopo",)),
                           ("tsne_pres", ("lopo",))):
        t = time.time()
        rows = run_variant(variant, X, Xv, rho_v if variant.startswith("pca_v") else rho, paper, keys_all, keysv, modes=modes)
        all_rows += rows
        n_ev = len({(r["mode"], r["deleted"]) for r in rows})
        print(f"  {variant:9s} {str(modes):18s} events={n_ev} rows={len(rows)} ({time.time()-t:.1f}s)")
    summary = []
    for variant in sorted({r["variant"] for r in all_rows}):
        for mode in ("lopo", "loro"):
            sub = [r for r in all_rows if r["variant"] == variant and r["mode"] == mode]
            if not sub:
                continue
            rr = np.asarray([r["rho"] for r in sub])
            dd = np.asarray([r["drift"] for r in sub])
            cc = np.asarray([r["churn"] for r in sub])
            ll = np.asarray([r["deleted"] for r in sub])
            st = swing_stats(rr, dd, ll)
            summary.append(dict(variant=variant, mode=mode,
                                n_events=len({r["deleted"] for r in sub}), n_obs=len(sub),
                                mean_drift=round(float(dd.mean()), 5),
                                med_drift=round(float(np.median(dd)), 5),
                                mean_churn=round(float(cc.mean()), 5),
                                SI=round(st["SI"], 4), med_lo=round(st["med_lo"], 5),
                                med_hi=round(st["med_hi"], 5),
                                spearman=round(st["spearman"], 4),
                                spearman_p=round(st["spearman_p"], 5),
                                ols_slope=round(st["ols_slope"], 5),
                                ols_r2=round(st["ols_r2"], 4),
                                perm_p=round(st["perm_p"], 4)))
    for path, hdr, data in _outputs(all_rows, summary):
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=hdr)
            w.writeheader()
            w.writerows(data)
        os.replace(tmp, path)
    # The compressed per-row log is read by five downstream scripts (C2, the
    # block-permutation and radius checks, the churn correction and the Fig. 2
    # builder). It is written here so the pipeline produces it rather than
    # inheriting a hand-made file; mtime=0 keeps the bytes reproducible.
    import gzip as _gzip
    _rows_csv = os.path.join(RESULTS, "exC1_rows.csv")
    _gz = _rows_csv + ".gz"
    _tmp_gz = _gz + ".tmp"
    with open(_rows_csv, "rb") as _src, _gzip.GzipFile(_tmp_gz, "wb", mtime=0) as _dst:
        _dst.write(_src.read())
    os.replace(_tmp_gz, _gz)
    fig_a(all_rows, os.path.join(FIGS, "fig2_a.png"))
    fig_b(summary, os.path.join(FIGS, "fig2_b.png"))
    fig_c(all_rows, os.path.join(FIGS, "fig2_c.png"))
    print()
    print(summary_table(summary))
    print(f"\nC1 done in {time.time()-t0:.0f}s → results/exC1_rows.csv, exC1_summary.csv, figures/fig2_*.png")


def summary_table(summary):
    lines = ["variant,mode,SI,perm_p,negSpearman,mean_drift,mean_churn"]
    for s in summary:
        lines.append(f"{s['variant']},{s['mode']},{s['SI']:.3f},{s['perm_p']:.3f},"
                     f"{-s['spearman']:.3f},{s['mean_drift']:.4f},{s['mean_churn']:.3f}")
    return "\n".join(lines)


def _outputs(rows, summary):
    r_hdr = list(rows[0].keys())
    s_hdr = list(summary[0].keys())
    return ((os.path.join(RESULTS, "exC1_rows.csv"), r_hdr, rows),
            (os.path.join(RESULTS, "exC1_summary.csv"), s_hdr, summary))






def _trend(xs, ys, nbins=60):
    """Binned-median trend line — O(n), deterministic (60 equal-width rho bins)."""
    x = np.asarray(xs, float); y = np.asarray(ys, float)
    edges = np.linspace(x.min(), x.max(), nbins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    cx, cy = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() >= 3:
            cx.append(float(np.median(x[m])))
            cy.append(float(np.median(y[m])))
    return np.asarray(cx), np.asarray(cy)


def fig_a(rows, path):
    """Fig 2a — drift vs sparse-row density, all variants (symlog + lowess)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    colors = {"pca_pres": "#1f77b4", "pca_vraw": "#ff7f0e", "pca_vstd": "#2ca02c",
              "mds_pres": "#9467bd", "tsne_pres": "#8c564b"}
    pr = [r["rho"] for r in rows if r["variant"] == "pca_pres"]
    q1, q3 = np.quantile(pr, [.25, .75])
    ax.axvspan(q1, q3, color="#90a4ae", alpha=.10, zorder=0)
    ax.axvline(q1, ls=":", c="k", lw=1); ax.axvline(q3, ls=":", c="k", lw=1)
    ax.annotate(f"Q1={q1:.2f}", (q1, 0.035), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", fontsize=6.5, va="bottom")
    ax.annotate(f"Q3={q3:.2f}", (q3, 0.035), xycoords=("data", "axes fraction"),
                xytext=(3, 0), textcoords="offset points", fontsize=6.5, va="bottom")
    for variant, c in colors.items():
        sub = [r for r in rows if r["variant"] == variant]
        xs = np.asarray([r["rho"] for r in sub])
        ys = np.asarray([r["drift"] for r in sub])
        ax.scatter(xs, ys, s=4, alpha=.20, c=c, label=f"{variant} ({len(sub)} rows)")
        nz = ys > 0
        if nz.sum() >= 30:
            cx, cy = _trend(xs[nz], ys[nz])
            ax.plot(cx, cy, c=c, lw=1.6, alpha=.95)
    ax.set_yscale("symlog", linthresh=1e-5)
    ax.set_xlabel("row density ρ")
    ax.set_ylabel("Procrustes drift δ (LOPO events, symlog)")
    ax.set_title("drift vs sparse-row density — LOPO, all ordination variants\n"
                 "(solid = binned-median trend; shaded = ρ IQR; dotted = Q1/Q3)", fontsize=8.5)
    ax.legend(markerscale=3, fontsize=6.5, loc="upper right")
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def s_p_fmt(p):
    return "n/a" if p != p else (f"{p:.3f}" if p >= 0.001 else "<0.001")


def fig_b(summary, path):
    """Fig 2b — SI per ordination variant (LOPO); vraw = drift≡0 → no bar."""
    vs = ["pca_pres", "pca_vraw", "pca_vstd", "mds_pres", "tsne_pres"]
    recs = [next((s for s in summary if s["variant"] == v and s["mode"] == "lopo"), None)
            for v in vs]
    si = [np.nan if v == "pca_vraw" else (s["SI"] if s else np.nan)
          for v, s in zip(vs, recs)]
    pp = [s["perm_p"] if s else np.nan for s in recs]
    ne = [s.get("n_events", "") if s else "" for s in recs]
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    cols = ["#b0bec5" if v == "pca_vraw"
            else ("#1b4f72" if (p == p and p < 0.05) else "#7fa6c9")
            for v, p in zip(vs, pp)]
    ax.bar(range(len(vs)), si, color=cols)
    for i, (v, s_i, p) in enumerate(zip(vs, si, pp)):
        if v == "pca_vraw":
            ax.annotate("drift ≡ 0\nSI undefined\n(jitter ratio)", (i, 0.02), fontsize=6.2,
                        ha="center", va="bottom", color="#455a64")
        elif s_i == s_i:
            ax.annotate(f"p={s_p_fmt(p)}", (i, s_i), fontsize=7, ha="center", va="bottom",
                        fontweight="bold" if (p == p and p < 0.05) else "normal")
    ax.axhline(1.0, ls="--", c="crimson", lw=1)
    ax.annotate("SDR = 1 (no density effect)", (0.01, 1.0), xycoords=("axes fraction", "data"),
                fontsize=6.2, ha="left", va="bottom", color="crimson")
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels([f"{v}\n(n={int(ne[i])})" if ne[i] != "" else v for i, v in enumerate(vs)],
                       fontsize=7.5)
    ax.set_ylabel("Sparsity drift ratio SDR (LOPO)")
    ax.set_ylim(0, 1.16)
    ax.set_title("sparse-row swing per ordination variant (LOPO)\n"
                 "SDR > 1 = sparse rows drift harder; dark bar = p < 0.05", fontsize=8.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


def fig_c(rows, path):
    """Fig 2c — kNN churn by density quartile (pca_pres)."""
    pr = [r["rho"] for r in rows if r["variant"] == "pca_pres"]
    qs = np.quantile(pr, [.25, .5, .75])
    byq = defaultdict(list)
    for r in rows:
        if r["variant"] != "pca_pres":
            continue
        byq[int(np.digitize(r["rho"], qs))].append(r["churn"])
    labels = ["Q1\n(sparsest)", "Q2", "Q3", "Q4\n(densest)"]
    fig, ax = plt.subplots(figsize=(6.0, 3.9))
    groups = [byq[i] for i in range(4) if byq[i]]
    ax.boxplot(groups,
               tick_labels=[labels[i] for i in range(4) if byq.get(i)])
    ax.set_ylim(-0.06, 2.55)  # headroom: annotations clear of title
    for i, g in enumerate(groups):  # informative: median + n per quartile
        ax.annotate(f"med {np.median(g):.2f}\nn={len(g)}", (i + 1, max(g)),
                    xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=6.5, color="#37474f")
    ax.set_ylabel("kNN churn χ (k=5)")
    ax.set_title("neighbour churn by density quartile (pca_pres, LOPO)\n"
                 "χ = fraction of an event's k=5 neighbourhood that changes", fontsize=8.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)


if __name__ == "__main__":
    main()
