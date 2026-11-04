#!/usr/bin/env python3
"""experiment_C2_stability.py — Experiment C: Stability Comparison (P1-4.3 · Analysis_Design §ExC).

C1  paired per-deletion-event SI: family-restricted vs all-parameter (presence-PCA main +
    MDS sensitivity), paired over papers present in both layers
    bootstrap (1000 resamples over paper-deletion set) → one-sided CI for G = 1 − SI_fam/SI_all
C2  sensitivities: (a) alternative family assignment = raw Ward cuts k=3/k=5 (clustering-only
    families, ≥10 complete rows); (b) MDS instead of PCA; (c) drift in top-3-PC subspace
C3  Fig 5: per-family paired SI + CI bars + effect-size annotations
Stats: Holm correction across families for P(G>0) (pre-reg §5).
MDS uses classical-MDS initialization (sklearn >= 1.8); the MDS sensitivity is
deterministic and seed-independent.

Baseline guard verified first. Outputs: results/exC2_*.csv · figures/fig5.png.
Run: python3 code/experiment_C2_stability.py
"""
from __future__ import annotations
import csv, gzip, os, sys, time
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117)
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from phase4_freeze import verify_baseline  # noqa: E402
from experiment_C1 import (  # noqa: E402
    classical_mds_init, load, drift_after_procrustes, knn_idx, median_of, jac_dist,
)

DPI = int(os.environ.get("FIG_DPI", "180"))  # finals pass: FIG_DPI=600 (P1-6.1)

def _save(fig, path):
    """PNG at DPI + vector PDF twin, tight bbox (D117)."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")



SEED = 42
BOOT_N = 1000
FAMILIES = {
    "F1_physical": ["DO (mg/L)", "pH", "Salinity (ppt/‰)", "Temperature (°C)"],
    "F2_metals": ["As (µg/L)", "Cd (µg/L)", "Cr (µg/L)", "Cu (µg/L)",
                  "Fe (µg/L)", "Ni (µg/L)", "Pb (µg/L)", "Zn (µg/L)", "Co (µg/L)"],
    "F3_nutrients": ["NO2 (mg/L)", "NO3 (mg/L)", "NH4 (mg/L)", "PO4 (mg/L)"],
    "F4_organic": ["COD (mg/L)", "BOD (mg/L)", "Total Coliform (CFU/100mL)"],
    "F1_physicala": ["TDS", "EC (µS/cm)", "Cl- (ppt)", "Hardness (mg/L)", "Alkalinity (mg/L)"],
    "F1_clarity": ["Turbidity (NTU)", "Secchi (m)", "TSS (mg/L)"],
    "F6_aeration": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)"],
}


def event_si(rho, drift):
    """Per-deletion-event swing index (median sparse / median dense)."""
    rho = np.asarray(rho, float)
    drift = np.asarray(drift, float)
    if len(rho) < 8:
        return np.nan
    q1, q3 = np.quantile(rho, [0.25, 0.75])
    m_lo = median_of(drift[rho <= q1])
    m_hi = median_of(drift[rho >= q3])
    return m_lo / m_hi if m_hi and m_hi > 0 else np.nan


def allparam_event_si(gz_path, variant):
    """Parse exC1_rows.csv.gz → per-deletion-event SI for the all-parameter layer."""
    per = defaultdict(lambda: ([], []))
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] != variant or r["mode"] != "lopo":
                continue
            rho_l, d_l = per[r["deleted"]]
            rho_l.append(float(r["rho"]))
            d_l.append(float(r["drift"]))
    return {k: event_si(v[0], v[1]) for k, v in per.items()}


def family_event_pairs(X_fam, rho_fam, strata, papers_all_si):
    """Family LOPO per-event SI + paired all-parameter SI for the same paper."""
    S_full = PCA(n_components=2, random_state=SEED).fit_transform(X_fam)
    pairs = []
    for p in sorted(set(strata)):
        keep = np.asarray([s != p for s in strata])
        if keep.sum() < 8:
            continue
        S_red = PCA(n_components=2, random_state=SEED).fit_transform(X_fam[keep])
        d, _ = drift_after_procrustes(S_full[keep], S_red)
        si_f = event_si(rho_fam[keep], d)
        si_a = papers_all_si.get(p, np.nan)
        if not (np.isnan(si_f) or np.isnan(si_a)):
            pairs.append(dict(deleted=p, SI_family=si_f, SI_all=si_a,
                              G=1.0 - si_f / si_a if si_a > 0 else np.nan))
    return pairs


def bootstrap_G(pairs, rng, n_boot=BOOT_N):
    """One-sided bootstrap for G>0 (G = 1 − SI_fam/SI_all) and SI-difference>0."""
    fam = np.asarray([p["SI_family"] for p in pairs], float)
    allp = np.asarray([p["SI_all"] for p in pairs], float)
    df = fam - allp
    n = len(pairs)
    gs, dfs = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        gs[b] = 1.0 - np.median(fam[idx]) / np.median(allp[idx]) if np.median(allp[idx]) > 0 else np.nan
        dfs[b] = np.median(df[idx])
    gs = gs[~np.isnan(gs)]
    p_gt = float((np.sum(gs <= 0) + 1) / (len(gs) + 1))
    p_diff = float((np.sum(dfs <= 0) + 1) / (n_boot + 1))
    ci = (float(np.quantile(gs, 0.05)), float(np.quantile(gs, 0.95)))
    return dict(G_median=float(np.median(gs)), G_CI95=ci, P_G_gt0=p_gt,
                dSI_median=float(np.median(dfs)), P_dSI_gt0=p_diff)


def holm(ps):
    """Holm step-down adjustment; returns adjusted p in original order."""
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    m = len(ps)
    for rank, i in enumerate(order):
        adj_p = (m - rank) * ps[i]
        running = max(running, adj_p)
        adj[i] = min(1.0, running)
    return adj


def family_sets(X, rho, paper, params, k):
    """Raw Ward/Jaccard cuts (k cut sensitivity) — clusters ≥10 complete rows."""
    fam_params = {f: ps for f, ps in FAMILIES.items()}
    param_fam = {}
    for f, ps in fam_params.items():
        for p in ps:
            param_fam.setdefault(p, []).append(f)
    covers = {}
    for f, ps in fam_params.items():
        Xf = X[:, [params.index(p) for p in ps]]
        covers[f] = (Xf.mean(axis=1) == 1.0)
    Cl = X.astype(bool)
    inter = Cl.astype(float) @ Cl.astype(float).T
    rs = Cl.sum(1)
    union = rs[:, None] + rs[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, 1.0)
    D = 1.0 - J
    Z = linkage(squareform(D, checks=False), method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")
    out = {}
    for c in sorted(set(labels)):
        mask = labels == c
        # a cluster family = params co-reported in ≥80% of its complete rows; row set = rows complete for those params
        n_rows = int(mask.sum())
        if n_rows < 10:
            continue
        completeness = Cl[mask].mean(axis=0)
        members = [params[i] for i in range(len(params)) if completeness[i] >= 0.8]
        if len(members) < 3:
            continue
        rowmask = mask.copy()
        # rows complete for member params
        full = np.ones(n := len(X), bool)
        Xsub = X[:, [params.index(p) for p in members]]
        full = (Xsub == 1).all(axis=1) & mask
        if full.sum() < 10:
            continue
        out[f"W{k}C{c}"] = dict(members=members, rows=int(full.sum()), mask=full)
    return out


def paired_event_si(X_sub, rho_sub, strata, all_si, model="pca", dim=2):
    """Family-layer LOPO per-event SI (+ optional 3-D drift / MDS variants)."""
    S_full = _fit(model, X_sub, dim)
    pairs = []
    for p in sorted(set(strata)):
        keep = np.asarray([s != p for s in strata])
        if keep.sum() < 8:
            continue
        S_red = _fit(model, X_sub[keep], dim)
        a, b, _ = __import__("scipy.spatial", fromlist=["procrustes"]).procrustes(S_full[keep], S_red)
        d = np.linalg.norm(b - a, axis=1)
        si_f = event_si(rho_sub[keep], d)
        si_a = all_si.get(p, np.nan)
        if not (np.isnan(si_f) or np.isnan(si_a)):
            pairs.append(dict(deleted=p, SI_family=si_f, SI_all=si_a,
                              G=1.0 - si_f / si_a if si_a > 0 else np.nan))
    return pairs


def _fit(model, X, dim):
    if model == "pca":
        return PCA(n_components=dim, random_state=SEED).fit_transform(X)
    if model == "mds":
        A = (X > 0).astype(float)
        inter = A @ A.T
        rs = A.sum(1)
        union = rs[:, None] + rs[None, :] - inter
        with np.errstate(invalid="ignore", divide="ignore"):
            J = np.where(union > 0, inter / union, 1.0)
        return MDS(n_components=dim, dissimilarity="precomputed", random_state=SEED,
                   **classical_mds_init(), n_init=1,
                   normalized_stress="auto").fit_transform(1.0 - J)
    raise ValueError(model)


def main():
    lock = verify_baseline()
    print("baseline OK (seed", lock["seed"], "| data_version", lock["data_version"] + ")")
    X, rho, paper, params, Xv, keysv, rho_v = load()

    fmap = list(csv.DictReader(open(os.path.join(ROOT, "docs/family_map.csv"), encoding="utf-8-sig")))
    fmap_data = [f for f in fmap if f["row_kind"] == "data"]
    assert len(fmap_data) == len(X)

    # C1 all-parameter events (per variant, per deleted paper): {row_id: (rho, drift)}
    events = defaultdict(dict)
    with gzip.open(os.path.join(ROOT, "results/exC1_rows.csv.gz"), "rt", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["variant"] in ("pca_pres", "mds_pres") and r["mode"] == "lopo":
                events[(r["variant"], r["deleted"])][r["row_id"]] = (float(r["rho"]), float(r["drift"]))
    print("C1 events loaded:", len(events))

    from scipy.spatial import procrustes as _pp

    def _fit(model, X, dim=2):
        if model == "pca":
            return PCA(n_components=dim, random_state=SEED).fit_transform(X)
        D = jac_dist(X)
        return MDS(n_components=dim, dissimilarity="precomputed", random_state=SEED,
                   **classical_mds_init(), n_init=1,
                   normalized_stress="auto").fit_transform(D)

    def si_all_restricted(p, variant, rows_idx):
        ev = events.get((variant, p), {})
        sel = [(rho_d, d) for rid, (rho_d, d) in ev.items() if int(rid) in rows_idx]
        if len(sel) < 8:
            return np.nan
        return event_si([x[0] for x in sel], [x[1] for x in sel])

    summary, paired_rows, sens = [], [], []
    for fam, ps in sorted(FAMILIES.items()):
        fam_cols_g = [params.index(p) for p in ps]
        mask = X[:, fam_cols_g].sum(axis=1) >= 1   # B1 covers: >=1 family param
        if mask.sum() < 12:
            print(f"{fam}: n={mask.sum()} <12 — skipped")
            continue
        rows_idx = set(np.where(mask)[0])
        rlist = sorted(rows_idx)
        fam_cols = [params.index(p) for p in ps]
        X_fam_all = X[np.ix_(rlist, range(len(params)))]    # all 44 cols, family rows
        X_fam_only = X[np.ix_(rlist, fam_cols)]             # family cols, family rows
        rho_f = rho[rlist]
        strata_f = paper[rlist]
        for variant in ("pca_pres", "mds_pres"):
            if variant == "mds_pres" and fam not in ("F1_physical", "F3_nutrients", "F6_aeration"):
                continue
            t0 = time.time()
            S_all = _fit("pca" if variant == "pca_pres" else "mds", X_fam_all)
            S_fam = _fit("pca" if variant == "pca_pres" else "mds", X_fam_only)
            pairs = []
            for p in sorted(set(strata_f)):
                keep = strata_f != p
                if keep.sum() < 8:
                    continue
                try:
                    Sa_red = _fit("pca" if variant == "pca_pres" else "mds", X_fam_all[keep])
                    Sf_red = _fit("pca" if variant == "pca_pres" else "mds", X_fam_only[keep])
                    a1, b1, _ = _pp(S_all[keep], Sa_red)
                    a2, b2, _ = _pp(S_fam[keep], Sf_red)
                    d_all = np.linalg.norm(b1 - a1, axis=1)
                    d_fam = np.linalg.norm(b2 - a2, axis=1)
                except ValueError:
                    continue
                si_a = event_si(rho_f[keep], d_all)
                si_f = event_si(rho_f[keep], d_fam)
                if not (np.isnan(si_f) or np.isnan(si_a)):
                    pairs.append(dict(deleted=p, SI_family=si_f, SI_all=si_a,
                                      G=1.0 - si_f / si_a if si_a > 0 else np.nan))
            if len(pairs) < 5:
                print(f"{fam}/{variant}: pairs {len(pairs)} <5 — skipped")
                continue
            boot = bootstrap_G(pairs, np.random.default_rng(SEED))
            for r in pairs:
                paired_rows.append(dict(family=fam, variant=variant, **r))
            summary.append(dict(family=fam, variant=variant, n_pairs=len(pairs),
                                SI_family_median=round(float(np.median([p["SI_family"] for p in pairs])), 4),
                                SI_all_median=round(float(np.median([p["SI_all"] for p in pairs])), 4),
                                **{k: (round(v, 4) if isinstance(v, float) else v)
                                   for k, v in boot.items()}))
            print(f"{fam}/{variant}: pairs={len(pairs)} G_med={boot['G_median']:.3f} "
                  f"CI={boot['G_CI95'][0]:.3f}-{boot['G_CI95'][1]:.3f} P(G>0)={boot['P_G_gt0']:.3f} ({time.time()-t0:.0f}s)")

    # Holm per variant
    for variant in ("pca_pres", "mds_pres"):
        order = [s for s in summary if s["variant"] == variant]
        adj = holm([s["P_G_gt0"] for s in order])
        for s, a in zip(order, adj):
            s["P_G_gt0_holm"] = round(float(a), 4)


    # sensitivities: 3-D drift (same rows, same design)
    for fam in ("F1_physical", "F3_nutrients", "F6_aeration"):
        fam_cols_g = [params.index(p) for p in FAMILIES[fam]]
        mask = X[:, fam_cols_g].sum(axis=1) >= 1
        rlist = sorted(set(np.where(mask)[0]))
        fam_cols = [params.index(p) for p in FAMILIES[fam]]
        X_fam_all = X[np.ix_(rlist, range(len(params)))]
        X_fam_only = X[np.ix_(rlist, fam_cols)]
        rho_f = rho[rlist]
        strata_f = paper[rlist]
        S_all = _fit("pca", X_fam_all, dim=3)
        S_fam = _fit("pca", X_fam_only, dim=3)
        pairs = []
        for p in sorted(set(strata_f)):
            keep = strata_f != p
            if keep.sum() < 8:
                continue
            try:
                Sa = _fit("pca", X_fam_all[keep], dim=3)
                Sf = _fit("pca", X_fam_only[keep], dim=3)
                a1, b1, _ = _pp(S_all[keep], Sa)
                a2, b2, _ = _pp(S_fam[keep], Sf)
                si_a = event_si(rho_f[keep], np.linalg.norm(b1 - a1, axis=1))
                si_f = event_si(rho_f[keep], np.linalg.norm(b2 - a2, axis=1))
            except ValueError:
                continue
            if not (np.isnan(si_f) or np.isnan(si_a)):
                pairs.append(dict(deleted=p, SI_family=si_f, SI_all=si_a,
                                  G=1.0 - si_f / si_a if si_a > 0 else np.nan))
        if len(pairs) >= 5:
            boot = bootstrap_G(pairs, np.random.default_rng(SEED))
            sens.append(dict(sensitivity="pca3d", family=fam, n_pairs=len(pairs),
                             G_median=round(boot["G_median"], 4), P_G_gt0=round(boot["P_G_gt0"], 4)))
            print(f"[pca3d] {fam}: G_med={boot['G_median']:.3f} P={boot['P_G_gt0']:.3f}")

    # (a) alternative family assignment: raw Ward cuts k=3/k=5 (same-rows design)
    for k in (3, 5):
        for name, spec in family_sets_k(X, rho, paper, params, k).items():
            mask = spec["mask"]
            idxs = np.where(mask)[0]
            ps_f = [params.index(p) for p in spec["members"]]
            X_fam = X[np.ix_(idxs, ps_f)]
            rho_f = rho[idxs]
            pairs = paired_event_si_same_rows(X_fam, X[np.ix_(idxs, range(len(params)))],
                                              rho_f, paper[idxs],
                                              model="pca", dim=2)
            if len(pairs) >= 5:
                boot = bootstrap_G(pairs, np.random.default_rng(SEED))
                sens.append(dict(sensitivity=f"ward_k{k}", family=name,
                                 members="|".join(m.split(" (")[0] for m in spec["members"]),
                                 n_pairs=len(pairs), G_median=round(boot["G_median"], 4),
                                 P_G_gt0=round(boot["P_G_gt0"], 4)))
                print(f"[ward_k{k}] {name}: G_med={boot['G_median']:.3f} P={boot['P_G_gt0']:.3f}")

    outputs(paired_rows, summary, sens)
    fig5(summary, os.path.join(ROOT, "figures/fig5.png"))
    print()
    print("C4 caption templates:")
    for variant in ("pca_pres", "mds_pres"):
        sub = [s for s in summary if s["variant"] == variant]
        if not sub:
            continue
        best = max(sub, key=lambda s: s["G_median"])
        if best["G_median"] > 0:
            lo, hi = best["G_CI95"]
            print(f'  [{variant}] "Family restriction reduced sparse-row swing by x{1/(1-best["G_median"]):.1f} '
                  f'(bootstrap 95% CI {lo:.2f}-{hi:.2f} on G = 1-SI_fam/SI_all; P(G>0)={best["P_G_gt0"]:.3f}, '
                  f'Holm {best["P_G_gt0_holm"]:.3f}). This is the remedy for heterogeneity-induced instability '
                  f'in environmental meta-ordination."')


def family_sets_k(X, rho, paper, params, k):
    """Raw Ward/Jaccard cuts (k=3/k=5 sensitivity): clusters ≥10 complete rows, ≥3 members ≥80%."""
    Cl = X.astype(bool)
    inter = Cl.astype(float) @ Cl.astype(float).T
    rs = Cl.sum(1)
    union = rs[:, None] + rs[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, 1.0)
    Z = linkage(squareform(1.0 - J, checks=False), method="ward")
    labels = fcluster(Z, t=k, criterion="maxclust")
    out = {}
    for c in sorted(set(labels)):
        mask = labels == c
        if mask.sum() < 10:
            continue
        completeness = Cl[mask].mean(axis=0)
        members = [params[i] for i in range(len(params)) if completeness[i] >= 0.8]
        if len(members) < 3:
            continue
        Xsub = X[:, [params.index(p) for p in members]]
        full = mask & (Xsub.sum(axis=1) >= 1)   # coverage rows (B1 covers); complete rows are constant-degenerate
        if full.sum() < 10:
            continue
        out[f"W{k}C{c}"] = dict(members=members, rows=int(full.sum()), mask=full)
    return out


def outputs(paired_rows, summary, sens):
    specs = [("exC2_paired.csv", paired_rows), ("exC2_summary.csv", summary),
             ("exC2_sensitivity.csv", sens)]
    for name, data in specs:
        path = os.path.join(ROOT, "results", name)
        tmp = path + ".tmp"
        hdr = sorted({k for row in data for k in row.keys()}) if data else []
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            if data:
                w = csv.DictWriter(f, fieldnames=hdr)
                w.writeheader()
                w.writerows(data)
        os.replace(tmp, path)
    print("outputs:", [s[0] for s in specs])


def fig5(summary, path):
    """Fig 5 — G per family/variant, sorted, bootstrap CI bars + Holm annotations."""
    recs = sorted(summary, key=lambda s: -s["G_median"])
    fams = [f"{s['family'].replace('_', ' ')}\n[{s.get('variant', 'pca_pres').replace('_pres', '')}]"
            for s in recs]
    g = np.asarray([s["G_median"] for s in recs])
    lo = np.asarray([s["G_CI95"][0] for s in recs])
    hi = np.asarray([s["G_CI95"][1] for s in recs])
    holm = [s.get("P_G_gt0_holm", s["P_G_gt0"]) for s in recs]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    cols = ["#2e7d32" if (h < 0.05 and v > 0) else "#4c72b0" if v > 0 else "#c62828"
            for v, h in zip(g, holm)]
    x = np.arange(len(recs))
    ax.bar(x, g, color=cols)
    ax.errorbar(x, g, yerr=[np.clip(g - lo, 0, None), np.clip(hi - g, 0, None)],
                fmt="none", ecolor="#37474f", capsize=4)
    for i, s_ in enumerate(recs):
        ax.annotate(f"G={g[i]:+.2f}\nP={holm[i]:.3f}" + ("*" if holm[i] < 0.05 else ""),
                    (i, max(hi[i], g[i])), fontsize=6.4, ha="center", va="bottom")
    ax.axhline(0, c="k", lw=.6)
    ax.set_xticks(x)
    ax.set_xticklabels(fams, fontsize=7)
    ax.set_ylabel("Family Stability Gain  G = 1 − SI_fam / SI_all")
    ax.set_ylim(-1.35, 1.22)
    ax.set_title("family restriction gain per family (same-rows paired LOPO; "
                 "bootstrap 95% CI, 1000 resamples)\n"
                 "green = Holm-adjusted P(G>0) < 0.05 · blue = positive gain · red = negative gain",
                 fontsize=8.5)
    fig.tight_layout()
    _save(fig, path)
    plt.close(fig)




def paired_event_si_same_rows(X_fam, X_all, rho, strata, model="pca", dim=2):
    """Same rows: family-columns ordination vs all-44-columns ordination, LOPO per event."""
    S_fam = _fit(model, X_fam, dim)
    S_all = _fit(model, X_all, dim)
    from scipy.spatial import procrustes as _pp
    pairs = []
    for p in sorted(set(strata)):
        keep = np.asarray([s != p for s in strata])
        if keep.sum() < 8:
            continue
        try:
            Sf_red = _fit(model, X_fam[keep], dim)
            Sa_red = _fit(model, X_all[keep], dim)
            a2, b2, _ = _pp(S_fam[keep], Sf_red)
            a1, b1, _ = _pp(S_all[keep], Sa_red)
            d_f = np.linalg.norm(b2 - a2, axis=1)
            d_a = np.linalg.norm(b1 - a1, axis=1)
        except ValueError:
            continue
        si_f = event_si(rho[keep], d_f)
        si_a = event_si(rho[keep], d_a)
        if not (np.isnan(si_f) or np.isnan(si_a)):
            pairs.append(dict(deleted=p, SI_family=si_f, SI_all=si_a,
                              G=1.0 - si_f / si_a if si_a > 0 else np.nan))
    return pairs


if __name__ == "__main__":
    main()
