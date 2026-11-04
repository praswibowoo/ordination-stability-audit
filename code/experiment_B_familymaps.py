#!/usr/bin/env python3
"""experiment_B_familymaps.py — Experiment B: Family-Restricted Maps (P1-4.2 · Analysis_Design §ExB).

B1  family-restricted conditioning (presence + value variants)
B2  family PCA + loadings (+ varimax)
B3  verdict statistics per family:
      PERMANOVA (999 perms, strata=paper, rho-residualized + raw) · ANOSIM R ·
      silhouette (verdict labels vs shuffled 1000×) · silhouette per density quartile
B4  loadings interpretability table
B5  figures: fig3_<family>.png biplots (verdict colors + loading arrows) · fig4 F2 v1-vs-expanded

Baseline guard (results/phase4_baseline.json) verified before running.
Outputs: results/exB_*.csv · figures/fig3_*.png · docs/phase4_C3_pack.md (assembled separately)

Run: python3 code/experiment_B_familymaps.py
"""
from __future__ import annotations
import csv, json, os, sys
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117)
from collections import defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
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



SEED = 42
PERM_N = 999
SIL_N = 1000
FIGS = os.path.join(ROOT, "figures")
RESULTS = os.path.join(ROOT, "results")

FAMILIES = {
    "F1_physical": ["DO (mg/L)", "pH", "Salinity (ppt/\u2030)", "Temperature (\u00b0C)"],
    "F2_metals": ["As (\u00b5g/L)", "Cd (\u00b5g/L)", "Cr (\u00b5g/L)", "Cu (\u00b5g/L)",
                  "Fe (\u00b5g/L)", "Ni (\u00b5g/L)", "Pb (\u00b5g/L)", "Zn (\u00b5g/L)", "Co (\u00b5g/L)"],
    "F3_nutrients": ["NO2 (mg/L)", "NO3 (mg/L)", "NH4 (mg/L)", "PO4 (mg/L)"],
    "F4_organic": ["COD (mg/L)", "BOD (mg/L)", "Total Coliform (CFU/100mL)"],
    "F1_physicala": ["TDS", "EC (\u00b5S/cm)", "Cl- (ppt)", "Hardness (mg/L)", "Alkalinity (mg/L)"],
    "F1_clarity": ["Turbidity (NTU)", "Secchi (m)", "TSS (mg/L)"],
    "F6_aeration": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)"],
}
PALETTE = {"Favorable": "#2e7d32", "Stable": "#66bb6a", "Mild stress": "#f9a825",
           "Stressful": "#ef6c00", "Harmful": "#c62828", "Reference standard": "#9e9e9e",
           "TBD-StageB": "#cfd8dc", "": "#eceff1"}
# Shape is a colour-independent encoding of the verdict class, so the biplots
# remain readable for colour-vision-deficient readers (Fig. 3 and SI panels).
MARKERS = {"Favorable": "o", "Stable": "s", "Mild stress": "^", "Stressful": "D",
           "Harmful": "v", "Reference standard": "P", "Reference baseline": "X",
           "Degraded": "<", "TBD-StageB": ".", "": "."}


def load_all():
    pres = list(csv.DictReader(open(os.path.join(
        ROOT, "docs/extraction/stageA/matrix_presence_v2.csv"), encoding="utf-8-sig")))
    hdr = list(pres[0].keys())
    params = hdr[7:51]
    rows, own, ordn = [], "", 0
    for r in pres:
        ordn += 1
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        if r["row_kind"] != "data":
            continue
        rows.append(dict(ordinal=ordn, batch=own, params=params,
                         X=[1.0 if r[p].strip() in ("Y", "NE") else 0.0 for p in params],
                         verdict=(r["verdict"] or "").strip(),
                         note=(r["cause_effect_note"] or "")))
    vals = defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(
            ROOT, "docs/extraction/stageB/matrix_values_v2.csv"), encoding="utf-8-sig")):
        if r["is_real_value"].strip() == "TRUE" and (r["value"] or "").strip():
            try:
                vals[(r["batch_id"], r["row_id"])][r["param"]] = float(r["value"])
            except ValueError:
                pass
    fmap = list(csv.DictReader(open(os.path.join(ROOT, "docs/family_map.csv"),
                                    encoding="utf-8-sig")))
    return rows, vals, fmap, params


def family_subset(rows, vals, fmap, family, fam_params):
    """ALL data rows with per-row family masks + sliced family presence vector."""
    fmap_data = [fm for fm in fmap if fm["row_kind"] == "data"]
    assert len(fmap_data) == len(rows)
    col = f"full_{family}"
    fam_idx = [rows[0]["params"].index(p) for p in fam_params]
    out = []
    for fm, r in zip(fmap_data, rows):
        X_fam = [r["X"][i] for i in fam_idx]
        v = vals.get((r["batch"], str(r["ordinal"])), {})
        out.append(dict(row=r, values=v, X_fam=X_fam,
                        cov=any(x > 0 for x in X_fam),
                        complete=all(x > 0 for x in X_fam),
                        vcomplete=all(p in v for p in fam_params)))
    return out


def median_of(v):
    v = np.asarray(v, float)
    return float(np.median(v)) if len(v) else float("nan")


def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    """Standard varimax rotation of a (n_params × n_components) loading matrix."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        L = Phi @ R
        u, s, vt = np.linalg.svd(
            Phi.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vt
        d_new = s.sum()
        if d_new < d * (1 + tol):
            break
        d = d_new
    return Phi @ R


def gower_from_dist(D):
    n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    return -0.5 * J @ D @ J


def permanova_f(D, labels):
    """Anderson (2001) pseudo-F — consistent distance-based pair:
    SS_T = sum(d²)/(2n) · SS_W = sum_g (sum_g d²)/(2n_g) · SS_A = SS_T − SS_W."""
    D2 = np.asarray(D, float) ** 2
    labels = np.asarray(labels)
    ids = sorted(set(labels))
    n = len(labels)
    groups = [np.where(labels == g)[0] for g in ids]
    sst = D2.sum() / (2 * n)
    ssw = sum((D2[np.ix_(g, g)].sum() / (2 * len(g))) for g in groups if len(g) > 0)
    ssa = sst - ssw
    a = len(ids)
    if a < 2 or n - a <= 0:
        return np.nan, np.nan
    f = (ssa / (a - 1)) / (ssw / (n - a))
    return f, ssw


def permanova_test(D, labels, strata, rng, n_perm=PERM_N):
    """Pseudo-F + permutation p (labels permuted within strata=paper)."""
    labels = np.asarray(labels, dtype=object)
    f_obs, _ = permanova_f(D, list(labels))
    strata_map = defaultdict(list)
    for i, s in enumerate(strata):
        strata_map[s].append(i)
    count = 0
    total = 0
    for _ in range(n_perm):
        lab = labels.copy()
        for idx in strata_map.values():
            lab[idx] = rng.permutation(lab[idx])
        f_p, _ = permanova_f(D, list(lab))
        if not np.isnan(f_p):
            total += 1
            count += f_p >= f_obs
    p = (count + 1) / (total + 1) if total else np.nan
    uniq = len({tuple(np.asarray(labels)[s].tolist() for s in []) for _ in []}) or None
    return f_obs, p, total


def anosim_r(D, labels):
    n = len(labels)
    iu = np.triu_indices(n, 1)
    d = D[iu]
    labs = np.asarray(labels)
    same = labs[iu[0]] == labs[iu[1]]
    ranks = stats_ranks(d)
    r_w = ranks[same].mean() if same.any() else np.nan
    r_b = ranks[~same].mean() if (~same).any() else np.nan
    denom = n * (n - 1) / 4
    return (r_b - r_w) / denom


def stats_ranks(x):
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x))
    ranks[order] = np.arange(1, len(x) + 1)
    return ranks


def permute_anosim_p(D, labels, strata, rng, n_perm=PERM_N):
    labels = np.asarray(labels, dtype=object)
    r_obs = anosim_r(D, list(labels))
    strata_map = defaultdict(list)
    for i, s in enumerate(strata):
        strata_map[s].append(i)
    count = total = 0
    for _ in range(n_perm):
        lab = labels.copy()
        for idx in strata_map.values():
            lab[idx] = rng.permutation(lab[idx])
        r_p = anosim_r(D, list(lab))
        if not np.isnan(r_p):
            total += 1
            count += r_p >= r_obs
    p = (count + 1) / (total + 1) if total else np.nan
    return r_obs, p


def silhouette_gate(D, labels, strata, rng, n_sil=SIL_N):
    """Silhouette on verdict-labeled subset (groups n>=3) vs shuffled labels."""
    lab = np.asarray(labels, dtype=object)
    keep = np.ones(len(lab), bool)
    counts = {g: int((lab == g).sum()) for g in set(lab)}
    for g, c in counts.items():
        if c < 3:
            keep &= lab != g
    idx = np.where(keep)[0]
    if len(idx) < 6 or len(set(lab[idx])) < 2:
        return dict(n=len(idx), sil=np.nan, sil_null_mean=np.nan, sil_p=np.nan, groups=len(set(lab[idx])))
    D_sub = D[np.ix_(idx, idx)]
    l_sub = lab[idx]
    strata_map = defaultdict(list)
    pos_of = {row_pos: sub_pos for sub_pos, row_pos in enumerate(idx)}
    for pos, s in enumerate(strata):
        if keep[pos]:
            strata_map[s].append(pos_of[pos])
    try:
        sil = silhouette_score(D_sub, l_sub, metric="precomputed")
    except ValueError:
        return dict(n=len(idx), sil=np.nan, sil_null_mean=np.nan, sil_p=np.nan, groups=len(set(l_sub)))
    null = np.empty(n_sil)
    for b in range(n_sil):
        l_p = l_sub.copy()
        for ids in strata_map.values():
            if len(ids) > 1:
                l_p[ids] = rng.permutation(l_p[ids])
        try:
            null[b] = silhouette_score(D_sub, l_p, metric="precomputed")
        except ValueError:
            null[b] = np.nan
    null = null[~np.isnan(null)]
    p = float((np.sum(null >= sil) + 1) / (len(null) + 1)) if len(null) else np.nan
    return dict(n=len(idx), sil=float(sil), sil_null_mean=float(np.nanmean(null)) if len(null) else np.nan,
                sil_p=p, groups=len(set(l_sub)))


def residualize(S, rho):
    """Regress each PC column on rho, return residuals (B3.4 confounder control)."""
    S = np.asarray(S, float)
    rho_c = np.asarray(rho, float) - rho.mean()
    out = np.empty_like(S)
    for j in range(S.shape[1]):
        beta = np.dot(rho_c, S[:, j] - S[:, j].mean()) / max(np.dot(rho_c, rho_c), 1e-12)
        out[:, j] = S[:, j] - (S[:, j].mean() + beta * rho_c)
    return out


def rho_of_row(row_dict):
    return float(np.mean(row_dict["X"]))


def main():
    lock = verify_baseline()
    rows, vals, fmap, params = load_all()
    print("data rows:", len(rows), "| family_map rows:", len(fmap))
    perm_rows, anosim_rows, sil_rows, load_rows, si_rows = [], [], [], [], []

    for family, fam_params in FAMILIES.items():
        subset = family_subset(rows, vals, fmap, family, fam_params)
        # ---- presence variant (B1: rows COVERING the family, v1-continuity) ----
        cov = [s for s in subset if s["cov"]]
        X = np.asarray([s["X_fam"] for s in cov], float)
        rho = np.asarray([rho_of_row(s["row"]) for s in cov])
        n_f = len(cov)
        papers_f = len({s["row"]["batch"] for s in cov})
        if n_f < 6:
            print(f"{family}: skipped (n={n_f})")
            continue
        verdicts = [s["row"]["verdict"] or "TBD-StageB" for s in cov]
        strata = [s["row"]["batch"] for s in cov]
        # Between-group tests use LABELED rows only. Rows with no verdict (blank
        # or TBD-StageB) are undecided, not a verdict class; including them as a
        # group inflates F and depresses silhouette (R5 finding). They stay in
        # the ordination and the figure, but not in the tests.
        lab = np.asarray([v not in ("", "TBD-StageB") for v in verdicts])
        verdicts_lab = [v for v, k in zip(verdicts, lab) if k]
        strata_lab = [s for s, k in zip(strata, lab) if k]
        n_lab = int(lab.sum())
        papers_lab = len({s["row"]["batch"] for s, k in zip(cov, lab) if k})
        pca = PCA(n_components=2, random_state=SEED).fit(X)
        S = pca.transform(X)
        evr = pca.explained_variance_ratio_
        loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
        load_rot = varimax(loadings)
        for i, p in enumerate(fam_params):
            load_rows.append(dict(family=family, param=p, PC1=round(loadings[i, 0], 4),
                                  PC2=round(loadings[i, 1], 4),
                                  PC1_varimax=round(load_rot[i, 0], 4),
                                  PC2_varimax=round(load_rot[i, 1], 4),
                                  evr_PC1=round(float(evr[0]), 4), evr_PC2=round(float(evr[1]), 4)))
        D = jac_dist_family(X)
        D_lab = D[np.ix_(lab, lab)]
        f_raw, p_raw, np_ = permanova_test(D_lab, verdicts_lab, strata_lab,
                                           np.random.default_rng(SEED))
        f_res, p_res, _ = permanova_test(D_lab, verdicts_lab, strata_lab, np.random.default_rng(SEED + 1)) \
            if False else (np.nan, np.nan, 0)
        # rho-residualized: residualize PC scores then recompute D via residual scores
        S_res = residualize(S, rho)
        D_res = squareform(pdist(S_res))
        D_res_lab = D_res[np.ix_(lab, lab)]
        f_res, p_res, np2 = permanova_test(D_res_lab, verdicts_lab, strata_lab,
                                           np.random.default_rng(SEED))
        perm_rows.append(dict(family=family, layer="presence", n=n_lab, papers=papers_lab,
                              groups=len(set(verdicts_lab)),
                              pseudo_F=round(float(f_raw), 4), p_perm=round(float(p_raw), 4),
                              pseudo_F_rho_residualized=round(float(f_res), 4),
                              p_perm_rho_residualized=round(float(p_res), 4)))
        r_obs, r_p = permute_anosim_p(D_lab, verdicts_lab, strata_lab,
                                      np.random.default_rng(SEED + 2))
        anosim_rows.append(dict(family=family, layer="presence", R=round(float(r_obs), 4),
                                p_perm=round(float(r_p), 4)))
        sil = silhouette_gate(D_lab, verdicts_lab, strata_lab, np.random.default_rng(SEED + 3))
        sil_rows.append(dict(family=family, layer="presence", **sil))
        # LOPO SI per family (feeds C2 pairing)
        rho_fam = X.mean(axis=1)   # density WITHIN family params
        si = family_lopo_si(X, X.mean(axis=1), strata)
        si_rows.append(dict(family=family, layer="presence", n_rows=n_f, papers=papers_f,
                            SI=round(si, 4)))
        fig_biplot(family, S, loadings, verdicts, evr, "presence", cov, fam_params)
        print(f"{family}: n={n_f} papers={papers_f} | PERMANOVA F={f_raw:.3f} p={p_raw:.3f} | "
              f"ANOSIM R={r_obs:.3f} p={r_p:.3f} | sil={sil.get('sil', float('nan')):.3f} p={sil.get('sil_p', float('nan')):.3f} | SI={si:.3f}")

        # ---- value variant (complete family rows only) ----
        vcomp = [s for s in subset if s["vcomplete"]]
        if len(vcomp) >= 10:
            Xv = np.asarray([[s["values"][p] for p in fam_params] for s in vcomp], float)
            verd_v = [s["row"]["verdict"] or "TBD-StageB" for s in vcomp]
            strat_v = [s["row"]["batch"] for s in vcomp]
            # Same rule as the presence layer: undecided rows are not a class.
            lab_v = np.asarray([v not in ("", "TBD-StageB") for v in verd_v])
            verd_v_lab = [v for v, k in zip(verd_v, lab_v) if k]
            strat_v_lab = [s for s, k in zip(strat_v, lab_v) if k]
            n_v_lab = int(lab_v.sum())
            papers_v_lab = len({s["row"]["batch"] for s, k in zip(vcomp, lab_v) if k})
            rho_v = np.full(len(vcomp), np.nan)
            try:
                Zv = StandardScaler().fit_transform(Xv)  # stats + biplot SAME space (D115 fix)
                pca_v = PCA(n_components=2).fit(Zv)
                S_v = pca_v.transform(Zv)
                evr_v = pca_v.explained_variance_ratio_
                Dv = squareform(pdist(Zv))
                Dv_lab = Dv[np.ix_(lab_v, lab_v)]
                f_v, p_v, _ = permanova_test(Dv_lab, verd_v_lab, strat_v_lab,
                                             np.random.default_rng(SEED))
                perm_rows.append(dict(family=family, layer="value", n=n_v_lab,
                                      papers=papers_v_lab,
                                      groups=len(set(verd_v_lab)),
                                      pseudo_F=round(float(f_v), 4), p_perm=round(float(p_v), 4),
                                      pseudo_F_rho_residualized=np.nan,
                                      p_perm_rho_residualized=np.nan))
                r_v, rp_v = permute_anosim_p(Dv_lab, verd_v_lab, strat_v_lab,
                                             np.random.default_rng(SEED + 2))
                anosim_rows.append(dict(family=family, layer="value", R=round(float(r_v), 4),
                                        p_perm=round(float(rp_v), 4)))
                sil_v = silhouette_gate(Dv_lab, verd_v_lab, strat_v_lab,
                                        np.random.default_rng(SEED + 3))
                sil_rows.append(dict(family=family, layer="value", **sil_v))
                si_v = family_lopo_si(Xv, np.ones(len(vcomp)), strat_v)
                si_rows.append(dict(family=family, layer="value", n_rows=len(vcomp),
                                    papers=len({s["row"]["batch"] for s in vcomp}), SI=round(si_v, 4)))
                fig_biplot(family, S_v, pca_v.components_.T * np.sqrt(pca_v.explained_variance_),
                           verd_v, evr_v, "value", vcomp, fam_params)
                print(f"  {family} value: n={len(vcomp)} F={f_v:.3f} p={p_v:.3f} R={r_v:.3f}")
            except Exception as e:
                print(f"  {family} value variant failed: {type(e).__name__} {str(e)[:60]}")
        else:
            print(f"  {family} value variant skipped (n={len(vcomp)})")
    outputs(perm_rows, anosim_rows, sil_rows, load_rows, si_rows)


def jac_dist_family(X):
    A = (X > 0).astype(float)
    inter = A @ A.T
    rs = A.sum(1)
    union = rs[:, None] + rs[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        J = np.where(union > 0, inter / union, 1.0)
    return 1.0 - J


def family_lopo_si(X_fam, rho_fam, strata):
    """Leave-one-paper-out per-row drift -> quartile-ratio SI on family-param density."""
    papers = sorted(set(strata))
    S_full = PCA(n_components=2, random_state=SEED).fit_transform(X_fam)
    drifts, rhos = [], []
    for p in papers:
        keep = np.asarray([s != p for s in strata])
        if keep.sum() < 3:
            continue
        try:
            S_red = PCA(n_components=2, random_state=SEED).fit_transform(X_fam[keep])
            d, _ = drift_after_procrustes_b(S_full[keep], S_red)
        except ValueError:
            continue
        drifts.extend(d.tolist())
        rhos.extend(rho_fam[keep].tolist())
    drifts = np.asarray(drifts)
    rhos = np.asarray(rhos)
    if len(drifts) < 8:
        return float("nan")
    q1, q3 = np.quantile(rhos, [0.25, 0.75])
    m_lo = median_of(drifts[rhos <= q1])
    m_hi = median_of(drifts[rhos >= q3])
    return m_lo / m_hi if m_hi and m_hi > 0 else float("nan")


def drift_after_procrustes_b(S_ref, S_red):
    a, b, disp = scipy_procrustes_compat(S_ref, S_red)
    return np.linalg.norm(b - a, axis=1), disp


def scipy_procrustes_compat(A, B):
    from scipy.spatial import procrustes as _spp
    return _spp(A, B)


SEV_ORDER = ["Favorable", "Stable", "Mild stress", "Stressful", "Harmful",
             "Reference standard", "Reference baseline", "Degraded", "TBD-StageB"]


def fig_biplot(family, S, loadings, verdicts, evr, layer, subset, param_names):
    """Biplot: audited PCA scores by verdict + unit-normalized labeled arrows (D115).
    Display winsorization at 1.5×IQR fences (D120) — display-only, annotated; raw scores untouched."""
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    S_d = S.copy()
    n_out = 0
    for k in range(2):
        q1, q3 = np.quantile(S_d[:, k], [0.25, 0.75])
        f_lo, f_hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        n_out += int(((S_d[:, k] < f_lo) | (S_d[:, k] > f_hi)).sum())
        S_d[:, k] = np.clip(S_d[:, k], f_lo, f_hi)
    present = [v for v in SEV_ORDER if v in set(verdicts)]
    present += sorted(v for v in set(verdicts) if v not in SEV_ORDER)
    for v in present:
        m = np.asarray([vv == v for vv in verdicts])
        ax.scatter(S_d[m, 0], S_d[m, 1], s=14, c=PALETTE.get(v, "#eceff1"), alpha=.55, label=v)
    xr = float(np.ptp(S_d[:, 0])) or 1.0
    yr = float(np.ptp(S_d[:, 1])) or 1.0
    rad = 0.34 * max(xr, yr)
    for j in range(loadings.shape[0]):
        vx, vy = float(loadings[j, 0]), float(loadings[j, 1])
        nrm = float(np.hypot(vx, vy))
        if nrm < 1e-12:
            continue
        ux, uy = vx / nrm * rad, vy / nrm * rad
        ax.annotate("", xy=(ux, uy), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=1.2, color="#37474f"))
        ax.annotate(param_names[j].split(" (")[0], (ux, uy),
                    xytext=(4 if ux >= 0 else -4, 4 if uy >= 0 else -4),
                    textcoords="offset points", fontsize=6.5, color="#263238",
                    ha="left" if ux >= 0 else "right",
                    va="bottom" if uy >= 0 else "top")
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_xlim(min(x0, -rad * 1.15), max(x1, rad * 1.15))
    ax.set_ylim(min(y0, -rad * 1.15), max(y1, rad * 1.15))
    ax.axhline(0, lw=.5, c="#90a4ae"); ax.axvline(0, lw=.5, c="#90a4ae")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title(f"{family} [{layer}] — verdict scores + parameter arrows\n"
                 f"(PCA on {'standardized values' if layer == 'value' else 'presence flags'}; "
                 "arrows unit-normalized)", fontsize=8.5)
    if n_out:
        ax.annotate(f"{n_out} score(s) beyond 1.5×IQR display fence (clipped for readability)",
                    (0.01, 0.01), xycoords="axes fraction", fontsize=7.0, color="#37474f")
    ax.legend(fontsize=6.2, markerscale=1.6, loc="best")
    fig.tight_layout()
    # D145: F1/F3 value panels are re-rendered at print size by make_fig3_print.py
    # and stitched into fig3_main.png; the standalone PNG/PDF were never referenced
    # by any document, so they are suppressed as build byproducts.
    if (family, layer) in (("F1_physical", "value"), ("F3_nutrients", "value")):
        plt.close(fig)
        return
    _save(fig, os.path.join(FIGS, f"fig3_{family}_{layer}.png"))
    plt.close(fig)


def add_holm_presence(perm_rows):
    """Holm adjustment of the density-residualized presence-layer p-values.

    The seven presence families are adjusted together (as in the S12a note and
    the SI prose), so the adjusted values are recoverable from the artifact
    instead of being quoted as literals.
    """
    pres = [r for r in perm_rows if r.get("layer") == "presence"]
    ps = sorted((float(r["p_perm_rho_residualized"]), r["family"]) for r in pres
                if str(r.get("p_perm_rho_residualized")) not in ("", "nan", "None"))
    m = len(ps)
    for i, (p, fam) in enumerate(ps):
        for r in pres:
            if r["family"] == fam:
                r["p_perm_rho_residualized_holm"] = round(min(1.0, p * (m - i)), 4)
    for r in perm_rows:
        r.setdefault("p_perm_rho_residualized_holm", "")
    return perm_rows


def outputs(perm_rows, anosim_rows, sil_rows, load_rows, si_rows):
    perm_rows = add_holm_presence(perm_rows)
    specs = [("exB_permanova.csv", perm_rows), ("exB_anosim.csv", anosim_rows),
             ("exB_silhouette.csv", sil_rows), ("exB_loadings.csv", load_rows),
             ("exB_family_si.csv", si_rows)]
    for name, data in specs:
        path = os.path.join(RESULTS, name)
        tmp = path + ".tmp"
        with open(tmp, "w", newline="", encoding="utf-8") as f:
            if data:
                w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
                w.writeheader(); w.writerows(data)
        os.replace(tmp, path)
    print("outputs:", [s[0] for s in specs])


if __name__ == "__main__":
    main()
    # outputs need the collected lists — rerun pattern: main collects internally
