#!/usr/bin/env python3
"""experiment_J_diagnostics.py — diagnostics for the two contested mechanism claims.

Answers two questions raised in review, from frozen layers only:

  (1) "Configuration saturation": how many distinct presence configurations does each
      family actually realise among its cover rows (and among its value-complete rows)?
      A family-level claim of saturation is only supportable if the count is small.

  (2) "PC3 captures noise/calibration offsets": what do the three leading components of
      the family-restricted fit contain (variance explained, PC3 loadings)?

Also records whether the per-event family-restricted drift is constant across rows
(the mechanical reason SI_fam can equal 1.000 exactly).

Reads frozen presence layer + value layer. Writes results/exJ_diagnostics.csv.
Run: python3 code/experiment_J_diagnostics.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")

from experiment_C1 import load  # noqa: E402

FAMS = {
    "F1_physical": ["DO (mg/L)", "pH", "Salinity (ppt/\u2030)", "Temperature (\u00b0C)"],
    "F2_metals": ["As (\u00b5g/L)", "Cd (\u00b5g/L)", "Cr (\u00b5g/L)", "Cu (\u00b5g/L)",
                  "Fe (\u00b5g/L)", "Ni (\u00b5g/L)", "Pb (\u00b5g/L)", "Zn (\u00b5g/L)",
                  "Co (\u00b5g/L)"],
    "F3_nutrients": ["NO2 (mg/L)", "NO3 (mg/L)", "NH4 (mg/L)", "PO4 (mg/L)"],
    "F4_organic": ["COD (mg/L)", "BOD (mg/L)", "Total Coliform (CFU/100mL)"],
    "F1_physicala": ["TDS", "EC (\u00b5S/cm)", "Cl- (ppt)", "Hardness (mg/L)",
                     "Alkalinity (mg/L)"],
    "F1_clarity": ["Turbidity (NTU)", "Secchi (m)", "TSS (mg/L)"],
    "F6_aeration": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)"],
}


def main() -> None:
    X, rho, paper, params, _Xv, _keysv, _rho_v = load()
    idx = {p: i for i, p in enumerate(params)}
    out = []

    # corpus-wide denominators quoted in the manuscript
    cover_any = int((X > 0).any(axis=1).sum())
    fam_cols = sorted({i for ps in FAMS.values() for i in (idx[p] for p in ps)})
    fcols = [idx[p] for ps in FAMS.values() for p in ps]
    cover_family = int((X[:, fcols].sum(axis=1) >= 1).sum())
    complete_family = int(np.array([any((X[i, [idx[p] for p in ps]] == 1).all()
                                        for ps in FAMS.values()) for i in range(len(X))]).sum())
    print(f"corpus: data rows={len(X)}  cover>=1 family={cover_family}  "
          f"presence-complete>=1 family={complete_family}")
    out.append(dict(family="ALL", layer="corpus", n_cols=len(params), n_rows=len(X),
                    n_configs=complete_family, max_possible=cover_family,
                    pc1_var="", pc2_var="", pc3_var="", pc3_loadings=""))

    print(f"{'family':14s} {'cols':>4s} {'cover':>6s} {'configs':>8s} {'maxpos':>6s}")
    for fam, ps in FAMS.items():
        cols = [idx[p] for p in ps]
        cover = X[:, cols].sum(axis=1) >= 1
        Xc = X[np.ix_(np.where(cover)[0], cols)]
        configs = len({tuple(r.astype(int)) for r in Xc})
        out.append(dict(family=fam, layer="presence", n_cols=len(cols),
                        n_rows=int(cover.sum()), n_configs=configs,
                        max_possible=2 ** len(cols),
                        pc1_var="", pc2_var="", pc3_var="", pc3_loadings=""))
        print(f"{fam:14s} {len(cols):4d} {int(cover.sum()):6d} {configs:8d} "
              f"{2**len(cols):6d}")

    # three-component structure of the two families that carry the 3-D discussion
    print("\n3-D PCA on the family-restricted presence fit (cover rows):")
    for fam in ("F6_aeration", "F1_physical"):
        cols = [idx[p] for p in FAMS[fam]]
        cover = X[:, cols].sum(axis=1) >= 1
        Xc = X[np.ix_(np.where(cover)[0], cols)]
        p = PCA(n_components=min(3, len(cols)), random_state=42).fit(Xc)
        vr = p.explained_variance_ratio_
        print(f"  {fam}: n={len(Xc)} vars={vr.round(4).tolist()}")
        for k in range(p.components_.shape[0]):
            ld = p.components_[k]
            print(f"    PC{k+1} loadings: "
                  + ", ".join(f"{FAMS[fam][j].split(' (')[0]}={ld[j]:+.3f}"
                              for j in range(len(ld))))
        out.append(dict(family=fam, layer="presence-pca3d", n_cols=len(cols),
                        n_rows=len(Xc), n_configs=len({tuple(r.astype(int)) for r in Xc}),
                        max_possible=2 ** len(cols),
                        pc1_var=round(float(vr[0]), 4),
                        pc2_var=round(float(vr[1]), 4) if len(vr) > 1 else "",
                        pc3_var=round(float(vr[2]), 4) if len(vr) > 2 else "",
                        pc12_pct=round(float(vr[0] + vr[1]) * 100, 1) if len(vr) > 1 else "",
                        pc1_pct=round(float(vr[0]) * 100, 1),
                        pc2_pct=round(float(vr[1]) * 100, 1) if len(vr) > 1 else "",
                        pc3_pct=round(float(vr[2]) * 100, 1) if len(vr) > 2 else "",
                        pc3_loadings="|".join(
                            f"{FAMS[fam][j].split(' (')[0]}={p.components_[2][j]:+.3f}"
                            for j in range(len(cols))) if p.components_.shape[0] > 2 else ""))

    # how many distinct per-row drift values does the restricted fit produce per event?
    print("\nper-event distinct drift values (F1 physical presence, all events):")
    import gzip  # event layer of the paired design is not stored per row; recompute one
    import matplotlib
    matplotlib.use("Agg")
    from experiment_C2_stability import _fit, event_si
    from scipy.spatial import procrustes as _pp
    cols = [idx[p] for p in FAMS["F1_physical"]]
    cover = X[:, cols].sum(axis=1) >= 1
    rlist = np.where(cover)[0]
    Xf = X[np.ix_(rlist, cols)]
    strata = paper[rlist]
    S_full = _fit("pca", Xf, 2)
    import collections
    counts = []
    for p_ in sorted(set(strata)):
        keep = strata != p_
        S_red = _fit("pca", Xf[keep], 2)
        a, b, _ = _pp(S_full[keep], S_red)
        d = np.linalg.norm(b - a, axis=1)
        counts.append(len(np.unique(np.round(d, 12))))
    modal, n_modal = collections.Counter(counts).most_common(1)[0]
    print(f"  events={len(counts)}  modal distinct drift values={modal}  "
          f"events at modal={n_modal}  distribution={dict(collections.Counter(counts))}")
    out.append(dict(family="F1_physical", layer="lopo-drift-classes", n_cols=len(cols),
                    n_rows=n_modal, n_configs=modal, max_possible=len(counts),
                    pc1_var="", pc2_var="", pc3_var="", pc3_loadings=""))

    path = os.path.join(RESULTS, "exJ_diagnostics.csv")
    hdr = ["family", "layer", "n_cols", "n_rows", "n_configs", "max_possible",
           "pc1_var", "pc2_var", "pc3_var", "pc12_pct", "pc1_pct", "pc2_pct",
           "pc3_pct", "pc3_loadings"]
    with open(path + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=hdr)
        w.writeheader()
        w.writerows(out)
    os.replace(path + ".tmp", path)
    print(f"\nwrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
