#!/usr/bin/env python3
"""robustness_exclB175.py — D108 #4: headline numbers with the duplicate paper (B175 ⊂ B111) removed."""
from __future__ import annotations
import csv, os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_C1 import load, run_variant, swing_stats
from experiment_C2_stability import paired_event_si_same_rows, bootstrap_G, FAMILIES, SEED

X, rho, paper, params, Xv, keysv, rho_v = load()
keep = paper != "B175"
Xk, rhok, pk = X[keep], rho[keep], paper[keep]
keepv = np.asarray([k[0] != "B175" for k in keysv])
Xvk = Xv[keepv]; keysk = [k for k in keysv if k[0] != "B175"]; rhovk = rho_v[keepv]
print(f"rows: {len(X)} → {len(Xk)} (removed {int((~keep).sum())} B175 rows) | value rows {len(Xv)} → {len(Xvk)}")

out=[]
for variant in ("pca_pres", "mds_pres", "pca_vstd"):
    rows = run_variant(variant, Xk, Xvk, rhovk if variant.startswith("pca_v") else rhok,
                       pk, [str(i) for i in range(len(Xk))], keysk, modes=("lopo",))
    rr = np.asarray([r["rho"] for r in rows]); dd = np.asarray([r["drift"] for r in rows])
    st = swing_stats(rr, dd, np.asarray([r["deleted"] for r in rows]))
    out.append(dict(test=f"C1 {variant} LOPO SI", value=round(st["SI"],4), perm_p=round(st["perm_p"],4),
                    n_events=len({r['deleted'] for r in rows})))
    print(f"C1 {variant}: SI={st['SI']:.4f} p={st['perm_p']:.4f}")

# C2 F6 paired G excluding B175
fam="F6_aeration"
fam_cols=[params.index(p) for p in FAMILIES[fam]]
mask = Xk[:, fam_cols].sum(axis=1) >= 1
idxs=np.where(mask)[0]
X_fam = Xk[np.ix_(idxs, fam_cols)]; X_all = Xk[np.ix_(idxs, range(len(params)))]
pairs = paired_event_si_same_rows(X_fam, X_all, rhok[idxs], pk[idxs], model="pca", dim=2)
boot = bootstrap_G(pairs, np.random.default_rng(SEED))
out.append(dict(test="C2 F6_aeration paired G", value=round(boot["G_median"],4),
                perm_p=round(boot["P_G_gt0"],4), n_events=len(pairs)))
print(f"C2 F6 G={boot['G_median']:.4f} P(G>0)={boot['P_G_gt0']:.4f} (n={len(pairs)})")

ap=os.path.join(ROOT,"results/robustness_exclB175.csv")
with open(ap,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("written results/robustness_exclB175.csv")
