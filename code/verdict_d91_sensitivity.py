#!/usr/bin/env python3
"""verdict_d91_sensitivity.py — corrections pass: impact of the D91 range-banding fix.

The D91 fix restated 9 historical verdicts to Harmful (B031 rows 163-167,
B084 rows 343-345, B127 row 505). Five of those rows are family-complete in
F1 physical and four in F6 aeration, so the D91 fix is inside the value-layer
subsets used by the verdict statistics. This script recomputes the PERMANOVA,
ANOSIM and silhouette results for the affected families with those rows removed
and compares them with the frozen results.

Reads frozen layers only. Writes results/exB_d91_sensitivity.csv.
Run: python3 code/verdict_d91_sensitivity.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
RESULTS = os.path.join(ROOT, "results")
SEED = 42

from experiment_B_familymaps import (FAMILIES, load_all, family_subset,
                                     permanova_test, permute_anosim_p,
                                     silhouette_gate)  # noqa: E402

D91 = {("B031", r) for r in range(163, 168)} | \
      {("B084", r) for r in range(343, 346)} | {("B127", 505)}


def main() -> None:
    rows, vals, fmap, params = load_all()
    out_rows = []
    for family in ("F1_physical", "F6_aeration"):
        fam_params = FAMILIES[family]
        subset = family_subset(rows, vals, fmap, family, fam_params)
        vcomp = [s for s in subset if s["vcomplete"]]
        keep = [s for s in vcomp
                if (s["row"]["batch"], s["row"]["ordinal"]) not in D91]
        dropped = len(vcomp) - len(keep)
        Xv = np.asarray([[s["values"][p] for p in fam_params] for s in keep], float)
        verd = [s["row"]["verdict"] or "TBD-StageB" for s in keep]
        strata = [s["row"]["batch"] for s in keep]
        from sklearn.preprocessing import StandardScaler
        from scipy.spatial.distance import pdist, squareform
        Z = StandardScaler().fit_transform(Xv)
        D = squareform(pdist(Z))
        f, p, _ = permanova_test(D, verd, strata, np.random.default_rng(SEED))
        r, rp = permute_anosim_p(D, verd, strata, np.random.default_rng(SEED + 2))
        sil = silhouette_gate(D, verd, strata, np.random.default_rng(SEED + 3))
        out_rows.append(dict(family=family, layer="value", variant="drop_D91_rows",
                             n=len(keep), dropped=dropped,
                             pseudo_F=round(float(f), 4), p_perm=round(float(p), 4),
                             anosim_R=round(float(r), 4), anosim_p=round(float(rp), 4),
                             sil=round(float(sil["sil"]), 4) if sil["sil"] == sil["sil"] else "",
                             sil_p=round(float(sil["sil_p"]), 4) if sil["sil_p"] == sil["sil_p"] else "",
                             sil_n=sil["n"]))
        print(f"{family}: n={len(keep)} (dropped {dropped}) | F={f:.3f} p={p:.3f} | "
              f"R={r:.3f} p={rp:.3f} | sil={sil['sil']:.3f} p={sil['sil_p']:.3f}")

    # frozen comparisons
    frozen = {}
    for name in ("exB_permanova.csv", "exB_anosim.csv", "exB_silhouette.csv"):
        with open(os.path.join(RESULTS, name), encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["family"] in ("F1_physical", "F6_aeration") and r["layer"] == "value":
                    frozen.setdefault(r["family"], {})[name] = r
    for r in out_rows:
        fr = frozen.get(r["family"], {})
        if fr:
            print(f"  frozen[{r['family']}]: F={fr['exB_permanova.csv']['pseudo_F']} "
                  f"p={fr['exB_permanova.csv']['p_perm']} | "
                  f"R={fr['exB_anosim.csv']['R']} p={fr['exB_anosim.csv']['p_perm']} | "
                  f"sil={fr['exB_silhouette.csv']['sil']} p={fr['exB_silhouette.csv']['sil_p']}")

    out = os.path.join(RESULTS, "exB_d91_sensitivity.csv")
    with open(out + ".tmp", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    os.replace(out + ".tmp", out)
    print(f"[d91] wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
