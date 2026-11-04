#!/usr/bin/env python3
"""verdict_multiclass.py — author-labeled multi-class verdict test (F1 physical, value layer).

Rebuilds the "13 multi-class papers / 57 rows" re-run as a persisted artifact
(provenance gap found in P8F: the numbers had been hardcoded into
make_tableS_corrections.py and the manuscript without a producer/CSV).

Method (identical machinery to experiment_B_stratified.py): F1-physical
value-complete rows, author-labeled stratum, restricted to papers carrying >= 2
distinct verdict classes among those rows; standardized Euclidean distances;
PERMANOVA + ANOSIM (pseudo-F / R, labels permuted within papers, 999
permutations) and silhouette width (1,000 shuffled-label draws). Seeds
42 / 44 / 45 as in the stratified suite.

NOTE (P8F, D173): the previously quoted values for this test
(F = 3.10, p = 0.375, R = 0.239, sil = 0.042) were found NOT reproducible from
the frozen layers under this documented procedure (and under all tested
variants); the persisted output supersedes them. The faithful subset is
57 rows / 13 papers / 6 verdict classes (the earlier "4 groups" was wrong).

Output: results/exB_multiclass.csv
Run: python3 code/verdict_multiclass.py
"""
import csv, os, sys
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from experiment_B_familymaps import (
    load_all, family_subset, FAMILIES, SEED,
    permanova_test, permute_anosim_p, silhouette_gate,
)


def main():
    rows, vals, fmap, params = load_all()
    dev_log = list(csv.DictReader(open(os.path.join(ROOT, "results/verdict_derivation_log.csv"), encoding="utf-8-sig")))
    dev_set = {(d["batch_id"], d["scenario"]) for d in dev_log}
    pres = list(csv.DictReader(open(os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv"), encoding="utf-8-sig")))

    # provenance map (identical construction to experiment_B_stratified.py)
    ord_to_prov = {}
    cur_b, ordn = "", 0
    for r in pres:
        ordn += 1
        b = (r["batch_id"] or "").strip()
        if b:
            cur_b = b
        if r.get("row_kind") != "data":
            continue
        scen = (r["scenario"] or "").strip()
        verd = (r["verdict"] or "").strip()
        if verd in ("", "TBD-StageB"):
            prov = "unlabeled"
        elif (cur_b, scen) in dev_set:
            prov = "rubric"
        else:
            prov = "author"
        ord_to_prov[ordn] = prov

    fam = "F1_physical"
    sub = family_subset(rows, vals, fmap, fam, FAMILIES[fam])
    vcomp = [s for s in sub if s["vcomplete"]]
    Xv = np.asarray([[s["values"][p] for p in FAMILIES[fam]] for s in vcomp], float)
    Zv = StandardScaler().fit_transform(Xv)
    Dv = squareform(pdist(Zv))
    verds = [s["row"]["verdict"] or "TBD-StageB" for s in vcomp]
    provs = [ord_to_prov[s["row"]["ordinal"]] for s in vcomp]
    strata = [s["row"]["batch"] for s in vcomp]

    # author stratum
    idx = [i for i, p in enumerate(provs) if p == "author" and verds[i] not in ("TBD-StageB", "")]
    # multi-class papers only: >= 2 distinct verdict classes among the author rows
    per_paper = {}
    for i in idx:
        per_paper.setdefault(strata[i], set()).add(verds[i])
    multi = {p for p, cs in per_paper.items() if len(cs) >= 2}
    idx = [i for i in idx if strata[i] in multi]

    n_sub = len(idx)
    n_papers = len({strata[i] for i in idx})
    n_grps = len({verds[i] for i in idx})
    assert (n_sub, n_papers, n_grps) == (57, 13, 6), \
        f"multi-class subset drifted: {(n_sub, n_papers, n_grps)} != (57, 13, 6)"
    print(f"multi-class subset: n={n_sub} rows, {n_papers} papers, {n_grps} classes")

    sub_Dv = Dv[np.ix_(idx, idx)]
    sub_v = [verds[i] for i in idx]
    sub_strat = [strata[i] for i in idx]

    f_v, p_v, _ = permanova_test(sub_Dv, sub_v, sub_strat, np.random.default_rng(SEED))
    r_v, rp_v = permute_anosim_p(sub_Dv, sub_v, sub_strat, np.random.default_rng(SEED + 2))
    sil_v = silhouette_gate(sub_Dv, sub_v, sub_strat, np.random.default_rng(SEED + 3))

    out = os.path.join(ROOT, "results", "exB_multiclass.csv")
    fields = ["family", "layer", "stratum", "n", "papers", "groups",
              "pseudo_F", "p_perm", "anosim_R", "p_anosim", "silhouette", "p_silhouette"]
    row = dict(family=fam, layer="value", stratum="author_multi_class",
               n=n_sub, papers=n_papers, groups=n_grps,
               pseudo_F=round(float(f_v), 4), p_perm=round(float(p_v), 4),
               anosim_R=round(float(r_v), 4), p_anosim=round(float(rp_v), 4),
               silhouette=round(float(sil_v.get("sil", np.nan)), 4),
               p_silhouette=round(float(sil_v.get("sil_p", np.nan)), 4))
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow(row)
    print(f"wrote {out}")
    print("values:", {k: row[k] for k in ("pseudo_F", "p_perm", "anosim_R", "p_anosim", "silhouette", "p_silhouette")})
    # The manuscript quotes these numbers (main §3.x and the Table S12a note):
    # F = 0.81 (p = 1.00), R = 0.16 (p = 0.73), silhouette = -0.03 (p = 0.40).
    print("quoted in manuscript: pseudo_F=0.81, p=1.00, R=0.16 (p=0.73), sil=-0.03 (p=0.40)")


if __name__ == "__main__":
    main()
