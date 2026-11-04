#!/usr/bin/env python3
"""experiment_B_stratified.py — Provenance-stratified verdict tests and LOPO sensitivity (M4/M11/M12).

Quantifies:
1. Complete verdict provenance breakdown (Author vs Rubric across all 8 classes).
2. Class separation tests (PERMANOVA, ANOSIM, Silhouette) stratified by label provenance:
   - Combined (all labeled)
   - Author-only
   - Rubric-only
3. LOPO paper-omission sensitivity for F3 nutrients value layer.
4. Gate audit: rows vs papers for each family configuration.

Outputs:
- results/exB_stratified.csv
- results/exB_f3_lopo_sensitivity.csv
- results/exB_verdict_provenance.csv
"""
import csv, os, sys
from collections import Counter, defaultdict
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))

from experiment_B_familymaps import (
    load_all, family_subset, FAMILIES, SEED, PERM_N, SIL_N,
    permanova_test, permute_anosim_p, silhouette_gate,
    jac_dist_family, residualize, rho_of_row
)

def run():
    rows, vals, fmap, params = load_all()
    dev_log = list(csv.DictReader(open(os.path.join(ROOT, "results/verdict_derivation_log.csv"), encoding="utf-8-sig")))
    dev_set = {(d["batch_id"], d["scenario"]) for d in dev_log}
    pres = list(csv.DictReader(open(os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv"), encoding="utf-8-sig")))

    # 1. Provenance map
    ord_to_prov = {}
    cur_b, ordn = "", 0
    class_counter = {"author": Counter(), "rubric": Counter(), "unlabeled": Counter()}
    low_conf_counter = Counter()
    for d in dev_log:
        if d.get("low_confidence") == "TRUE":
            low_conf_counter[d["verdict"]] += 1

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
            c = verd if verd else "Blank"
        elif (cur_b, scen) in dev_set:
            prov = "rubric"
            c = verd
        else:
            prov = "author"
            c = verd
        ord_to_prov[ordn] = (cur_b, scen, verd, prov)
        class_counter[prov][c] += 1

    # Save verdict provenance table with low-confidence counts (M12)
    prov_out = os.path.join(ROOT, "results/exB_verdict_provenance.csv")
    with open(prov_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["verdict_class", "author_rows", "rubric_rows", "low_conf_rows", "total_labeled", "unlabeled_rows", "total_rows"])
        all_classes = sorted(set(list(class_counter["author"].keys()) +
                                 list(class_counter["rubric"].keys()) +
                                 list(class_counter["unlabeled"].keys())))
        for c in all_classes:
            a = class_counter["author"][c]
            ru = class_counter["rubric"][c]
            lc = low_conf_counter.get(c, 0)
            un = class_counter["unlabeled"][c]
            w.writerow([c, a, ru, lc, a + ru, un, a + ru + un])
    print(f"wrote {prov_out}")

    # 2. Stratified tests
    strat_rows = []
    for fam, fparams in FAMILIES.items():
        sub = family_subset(rows, vals, fmap, fam, fparams)
        cov = [s for s in sub if s["cov"]]
        vcomp = [s for s in sub if s["vcomplete"]]

        # Presence
        if len(cov) >= 6:
            X = np.asarray([s["X_fam"] for s in cov], float)
            rho = np.asarray([rho_of_row(s["row"]) for s in cov])
            D = jac_dist_family(X)
            strata = [s["row"]["batch"] for s in cov]
            all_verds = [s["row"]["verdict"] or "TBD-StageB" for s in cov]
            provs = [ord_to_prov[s["row"]["ordinal"]][3] for s in cov]

            pca = PCA(n_components=2, random_state=SEED).fit(X)
            S = pca.transform(X)
            S_res = residualize(S, rho)
            D_res = squareform(pdist(S_res))

            for strat_name, mask in [
                ("all", [p in ("author", "rubric") for p in provs]),
                ("author", [p == "author" for p in provs]),
                ("rubric", [p == "rubric" for p in provs]),
            ]:
                idx = [i for i, m in enumerate(mask) if m and all_verds[i] not in ("TBD-StageB", "")]
                n_sub = len(idx)
                n_papers = len(set(strata[i] for i in idx))
                n_grps = len(set(all_verds[i] for i in idx))
                if n_sub < 10 or n_papers < 2 or n_grps < 2:
                    strat_rows.append(dict(
                        family=fam, layer="presence", stratum=strat_name,
                        n=n_sub, papers=n_papers, groups=n_grps,
                        pseudo_F=np.nan, p_perm=np.nan,
                        pseudo_F_res=np.nan, p_perm_res=np.nan,
                        anosim_R=np.nan, p_anosim=np.nan,
                        silhouette=np.nan, p_silhouette=np.nan
                    ))
                    continue
                sub_D = D[np.ix_(idx, idx)]
                sub_Dres = D_res[np.ix_(idx, idx)]
                sub_v = [all_verds[i] for i in idx]
                sub_strat = [strata[i] for i in idx]

                f_raw, p_raw, _ = permanova_test(sub_D, sub_v, sub_strat, np.random.default_rng(SEED))
                f_res, p_res, _ = permanova_test(sub_Dres, sub_v, sub_strat, np.random.default_rng(SEED))
                r_obs, r_p = permute_anosim_p(sub_D, sub_v, sub_strat, np.random.default_rng(SEED + 2))
                sil = silhouette_gate(sub_D, sub_v, sub_strat, np.random.default_rng(SEED + 3))

                strat_rows.append(dict(
                    family=fam, layer="presence", stratum=strat_name,
                    n=n_sub, papers=n_papers, groups=n_grps,
                    pseudo_F=round(float(f_raw), 4), p_perm=round(float(p_raw), 4),
                    pseudo_F_res=round(float(f_res), 4), p_perm_res=round(float(p_res), 4),
                    anosim_R=round(float(r_obs), 4), p_anosim=round(float(r_p), 4),
                    silhouette=round(float(sil.get("sil", np.nan)), 4),
                    p_silhouette=round(float(sil.get("sil_p", np.nan)), 4)
                ))

        # Value
        if len(vcomp) >= 10:
            Xv = np.asarray([[s["values"][p] for p in fparams] for s in vcomp], float)
            strat_v = [s["row"]["batch"] for s in vcomp]
            all_verds_v = [s["row"]["verdict"] or "TBD-StageB" for s in vcomp]
            provs_v = [ord_to_prov[s["row"]["ordinal"]][3] for s in vcomp]
            Zv = StandardScaler().fit_transform(Xv)
            Dv = squareform(pdist(Zv))

            for strat_name, mask in [
                ("all", [p in ("author", "rubric") for p in provs_v]),
                ("author", [p == "author" for p in provs_v]),
                ("rubric", [p == "rubric" for p in provs_v]),
            ]:
                idx = [i for i, m in enumerate(mask) if m and all_verds_v[i] not in ("TBD-StageB", "")]
                n_sub = len(idx)
                n_papers = len(set(strat_v[i] for i in idx))
                n_grps = len(set(all_verds_v[i] for i in idx))
                if n_sub < 6 or n_papers < 2 or n_grps < 2:
                    strat_rows.append(dict(
                        family=fam, layer="value", stratum=strat_name,
                        n=n_sub, papers=n_papers, groups=n_grps,
                        pseudo_F=np.nan, p_perm=np.nan,
                        pseudo_F_res=np.nan, p_perm_res=np.nan,
                        anosim_R=np.nan, p_anosim=np.nan,
                        silhouette=np.nan, p_silhouette=np.nan
                    ))
                    continue
                sub_Dv = Dv[np.ix_(idx, idx)]
                sub_v = [all_verds_v[i] for i in idx]
                sub_strat = [strat_v[i] for i in idx]

                f_v, p_v, _ = permanova_test(sub_Dv, sub_v, sub_strat, np.random.default_rng(SEED))
                r_v, rp_v = permute_anosim_p(sub_Dv, sub_v, sub_strat, np.random.default_rng(SEED + 2))
                sil_v = silhouette_gate(sub_Dv, sub_v, sub_strat, np.random.default_rng(SEED + 3))

                strat_rows.append(dict(
                    family=fam, layer="value", stratum=strat_name,
                    n=n_sub, papers=n_papers, groups=n_grps,
                    pseudo_F=round(float(f_v), 4), p_perm=round(float(p_v), 4),
                    pseudo_F_res=np.nan, p_perm_res=np.nan,
                    anosim_R=round(float(r_v), 4), p_anosim=round(float(rp_v), 4),
                    silhouette=round(float(sil_v.get("sil", np.nan)), 4),
                    p_silhouette=round(float(sil_v.get("sil_p", np.nan)), 4)
                ))

    # Save stratified CSV
    strat_out = os.path.join(ROOT, "results/exB_stratified.csv")
    with open(strat_out, "w", newline="", encoding="utf-8") as f:
        fields = ["family", "layer", "stratum", "n", "papers", "groups",
                  "pseudo_F", "p_perm", "pseudo_F_res", "p_perm_res",
                  "anosim_R", "p_anosim", "silhouette", "p_silhouette"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(strat_rows)
    print(f"wrote {strat_out}")

    # 3. F3 nutrients LOPO sensitivity
    sub_f3 = family_subset(rows, vals, fmap, "F3_nutrients", FAMILIES["F3_nutrients"])
    vcomp_f3 = [s for s in sub_f3 if s["vcomplete"]]
    labeled_f3 = [s for s in vcomp_f3 if (s["row"]["verdict"] or "").strip() not in ("", "TBD-StageB")]
    papers_f3 = sorted(set(s["row"]["batch"] for s in labeled_f3))

    lopo_rows = []
    # baseline full
    X_full = np.asarray([[s["values"][p] for p in FAMILIES["F3_nutrients"]] for s in labeled_f3], float)
    Z_full = StandardScaler().fit_transform(X_full)
    D_full = squareform(pdist(Z_full))
    v_full = [s["row"]["verdict"] for s in labeled_f3]
    s_full = [s["row"]["batch"] for s in labeled_f3]
    f_b, p_b, _ = permanova_test(D_full, v_full, s_full, np.random.default_rng(SEED))
    r_b, rp_b = permute_anosim_p(D_full, v_full, s_full, np.random.default_rng(SEED + 2))
    sil_b = silhouette_gate(D_full, v_full, s_full, np.random.default_rng(SEED + 3))
    lopo_rows.append(dict(omitted_paper="none", n=len(labeled_f3), papers=len(papers_f3),
                          pseudo_F=round(float(f_b), 4), p_perm=round(float(p_b), 4),
                          anosim_R=round(float(r_b), 4), p_anosim=round(float(rp_b), 4),
                          silhouette=round(float(sil_b.get("sil", np.nan)), 4),
                          p_silhouette=round(float(sil_b.get("sil_p", np.nan)), 4)))

    for p in papers_f3:
        keep = [s for s in labeled_f3 if s["row"]["batch"] != p]
        Xv = np.asarray([[s["values"][param] for param in FAMILIES["F3_nutrients"]] for s in keep], float)
        verd = [s["row"]["verdict"] for s in keep]
        strat = [s["row"]["batch"] for s in keep]
        Zv = StandardScaler().fit_transform(Xv)
        Dv = squareform(pdist(Zv))
        f_v, p_v, _ = permanova_test(Dv, verd, strat, np.random.default_rng(SEED), n_perm=999)
        r_v, rp_v = permute_anosim_p(Dv, verd, strat, np.random.default_rng(SEED + 2), n_perm=999)
        sil_v = silhouette_gate(Dv, verd, strat, np.random.default_rng(SEED + 3))
        lopo_rows.append(dict(
            omitted_paper=p, n=len(keep), papers=len(set(strat)),
            pseudo_F=round(float(f_v), 4), p_perm=round(float(p_v), 4),
            anosim_R=round(float(r_v), 4), p_anosim=round(float(rp_v), 4),
            silhouette=round(float(sil_v.get("sil", np.nan)), 4),
            p_silhouette=round(float(sil_v.get("sil_p", np.nan)), 4)
        ))

    lopo_out = os.path.join(ROOT, "results/exB_f3_lopo_sensitivity.csv")
    with open(lopo_out, "w", newline="", encoding="utf-8") as f:
        fields = ["omitted_paper", "n", "papers", "pseudo_F", "p_perm", "anosim_R", "p_anosim", "silhouette", "p_silhouette"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(lopo_rows)
    print(f"wrote {lopo_out}")

if __name__ == "__main__":
    run()
