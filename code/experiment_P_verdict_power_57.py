#!/usr/bin/env python3
"""experiment_P_verdict_power_57.py — positive control on the 57-row multi-class subset.

Companion to `experiment_P_verdict_power.py`. The main null result is the
57-row / 13-paper author multi-class subset, not the full 421-row value layer.
This script rebuilds that exact subset (same construction as
`verdict_multiclass.py`), assigns balanced three-group labels at random, adds a
known shift to salinity in one group, and runs the same within-paper
PERMANOVA/ANOSIM tests.

Design: 100 replicates per delta, 99 within-paper permutations, deltas
0 / 0.25 / 0.5 / 1.0 SD. Output: results/exP_verdict_power_57.csv.

Run: python3 code/experiment_P_verdict_power_57.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from experiment_B_familymaps import (  # noqa: E402
    FAMILIES, family_subset, load_all, permanova_test, permute_anosim_p,
)

N_REP = 100
N_PERM = 99
DELTAS = [0.0, 0.25, 0.5, 1.0]
SEED = 42
OUT = os.path.join(ROOT, "results", "exP_verdict_power_57.csv")


def multi57():
    """Rebuild the 57-row / 13-paper author multi-class subset (verdict_multiclass.py)."""
    rows, vals, fmap, _params = load_all()
    dev_log = list(csv.DictReader(open(
        os.path.join(ROOT, "results/verdict_derivation_log.csv"), encoding="utf-8-sig")))
    dev_set = {(d["batch_id"], d["scenario"]) for d in dev_log}
    pres = list(csv.DictReader(open(
        os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv"),
        encoding="utf-8-sig")))
    ord_to_prov: dict[int, str] = {}
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
    Z = StandardScaler().fit_transform(Xv)
    verds = [s["row"]["verdict"] or "TBD-StageB" for s in vcomp]
    provs = [ord_to_prov[s["row"]["ordinal"]] for s in vcomp]
    strata = [s["row"]["batch"] for s in vcomp]
    idx = [i for i, p in enumerate(provs) if p == "author" and verds[i] not in ("TBD-StageB", "")]
    per_paper: dict[str, set] = {}
    for i in idx:
        per_paper.setdefault(strata[i], set()).add(verds[i])
    multi = {p for p, cs in per_paper.items() if len(cs) >= 2}
    idx = [i for i in idx if strata[i] in multi]
    assert (len(idx), len({strata[i] for i in idx})) == (57, 13), \
        f"multi-class subset drifted: {(len(idx), len({strata[i] for i in idx}))}"
    return Z[idx], np.asarray([strata[i] for i in idx])


def main() -> int:
    Z, papers = multi57()
    n = len(Z)
    base = np.array(["A"] * (n // 3) + ["B"] * (n // 3) + ["C"] * (n - 2 * (n // 3)))
    rows = []
    for delta in DELTAS:
        rng = np.random.default_rng(SEED + int(1000 * delta))
        rej_f = rej_r = 0
        fs = []
        for _ in range(N_REP):
            labels = rng.permutation(base)
            Zs = Z.copy()
            Zs[labels == "A", 2] += delta
            D = squareform(pdist(Zs))
            f, p, _ = permanova_test(D, labels, papers, rng, n_perm=N_PERM)
            _r, rp = permute_anosim_p(D, labels, papers, rng, n_perm=N_PERM)
            rej_f += int(p < 0.05)
            rej_r += int(rp < 0.05)
            fs.append(float(f))
        rows.append(dict(
            delta=delta, n_rep=N_REP, n_perm=N_PERM,
            perm_rej=round(rej_f / N_REP, 4),
            anosim_rej=round(rej_r / N_REP, 4),
            median_F=round(float(np.median(fs)), 4),
        ))
        print(f"[positive-control-57] delta={delta:.2f}  "
              f"PERMANOVA={rej_f/N_REP:.3f}  ANOSIM={rej_r/N_REP:.3f}  "
              f"median F={np.median(fs):.3f}")
    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, OUT)
    print(f"[positive-control-57] wrote {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
