#!/usr/bin/env python3
"""Fig 1 — paper-level parameter co-reporting heatmap (P1-6.1, deliverable 2/6).

Manuscript outline §3.1: "Fig 1: co-reporting + families".
Matrix M[i,j] = number of DATA papers (row_kind=='data') reporting both
parameter i and parameter j in at least one row (paper-level co-report).
Parameters ordered by frozen family groups (FAMILIES from
code/family_derivation.py, parsed via AST — no import side effects),
bridge params (DO/BOD/COD) shown once under their primary family,
unassigned params (16, per family_rules.md v1.1) last.

Armor (X1 lesson — no hand-written numbers):
  * asserts param count = 44 (hdr[7:51], C1 pack convention)
  * asserts 122 data papers (C1 pack)
  * asserts family sizes vs docs/family_rules.md (4/9/4/3/5/3/3)
  * asserts 16 unassigned params matching the rules-doc list
  * asserts matrix symmetry + diagonal dominance

Outputs (atomic tmp->replace): figures/fig1_coreport_families.png (600 dpi) + .pdf
Run: python3 code/figures/make_fig1_coreport.py
"""
import ast
import csv
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117)
import sys

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
PRESENCE = os.path.join(ROOT, "docs", "extraction", "stageA", "matrix_presence_v2.csv")
DERIV_A = os.path.join(ROOT, "code", "experiment_B_familymaps.py")
DERIV_B = os.path.join(ROOT, "code", "experiment_C2_stability.py")
RULES = os.path.join(ROOT, "docs", "family_rules.md")
OUT_PNG = os.path.join(ROOT, "figures", "fig1_coreport_families.png")
OUT_PDF = os.path.join(ROOT, "figures", "fig1_coreport_families.pdf")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch


def atomic_save(fig, path, **kw):
    base, ext = os.path.splitext(path)
    tmp = base + ".tmp" + ext
    fig.savefig(tmp, **kw)
    os.replace(tmp, path)


def load_families():
    """Parse the frozen FAMILIES dict from BOTH experiment scripts via AST and
    assert they are identical (drift check between duplicated dicts)."""
    out = []
    for path in (DERIV_A, DERIV_B):
        tree = ast.parse(open(path, encoding="utf-8").read())
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "FAMILIES":
                        found = ast.literal_eval(node.value)
        assert found is not None, f"FAMILIES dict not found in {os.path.basename(path)}"
        out.append(found)
    assert out[0] == out[1], "FAMILIES dicts diverge between experiment_B and experiment_C2"
    return out[0]


def short_name(param):
    return param.split(" (")[0]


def main():
    families = load_families()

    # --- rules-doc asserts (family sizes + unassigned list) ---
    rules = open(RULES, encoding="utf-8").read()
    expect_sizes = {"F1_physical": 4, "F2_metals": 9, "F3_nutrients": 4,
                    "F4_organic": 3, "F1_physicala": 5, "F1_clarity": 3,
                    "F6_aeration": 3}
    for fam, size in expect_sizes.items():
        assert len(families[fam]) == size, f"{fam} size {len(families[fam])} != {size}"
        assert fam.replace("_", " ") in rules or fam in rules, f"{fam} missing in rules doc"
    for tokens in ["Unassigned (16)", "bridge params"]:
        assert tokens in rules, f"rules-doc marker missing: {tokens}"

    # --- presence layer ---
    with open(PRESENCE, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    hdr = list(rows[0].keys())
    params = hdr[7:51]
    assert len(params) == 44, f"param count {len(params)} != 44"
    data_rows = [r for r in rows if r["row_kind"] == "data"]
    # NOTE (R5-1): batch_id is a merged-cell column — 474 of 764 data rows are
    # continuation rows with a BLANK batch_id, and for one paper the id appears
    # only on a preceding non-data row. Ownership must therefore be forward-filled
    # over ALL 867 rows in file order (exactly as experiment_C1.load() does) and
    # only then restricted to data rows. Filling over data rows alone loses that
    # paper; keying on the raw batch_id instead collapses 474 continuation rows
    # into a phantom "" paper holding 42 of 44 params, which spuriously satisfies
    # `len(papers) == 122` (121 real + 1 phantom) and under-counts three of the
    # top-five co-reporting pairs.
    papers = {}
    own = ""
    for r in rows:
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        if r["row_kind"] != "data":
            continue
        papers.setdefault(own, set())
        for p in params:
            if r[p] == "Y":
                papers[own].add(p)
    assert "" not in papers, (
        "unattributed continuation rows: batch_id must be forward-filled "
        "over all rows in file order (R5-1 provenance fix)")
    assert len(papers) == 122, f"data papers {len(papers)} != 122"
    assert len(data_rows) == 764, f"data rows {len(data_rows)} != 764"

    # --- display order ---
    display_order = ["F1_physical", "F3_nutrients", "F4_organic", "F2_metals",
                     "F1_physicala", "F1_clarity", "F6_aeration"]
    assigned, primary = [], {}
    for fam in display_order:
        for p in families[fam]:
            if p not in primary:
                primary[p] = fam
                assigned.append(p)
    unassigned = [p for p in params if p not in primary]
    assert len(unassigned) == 16, f"unassigned {len(unassigned)} != 16"
    unassigned.sort(key=lambda p: (-len([k for k in papers if p in papers[k]]), p))
    order = assigned + unassigned
    assert set(order) == set(params) and len(order) == 44

    # --- co-report matrix ---
    idx = {p: i for i, p in enumerate(order)}
    M = np.zeros((44, 44), dtype=int)
    for k, pset in papers.items():
        plist = sorted(pset, key=lambda p: idx[p])
        for a_i, a in enumerate(plist):
            M[idx[a], idx[a]] += 1
            for b in plist[a_i + 1:]:
                M[idx[a], idx[b]] += 1
                M[idx[b], idx[a]] += 1
    assert (M == M.T).all(), "matrix not symmetric"
    assert (M.diagonal() >= M.max(axis=1)).all(), "diagonal dominance violated"

    # --- figure ---
    fam_colors = {"F1_physical": "#1f77b4", "F3_nutrients": "#2ca02c",
                  "F4_organic": "#9467bd", "F2_metals": "#d62728",
                  "F1_physicala": "#17becf", "F1_clarity": "#8c564b",
                  "F6_aeration": "#ff7f0e", "unassigned": "#b0bec5"}
    fam_tags = {"F1_physical": "F1 phys", "F3_nutrients": "F3 nutr",
                "F4_organic": "F4 org", "F2_metals": "F2 met",
                "F1_physicala": "F1a", "F1_clarity": "F1 clar",
                "F6_aeration": "F6 aer", "unassigned": "unassigned"}
    band = [fam_colors[primary.get(p, "unassigned")] for p in order]

    cmap = LinearSegmentedColormap.from_list(
        "coreport", ["#fafafa", "#c6dbef", "#6baed6", "#2171b5", "#08306b"])
    M_disp = M
    vmax = float(M.max())

    fig = plt.figure(figsize=(6.6, 5.8))
    gs = fig.add_gridspec(1, 1, left=0.22, right=0.84, top=0.90, bottom=0.22)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(M_disp, cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")

    # family boundary lines
    bounds, run = [], 1
    for i in range(1, 45):
        if i < 44 and primary.get(order[i], "unassigned") == primary.get(order[i - 1], "unassigned"):
            run += 1
        else:
            bounds.append(run)
            run = 1
    edges = np.cumsum(bounds)[:-1] - 0.5
    for e in edges:
        ax.axhline(e, color="white", lw=2.0)
        ax.axvline(e, color="white", lw=2.0)

    # family band strips (top + left) + tags on top strip
    bounds_cum = np.cumsum(bounds)
    start = 0
    for n_cells, b in zip(bounds, bounds_cum):
        fam = primary.get(order[start], "unassigned")
        c = fam_colors[fam]
        ax.axhline(-1.6, xmin=0, color="none")  # noop keep spacing stable
        for e_i in range(start, b):
            ax.add_patch(plt.Rectangle((e_i - 0.5, -2.6), 1, 1.4, fc=c, ec="none",
                                       clip_on=False))
        ax.text((start + b - 1) / 2, -3.3, fam_tags[fam], ha="center", va="bottom",
                fontsize=6.5, rotation=38 if n_cells < 5 else 0,
                color=c if fam != "unassigned" else "#546e7a",
                fontweight="bold" if fam != "unassigned" else "normal")
        start = b

    labels = [short_name(p) for p in order]
    ax.set_xticks(range(44))
    ax.set_yticks(range(44))
    ax.set_xticklabels(labels, rotation=90, fontsize=6.2)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    cax = fig.add_axes([0.865, 0.22, 0.022, 0.62])
    cbar = fig.colorbar(ax.images[0], cax=cax)
    cbar.set_label("# papers reporting both parameters", fontsize=7.6,
                   rotation=270, labelpad=9)
    cbar.ax.tick_params(labelsize=7.0)

    atomic_save(fig, OUT_PNG, dpi=600, bbox_inches="tight", pad_inches=0.03)
    atomic_save(fig, OUT_PDF, bbox_inches="tight", pad_inches=0.03)
    # R5-1 armour: emit the top co-reporting pairs as a machine-readable layer so
    # code/check_numbers.py can assert the manuscript quotes match (X1 lesson —
    # no hand-written numbers). §3.1 quotes exactly these five values.
    top_pairs = sorted(((order[i], order[j], M[i, j])
                        for i in range(44) for j in range(i + 1, 44)),
                       key=lambda t: (-t[2], t[0], t[1]))[:10]
    coreport_csv = os.path.join(ROOT, "results", "fig1_coreport_pairs.csv")
    tmp = coreport_csv + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "param_a", "param_b", "papers_co_reporting"])
        for rank, (a, b, n) in enumerate(top_pairs, 1):
            w.writerow([rank, short_name(a), short_name(b), int(n)])
    os.replace(tmp, coreport_csv)
    print("[fig1] params 44 · data-papers 122 · rows 764 · unassigned 16 · "
          f"family params {len(assigned)}")
    print("[fig1] top co-reported pairs:", ", ".join(f"{a}+{b}={n}" for a, b, n in top_pairs[:5]))
    print(f"[fig1] wrote {os.path.relpath(OUT_PNG, ROOT)} + .pdf")
    print(f"[fig1] wrote {os.path.relpath(coreport_csv, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
