#!/usr/bin/env python3
"""Manuscript table generators (P1-6.1 batch, D115) — X1 rule: numbers never hand-written.

  table1_matrix.md   — matrix formalization (outline §2.2: layers, NR/NE vocab, density ρ)
  table2_families.md — frozen family derivation (outline §2.3; numbers cross-asserted
                       against docs/family_rules.md parse + baseline families)
  (table3_c2.md is written by make_tableS_corrections.py — sole writer, D194)

Outputs: manuscript/tables/*.md (atomic tmp->replace). Read-only on data layers.
Run: python3 code/make_tables.py
"""
from __future__ import annotations
import ast
import csv
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "manuscript", "tables")
PRESENCE = os.path.join(ROOT, "docs", "extraction", "stageA", "matrix_presence_v2.csv")
VALUES = os.path.join(ROOT, "docs", "extraction", "stageB", "matrix_values_v2.csv")
FAMILY_MAP = os.path.join(ROOT, "docs", "family_map.csv")
RULES = os.path.join(ROOT, "docs", "family_rules.md")
DERIV_A = os.path.join(ROOT, "code", "experiment_B_familymaps.py")
DERIV_B = os.path.join(ROOT, "code", "experiment_C2_stability.py")


def paper_owner(pres: list[dict]) -> list[str]:
    """Forward-fill the merged-cell batch_id column over ALL rows in file order.

    R5-1: batch_id is a merged-cell column; 474 of 764 data rows are continuation
    rows with a BLANK batch_id, and for one paper the id appears only on a
    preceding non-data row. Keying on the raw batch_id therefore drops that paper
    and silts the continuation rows into a phantom "" paper, which spuriously
    satisfies the `len(papers) == 122` assert (121 real + 1 phantom) and
    under-counts three of the top-five paper-level co-reporting pairs.
    """
    out, own = [], ""
    for r in pres:
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        out.append(own)
    return out


def atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def load_families() -> dict:
    out = []
    for p in (DERIV_A, DERIV_B):
        tree = ast.parse(open(p, encoding="utf-8").read())
        found = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "FAMILIES":
                        found = ast.literal_eval(node.value)
        assert found is not None, f"FAMILIES not found in {p}"
        out.append(found)
    assert out[0] == out[1], "FAMILIES dicts diverge"
    return out[0]


def table1() -> str:
    with open(PRESENCE, encoding="utf-8-sig") as fh:
        pres = list(csv.DictReader(fh))
    params = list(pres[0].keys())[7:51]
    kinds = Counter(r["row_kind"] for r in pres)
    owner = paper_owner(pres)
    vocab = Counter()
    for r in pres:
        for p in params:
            vocab[(r[p] or "blank").strip() or "blank"] += 1
    base = json.load(open(os.path.join(ROOT, "results", "phase4_baseline.json")))
    b = base["baselines"]
    papers = len({o for o, r in zip(owner, pres) if r["row_kind"] == "data" and o})
    data_blank = sum(1 for o, r in zip(owner, pres) if r["row_kind"] == "data" and not o)
    assert len(params) == 44 and len(pres) == 867 and papers == 122
    assert data_blank == 0, f"{data_blank} unattributed continuation rows (R5-1)"
    assert b["value_cells"] == 3216 and b["filled_cells"] == 2658
    assert (b["value_records"], b["censored_values"], b["documented_blanks"]) == (2705, 7, 551)
    lines = [
        "# Table 1 — Matrix formalization",
        "",
        "| property | layer | value |",
        "|---|---|---|",
        f"| papers in corpus | — | {b['n_papers']} |",
        f"| parameters | — | {len(params)} |",
        f"| total rows | presence | {len(pres)} "
        f"(data {kinds['data']} · padding {kinds['padding']} · placeholder {kinds['placeholder']} · context {kinds['context']}) |",
        f"| papers with data rows | presence | {papers} |",
        f"| presence cell vocabulary | presence | Y ({vocab['Y']}) · NE ({vocab['NE']}) · blank/NR ({vocab['blank']}) |",
        f"| value-layer cells | value | {b['value_cells']} "
        f"(numeric {b['filled_cells']} · censored {b['censored_values']} · blank {b['documented_blanks']}; "
        f"{b['value_records']} records) |",
        f"| logged unit conversions | value | {b['conversions']} |",
        f"| QA flags (all triaged) | value | {b['qa_flags']} |",
        f"| semantic statuses labeled | verdict | {b['statuses_labeled']} "
        f"({b['paper_labeled']} paper-labeled + {b['derived']} rubric-derived) |",
        "| row density ρ | presence | mean of a row's presence flags across the 44 parameters (0–1) |",
        "| sparsity drift ratio | both | SDR = median LOPO drift (ρ ≤ Q1) / median LOPO drift (ρ ≥ Q3) |",
        "",
        "_NR = parameter not reported in the source row; NE = explicitly examined but not "
        "found or above detection in the source study; field definitions are given in the "
        "released data dictionary._",
        "",
    ]
    return "\n".join(lines)


def table2() -> str:
    families = load_families()
    rules = open(RULES, encoding="utf-8").read()
    with open(PRESENCE, encoding="utf-8-sig") as fh:
        pres = list(csv.DictReader(fh))
    owner = paper_owner(pres)
    pres_params = list(pres[0].keys())[7:51]
    fmap = list(csv.DictReader(open(FAMILY_MAP, encoding="utf-8")))
    base = json.load(open(os.path.join(ROOT, "results", "phase4_baseline.json")))
    fam_row_cols = {f"full_{k}": k for k in families}
    display = ["F1_physical", "F3_nutrients", "F4_organic", "F2_metals",
               "F1_physicala", "F1_clarity", "F6_aeration"]
    lines = [
        "# Table 2 — Parameter families",
        "",
        "| family | kind | parameters | cover rows | cover papers | rows (full) | papers |",
        "|---|---|---|---|---|---|---|",
    ]
    for fam in display:
        col = f"full_{fam}"
        rows_full = sum(1 for r in fmap if r[col] == "1" and r["row_kind"] == "data")
        papers = len({r["batch_id_filled"] for r in fmap
                      if r[col] == "1" and r["row_kind"] == "data"})
        params = families[fam]
        # cross-asserts vs family_rules.md + baseline
        m = re.search(rf"\|\s\*\*{re.escape(fam.replace('_', ' '))}\*\*\s\|\s(\d+)\s\|\s(\d+)\s\|\s(\d+)\s\|", rules)
        if not m:
            m = re.search(rf"\|\s\*\*{re.escape(fam)}\*\*\s\|\s(\d+)\s\|\s(\d+)\s\|\s(\d+)\s\|", rules)
        assert m, f"rules-doc row not found: {fam}"
        size_doc, rows_doc, papers_doc = (int(g) for g in m.groups())
        assert size_doc == len(params), f"{fam}: size {size_doc} != {len(params)}"
        assert rows_doc == rows_full, f"{fam}: rules rows {rows_doc} != computed {rows_full}"
        assert papers_doc == papers, f"{fam}: rules papers {papers_doc} != computed {papers}"
        assert base["baselines"]["families"][fam] == rows_full, f"{fam}: baseline drift"
        # cover rows / cover papers (presence analyses; Table S5 and Figs. S2-S12)
        cover = [1 for o, r in zip(owner, pres)
                 if r["row_kind"] == "data" and o
                 and any((r[q] or "").strip() in ("Y", "NE") for q in params)]
        cover_papers = len({o for o, r in zip(owner, pres)
                            if r["row_kind"] == "data" and o
                            and any((r[q] or "").strip() in ("Y", "NE") for q in params)})
        assert len(cover) >= rows_full, f"{fam}: cover {len(cover)} < full {rows_full}"
        kind_m = re.search(
            rf"\|\s\*\*{re.escape(fam.replace('_', ' '))}\*\*\s\|\s\d+\s\|\s\d+\s\|\s\d+\s\|\s([^|]+?)\s\|",
            rules)
        kind = kind_m.group(1) if kind_m else "?"
        KIND_SHORT = {
            "clustering-derived (restructured: NO3/PO4 moved to F3)": "Derived (restructured)",
            "clustering-derived (pruned from k=4 cluster 3)": "Derived (pruned)",
            "amendment (rule-4 v1-continuity)": "Amended (continuity)",
            "amendment (domain-lead semantic)": "Amended (semantic)",
            "amendment (discovery: exhaustive search + binding matrix)": "Amended (exhaustive search)",
            "clustering-derived": "Derived",
        }
        kind = KIND_SHORT.get(kind, kind)
        lines.append(f"| {fam.replace('_', ' ')} | {kind} | "
                     f"{', '.join(p.split(' (')[0] for p in params)} | "
                     f"{len(cover)} | {cover_papers} | {rows_full} | {papers} |")
    lines += [
        "",
        "",
        "",
    ]
    return "\n".join(lines)


def table3() -> str:
    recs = list(csv.DictReader(open(os.path.join(ROOT, "results", "exG_absratio.csv"), encoding="utf-8")))
    recs.sort(key=lambda r: float(r["ratio_raw"]), reverse=True)
    fam_display = {
        "F1_physical": "F1 physical",
        "F1_physicala": "F1a physical--anion",
        "F1_clarity": "F1 clarity",
        "F2_metals": "F2 metals",
        "F3_nutrients": "F3 nutrients",
        "F4_organic": "F4 organic",
        "F6_aeration": "F6 aeration",
    }
    def sci(v):
        if not v or float(v) == 0:
            return "0"
        mant, exp = f"{float(v):.2e}".split("e")
        return f"${mant} \\times 10^{{{int(exp)}}}$"

    lines = [
        "# Table 3 — Absolute Procrustes drift reduction across co-reporting families (presence PCA, 2-D)",
        "",
        "| Family | Events | All $\\delta$ | Family $\\delta$ | Raw (CI) | Paired (CI) | Floor status |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in recs:
        fam_name = fam_display.get(r["family"], r["family"].replace("_", " "))
        raw_ci = f"({int(round(float(r['ratio_raw_ci_lo']))):d}, {int(round(float(r['ratio_raw_ci_hi']))):d})"
        paired_ci = f"({int(round(float(r['median_paired_ratio_ci_lo']))):d}, {int(round(float(r['median_paired_ratio_ci_hi']))):d})"
        if r["family"] == "F6_aeration":
            status = "Median at floor"
        elif r["at_floor"] == "True" or r["at_floor"] is True:
            status = "At floor ($\\tau = 10^{-6}$)"
        elif int(r["fam_sparse_eq_dense_events"]) > 0:
            status = f"$\\mathrm{{SDR}}_{{\\mathrm{{fam}}}}=1$ ({r['fam_sparse_eq_dense_events']}/{r['n_events']})"
        else:
            status = "Above floor"
        lines.append(
            f"| {fam_name} | {r['n_events']} | {sci(r['all_med'])} | {sci(r['fam_med'])} | "
            f"{float(r['ratio_raw']):.0f} {raw_ci} | {float(r['median_paired_ratio']):.0f} {paired_ci} | {status} |"
        )
    lines += [
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    os.makedirs(OUTDIR, exist_ok=True)
    out = {
        "table1_matrix.md": table1(),
        "table2_families.md": table2(),
    }
    table3()  # asserts only: table3_c2.md is written by make_tableS_corrections.py (sole writer, D194)
    for name, text in out.items():
        atomic_write(os.path.join(OUTDIR, name), text)
        print(f"[tables] wrote manuscript/tables/{name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
