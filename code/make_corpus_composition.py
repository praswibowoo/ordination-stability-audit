#!/usr/bin/env python3
"""make_corpus_composition.py — Table 1 (and §2.2) corpus composition, computed.

Source of truth: the RELEASED layers only (presence, values, verdicts,
family_map, corpus_manifest), so the numbers in the manuscript's Table 1 and
in the corpus paragraph of §2.4 are recomputed from the shipped data rather
than restated.

Definitions follow the manuscript:
  * presence layer  : a cell is Y (reported) or NE (examined, not found); both
    count as examined, so density and configuration patterns are binary.
  * configuration   : the binary (Y/NE) pattern across the 44 parameters.
  * cover row       : reports at least one parameter of at least one family.
  * complete row    : reports every parameter of at least one family.
  * labeled row    : carries a verdict class; TBD-StageB is a placeholder,
    not a label, and an empty verdict is unlabeled.
  * author label    : a verdict supplied by the study's authors (route
    author); rubric labels are derived from harmonized values.

Outputs: results/corpus_composition.csv (metric,value,unit,definition)

Run: python3 code/make_corpus_composition.py
"""
import csv
import os
import statistics
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(ROOT, "release", "p1_family_pca_data_v1.0")
RESULTS = os.path.join(ROOT, "results")

META = {"batch_id", "paper_doi", "scenario", "scenario_type", "verdict",
        "cause_effect_note", "source_table_ref", "Other reported parameters",
        "row_kind"}

# family parameter membership (docs/family_rules.md); bridges DO/BOD/COD are
# shared by two families.
FAMILIES = {
    "F1 physical": ["DO (mg/L)", "pH", "Salinity (ppt/‰)", "Temperature (°C)"],
    "F2 metals": ["As (µg/L)", "Cd (µg/L)", "Cr (µg/L)", "Cu (µg/L)", "Fe (µg/L)",
                  "Ni (µg/L)", "Pb (µg/L)", "Zn (µg/L)", "Co (µg/L)"],
    "F3 nutrients": ["NO2 (mg/L)", "NO3 (mg/L)", "NH4 (mg/L)", "PO4 (mg/L)"],
    "F4 organic": ["COD (mg/L)", "BOD (mg/L)", "Total Coliform (CFU/100mL)"],
    "F1 physicala": ["TDS", "EC (µS/cm)", "Cl- (ppt)", "Hardness (mg/L)",
                     "Alkalinity (mg/L)"],
    "F1 clarity": ["Turbidity (NTU)", "Secchi (m)", "TSS (mg/L)"],
    "F6 aeration": ["DO (mg/L)", "BOD (mg/L)", "COD (mg/L)"],
}
UNASSIGNED = ["H2S (µg/L)", "NH3 (µg/L)", "TN (mg/L)", "TOC (mg/L)", "ORP (mV)",
              "SAR", "Hg (µg/L)", "Mn (µg/L)", "Fecal Coliform (CFU/100mL)",
              "Oil & Grease (mg/L)", "Chlorophyll-a (µg/m3)", "Si (µmol/L)",
              "TP (mg/L)", "DOC (mg/L)", "Al (µg/L)", "E. coli (CFU/100mL)"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def main():
    presence = read_csv(os.path.join(REL, "presence.csv"))
    values = read_csv(os.path.join(REL, "values.csv"))
    verdicts = read_csv(os.path.join(REL, "verdicts.csv"))
    fmap = read_csv(os.path.join(REL, "family_map.csv"))
    manifest = read_csv(os.path.join(REL, "corpus_manifest.csv"))
    trace = read_csv(os.path.join(REL, "verification_trace.csv"))

    pcols = [c for c in presence[0] if c not in META]
    assert len(pcols) == 44, len(pcols)
    fam_params = sorted({p for ps in FAMILIES.values() for p in ps})

    def examined(row, col):
        return row[col].strip() in ("Y", "NE")

    data = [r for r in presence if r["row_kind"] == "data"]
    kinds = Counter(r["row_kind"] for r in presence)

    y = ne = blank = 0
    for r in presence:
        for c in pcols:
            v = r[c].strip()
            if v == "Y":
                y += 1
            elif v == "NE":
                ne += 1
            else:
                blank += 1

    # verdict layer over the 764 data rows
    lab = [r for r in data if r["verdict"].strip() and r["verdict"].strip() != "TBD-StageB"]
    unl = [r for r in data if not r["verdict"].strip() or r["verdict"].strip() == "TBD-StageB"]
    rubric = [r for r in verdicts]
    rubric_counts = Counter(r["verdict"] for r in rubric)
    low_conf = sum(1 for r in rubric if r["low_confidence"].strip().upper() == "TRUE")
    author_labels = sum(1 for r in presence
                        if r["verdict"].strip() and r["verdict"].strip() != "TBD-StageB"
                        and r["batch_id"].strip() and r["cause_effect_note"].strip()
                        and "author" in r["cause_effect_note"].lower())
    insuff = [r for r in presence if "insufficient" in (r["verdict"] + r["cause_effect_note"]).lower()]

    # configurations (binary presence pattern) and row density
    pat = Counter(tuple(1 if examined(r, c) else 0 for c in pcols) for r in data)
    dens = [sum(1 for c in pcols if examined(r, c)) / len(pcols) for r in data]
    q = statistics.quantiles(dens, n=4)

    # family cover / completeness
    cover = sum(1 for r in data if any(examined(r, p) for p in fam_params))
    fdata = [r for r in fmap if r["row_kind"] == "data"]
    complete = sum(1 for r in fdata if int(r["n_families_complete"]) >= 1)

    # value-layer accounting: records vs numeric values vs censored vs blanks.
    # A "record" is a row in values.csv; only is_real_value=TRUE with a value
    # is a numeric observation. The manuscript must not call all records "filled".
    n_numeric = sum(1 for r in values
                    if r["is_real_value"].strip().upper() == "TRUE"
                    and (r["value"] or "").strip())
    n_censored = sum(1 for r in values
                     if r["is_real_value"].strip().upper() != "TRUE"
                     and (r["value"] or "").strip())
    n_blank_records = len(values) - n_numeric - n_censored
    stb = read_csv(os.path.join(ROOT, "docs/extraction/stageB/STB_manifest.csv"))
    value_template_cells = sum(int(r["n_cells"]) for r in stb)
    assert value_template_cells == 3216, value_template_cells
    assert n_numeric + n_censored + n_blank_records == len(values)

    # family-complete value-layer scope: sum of family sizes over presence-complete rows
    FAM_COL = {"F1 physical": "full_F1_physical", "F2 metals": "full_F2_metals",
               "F3 nutrients": "full_F3_nutrients", "F4 organic": "full_F4_organic",
               "F1 physicala": "full_F1_physicala", "F1 clarity": "full_F1_clarity",
               "F6 aeration": "full_F6_aeration"}
    family_complete_cells = sum(
        sum(len(params) for fname, params in FAMILIES.items()
            if r[FAM_COL[fname]] == "1")
        for r in fdata if int(r["n_families_complete"]) >= 1)
    assert family_complete_cells == 3305, family_complete_cells

    # presence counts on the analyzed 764-row basis (the 867-row counts include
    # padding/placeholder/context rows, which are blank by construction)
    y_data = sum(1 for r in data for c in pcols if r[c].strip() == "Y")
    ne_data = sum(1 for r in data for c in pcols if r[c].strip() == "NE")
    blank_data = sum(1 for r in data for c in pcols if r[c].strip() not in ("Y", "NE"))

    # paper-level structure (forward-filled ids, as in the released family map).
    # The manuscript counts DATA rows per paper, so grouping uses data rows only.
    papers = {}
    for r, fm in zip(presence, fmap):
        assert r["scenario"] == fm["scenario"], "presence/family_map row order differs"
        if r["row_kind"] == "data":
            papers.setdefault(fm["batch_id_filled"], []).append(r)
    data_papers = set(papers)
    multi = [p for p in data_papers if len(papers[p]) > 1]
    same_panel = sum(1 for p in multi
                     if len({tuple(1 if examined(r, c) else 0 for c in pcols)
                             for r in papers[p]}) == 1)

    f6_cover = [r for r in data if any(examined(r, p) for p in FAMILIES["F6 aeration"])]
    f6_do_only = [r for r in f6_cover
                  if examined(r, "DO (mg/L)")
                  and not examined(r, "BOD (mg/L)") and not examined(r, "COD (mg/L)")]

    # F1 physical value-layer author stratum (S12a note): value-complete rows,
    # author-assigned labels only. Provenance follows the producer: a row is
    # rubric-derived when its (paper, scenario) pair is in the rubric log.
    f1_params = FAMILIES["F1 physical"]
    # real numeric values only (the producer's rule: is_real_value AND a
    # parseable value), keyed by (paper, row) exactly as the analysis code does
    vals_by_row = {}
    for r in values:
        if (r.get("is_real_value") or "").strip().upper() != "TRUE":
            continue
        if not (r.get("value") or "").strip():
            continue
        vals_by_row.setdefault((r["batch_id"], r["row_id"]), set()).add(r["param"])
    rubric_keys = {(r["batch_id"], r["scenario"]) for r in verdicts}
    author_by_paper = {}
    for pres_row, fm in zip(presence, fmap):
        if fm["row_kind"] != "data" or fm["full_F1_physical"] != "1":
            continue
        if not all(p in vals_by_row.get((fm["batch_id_filled"], fm["row_id"]), ())
                   for p in f1_params):
            continue
        v = (pres_row["verdict"] or "").strip()
        if v in ("", "TBD-StageB"):
            continue
        if (fm["batch_id_filled"], pres_row["scenario"]) in rubric_keys:
            continue
        author_by_paper.setdefault(fm["batch_id_filled"], []).append(v)
    n_author_rows_f1 = sum(len(v) for v in author_by_paper.values())
    n_author_papers_f1 = len(author_by_paper)
    single_papers = [p for p, v in author_by_paper.items() if len(set(v)) == 1]
    n_single_rows = sum(len(author_by_paper[p]) for p in single_papers)

    n_not_retrieved = sum(1 for r in manifest
                           if r["data_status"] not in ("extracted", "extracted (extension round)"))

    # per-family cover / complete rows and their paper counts (Table 2)
    fam_rows = {}
    for fname, params in FAMILIES.items():
        cov_rows = [r for r in data if any(examined(r, p) for p in params)]
        comp_rows = [r for r in data if all(examined(r, p) for p in params)]
        key = {"F1 physical": "F1_physical", "F2 metals": "F2_metals",
               "F3 nutrients": "F3_nutrients", "F4 organic": "F4_organic",
               "F1 physicala": "F1_physicala", "F1 clarity": "F1_clarity",
               "F6 aeration": "F6_aeration"}[fname]
        papers_of = lambda rows: len({fm["batch_id_filled"] for r, fm in zip(presence, fmap)
                                       if r["row_kind"] == "data" and r in rows})
        fam_rows[(fname, "cover_rows")] = len(cov_rows)
        fam_rows[(fname, "cover_papers")] = papers_of(cov_rows)
        fam_rows[(fname, "complete_rows")] = len(comp_rows)
        fam_rows[(fname, "complete_papers")] = papers_of(comp_rows)
        assert (fname, "complete_rows") and fam_rows[(fname, "complete_rows")] > 0

    # verdict concordance counts (results/verdict_agreement_summary.csv)
    vag = {r["metric"]: r["value"] for r in
           read_csv(os.path.join(ROOT, "results", "verdict_agreement_summary.csv"))}
    n_author_rows = int(float(vag["n_author_rows"]))
    n_compared = int(float(vag["n_compared"]))

    # intake removals (corpus_funnel.md stage 6: off-topic + flagged venues)
    import re
    funnel_md = open(os.path.join(ROOT, "manuscript", "corpus_funnel.md"),
                     encoding="utf-8").read()
    m_off = re.search(r"removals \((\d+):", funnel_md)
    m_pred = re.search(r"removal \((\d+);", funnel_md)
    assert m_off and m_pred, "intake-removal counts not found in corpus_funnel.md"
    n_offtopic, n_predatory = int(m_off.group(1)), int(m_pred.group(1))
    trace_hits = sum(1 for r in trace if r["found_in_source_text"].strip().upper() == "TRUE")

    out = [
        ("total_rows", len(presence), "rows", "presence matrix rows (data + padding + placeholder + context)"),
        ("data_rows", kinds["data"], "rows", "row_kind = data"),
        ("padding_rows", kinds["padding"], "rows", "row_kind = padding"),
        ("placeholder_rows", kinds["placeholder"], "rows", "row_kind = placeholder"),
        ("context_rows", kinds["context"], "rows", "row_kind = context"),
        ("parameters", len(pcols), "parameters", "columns of the presence layer"),
        ("presence_y", y, "cells", "cells reported (Y) over all 867 rows"),
        ("presence_ne", ne, "cells", "cells examined but not found (NE) over all 867 rows"),
        ("presence_blank", blank, "cells", "cells not reported (blank/NR) over all 867 rows"),
        ("presence_y_data", y_data, "cells", "Y cells over the 764 analyzed data rows"),
        ("presence_ne_data", ne_data, "cells", "NE cells over the 764 analyzed data rows"),
        ("presence_blank_data", blank_data, "cells", "blank/NR cells over the 764 analyzed data rows"),
        ("value_cells", len(values), "cells", "value-layer records (numeric + censored + blank)"),
        ("value_numeric", n_numeric, "cells", "numeric value observations (is_real_value TRUE)"),
        ("value_censored", n_censored, "cells", "censored/text-coded records (non-real, non-empty)"),
        ("value_blank_records", n_blank_records, "cells", "blank records in values.csv"),
        ("value_template_cells", value_template_cells, "cells", "Stage-B extraction cells"),
        ("family_complete_cells", family_complete_cells, "cells", "family-complete value-layer cells"),
        ("verdict_records", len(verdicts), "rows", "rows in the verdict derivation log"),
        ("rubric_labels", len(rubric), "rows", "rubric-assigned verdict labels"),
        ("low_confidence", low_conf, "rows", "verdict rows flagged low confidence"),
        ("unassigned_parameters", len(UNASSIGNED), "parameters", "parameters in no family"),
        ("sar_rows", sum(1 for r in presence if examined(r, "SAR")), "rows", "rows reporting SAR"),
        ("al_rows", sum(1 for r in presence if examined(r, "Al (µg/L)")), "rows", "rows reporting Al"),
        ("configurations", len(pat), "patterns", "unique binary presence configurations"),
        ("rows_sharing_pattern", sum(v for v in pat.values() if v > 1), "rows", "rows sharing a configuration"),
        ("rows_in_groups_gt5", sum(v for v in pat.values() if v > 5), "rows", "rows in configuration groups larger than five"),
        ("cover_rows", cover, "rows", "data rows reporting at least one family parameter"),
        ("presence_complete_rows", complete, "rows", "data rows complete for at least one family"),
        ("data_papers", len(data_papers), "papers", "papers contributing data rows"),
        ("multi_row_papers", len(multi), "papers", "papers contributing more than one row"),
        ("identical_panel_papers", same_panel, "papers", "multi-row papers repeating one parameter panel"),
        ("median_rows_per_paper", statistics.median([len(papers[p]) for p in data_papers]), "rows", "median rows per data paper"),
        ("density_median", round(statistics.median(dens), 2), "fraction", "median presence flags per row"),
        ("density_q1", round(q[0], 2), "fraction", "first quartile of presence flags per row"),
        ("density_q3", round(q[2], 2), "fraction", "third quartile of presence flags per row"),
        ("max_parameters_per_row", max(sum(1 for c in pcols if examined(r, c)) for r in data), "parameters", "largest panel in one row"),
        ("f6_cover_rows", len(f6_cover), "rows", "F6 aeration cover rows"),
        ("f6_do_only_rows", len(f6_do_only), "rows", "F6 cover rows with DO but neither BOD nor COD"),
        ("papers_in_manifest", len(manifest), "papers", "released corpus manifest"),
        ("papers_not_retrieved", n_not_retrieved, "papers", "manifest papers without extracted data"),
        ("papers_extracted", len(manifest) - n_not_retrieved, "papers", "manifest papers with extracted data"),
        ("trace_rows", len(trace), "cells", "extraction verification trace"),
        ("trace_hits", trace_hits, "cells", "trace values confirmed verbatim in the source"),
        ("trace_share_pct", round(100.0 * len(trace) / n_numeric, 1), "percent",
         "trace cells as a share of the numeric value cells"),
        ("labelled_rows", len(lab), "rows", "data rows carrying a verdict class"),
        ("unlabelled_rows", len(unl), "rows", "data rows without a verdict class"),
        ("unlabelled_placeholder", sum(1 for r in unl if r["verdict"].strip() == "TBD-StageB"),
         "rows", "unlabeled data rows carrying the placeholder label"),
        ("unlabelled_blank", sum(1 for r in unl if not r["verdict"].strip()),
         "rows", "unlabeled data rows with an empty verdict"),
        ("insufficient_value_rows", len(insuff), "rows", "rows flagged insufficient values for verdict"),
        ("insufficient_value_labelled", sum(1 for r in insuff if r["verdict"].strip()),
         "rows", "insufficient-value rows that carry a label"),
        ("author_labels_data", n_author_rows, "rows", "author-supplied verdict labels on data rows"),
        ("rubric_compared_rows", n_compared, "rows", "rows where rubric and author labels can be compared"),
        ("undecidable_rows", n_author_rows - n_compared, "rows",
         "author-labeled rows the rubric cannot classify"),
        ("intake_removed", n_offtopic + n_predatory, "records",
         "records removed at intake (off-topic plus flagged venues)"),
        ("f1_value_author_rows", n_author_rows_f1, "rows",
         "F1 physical value-layer rows with an author-assigned label"),
        ("f1_value_author_papers", n_author_papers_f1, "papers",
         "papers contributing those F1 physical author-labeled value rows"),
        ("f1_value_single_class_papers", len(single_papers), "papers",
         "of those, papers whose author labels carry a single verdict class"),
        ("f1_value_single_class_rows", n_single_rows, "rows",
         "rows in the single-class papers"),
        ("f6_do_only_pct", round(100.0 * len(f6_do_only) / len(f6_cover), 1), "percent",
         "share of F6 cover rows reporting DO but neither BOD nor COD"),
        ("rubric_share_of_labelled_pct", round(100.0 * len(rubric) / len(lab), 1), "percent",
         "share of labeled data rows whose label is rubric-derived"),
        ("fig3_b086_unlabelled_rows", sum(1 for r, fm in zip(presence, fmap)
                                          if fm["batch_id_filled"] == "B086"
                                          and r["row_kind"] == "data"
                                          and r["verdict"].strip() == "TBD-StageB"),
         "rows", "unlabeled placeholder rows of study B086 (Fig. 3 caption)"),
    ]
    for fname in FAMILIES:
        for what in ("cover_rows", "cover_papers", "complete_rows", "complete_papers"):
            key = fname.lower().replace(" ", "_")
            out.append((f"family_{key}_{what}", fam_rows[(fname, what)], "rows/papers",
                        f"{fname}: {what}"))
    for fam, cnt in sorted(rubric_counts.items()):
        out.append((f"rubric_{fam.lower().replace(' ', '_')}", cnt, "rows",
                    f"rubric-assigned {fam} labels"))

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, "corpus_composition.csv")
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value", "unit", "definition"])
        for row in out:
            w.writerow(row)
    os.replace(tmp, path)
    print(f"[corpus-composition] wrote {os.path.relpath(path, ROOT)} "
          f"({len(out)} metrics)")


if __name__ == "__main__":
    main()
