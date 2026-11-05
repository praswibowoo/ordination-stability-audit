#!/usr/bin/env python3
"""stageB_merge.py — Stage-B merge + unit harmonization + QA (P1-2.2).

Reads filled chunk templates (docs/extraction/stageB/STB*_template.csv), re-parses every
value_as_published cell with harmonize_units.parse_cell, runs harmonize() for unit
normalization, applies QA ranges, and writes:

  docs/extraction/stageB/matrix_values_v2.csv     (long format, frozen value layer)
  results/unit_conversion_log.csv                 (batch_id, param, raw, rule, normalized)
  docs/extraction/stageB/stageB_qa_report.md      (completion + QA flags)

Flags per value-layer row (Data_Dictionary §3.4):
  values_available (cell resolved) · stageB_done (all cells of that row resolved)
  era (observational | classic-tox) · value_as_published (verbatim anchor)

CLI:
  python3 code/data/stageB_merge.py            # merge whatever is filled; report missing
  python3 code/data/stageB_merge.py --selftest # exercise parse+hrmonize path on synthetic cells
"""
from __future__ import annotations
import csv, glob, os, re, sys
from datetime import date

import sys
_CODE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(_CODE)
for _sub in ("core", "data", "ees", "mds"):
    _p = os.path.join(_CODE, _sub)
    if _p not in sys.path:
        sys.path.insert(0, _p)
sys.path.insert(0, os.path.join(ROOT, "code"))
from harmonize_units import parse_cell, harmonize, qa  # noqa: E402

SB = os.path.join(ROOT, "docs/extraction/stageB")
MATRIX = os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv")
OUT_VALUES = os.path.join(SB, "matrix_values_v2.csv")
OUT_LOG = os.path.join(ROOT, "results/unit_conversion_log.csv")
OUT_QA = os.path.join(SB, "stageB_qa_report.md")
TODAY = date.today().isoformat()

ELEMENT = {  # for harmonize(element, value, unit) — all harmonizable params
    "As (µg/L)": "as", "Cd (µg/L)": "cd", "Cr (µg/L)": "cr", "Cu (µg/L)": "cu",
    "Fe (µg/L)": "fe", "Hg (µg/L)": "hg", "Mn (µg/L)": "mn", "Ni (µg/L)": "ni",
    "Pb (µg/L)": "pb", "Zn (µg/L)": "zn", "Al (µg/L)": "al", "Co (µg/L)": "co",
    "Si (µmol/L)": "sio2", "DO (mg/L)": "do", "Secchi (m)": "secchi",
    "Salinity (ppt/‰)": "salinity", "TDS": "tds", "EC (µS/cm)": "ec",
    "Temperature (°C)": "temp", "Cl- (ppt)": "cl", "Hardness (mg/L)": "hardness",
    "NO2 (mg/L)": "no2", "NO3 (mg/L)": "no3", "NH4 (mg/L)": "nh4", "PO4 (mg/L)": "po4",
    "H2S (µg/L)": "h2s",
}
VALUE_HDR = ["batch_id", "row_id", "scenario", "param", "family", "value_as_published",
             "value", "value_max", "unit_published", "unit_rule_id", "censored", "sd_or_note",
             "extraction_confidence", "is_real_value", "values_available", "stageB_done",
             "era", "edited_by", "entry_date", "notes"]


def atomic_write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def era_of(scenario: str, notes: str) -> str:
    s = (scenario or "") + " " + (notes or "")
    return "classic-tox" if re.search(r"lc\s*50|lc50|mortality|bioassay|96h|96 h", s, re.I) else "observational"


def load_prev_dates(path):
    """Existing matrix dates, keyed (batch_id, row_id, param) — entry_date is provenance
    and must survive re-merges (templates carry no date column; D77)."""
    prev = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                k = ((r.get("batch_id") or "").strip(),
                     str((r.get("row_id") or "").strip()),
                     (r.get("param") or "").strip())
                prev[k] = {"entry_date": (r.get("entry_date") or "").strip(),
                           "edited_by": (r.get("edited_by") or "").strip()}
    return prev


def main():
    filled_rows, missing_rows, conv_log, qa_flags = [], [], [], []
    prev_dates = load_prev_dates(OUT_VALUES)
    n_cells = n_filled = 0
    row_tot, row_got = {}, {}
    for path in sorted(glob.glob(os.path.join(SB, "STB*_template.csv"))):
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            n_cells += 1
            key = (r["batch_id"], str(r["row_id"]))
            row_tot[key] = row_tot.get(key, 0) + 1
            va = (r.get("value_as_published") or "").strip()
            v = (r.get("value") or "").strip()
            cens = (r.get("censored") or "").strip()
            if not va and not v and not cens:
                missing_rows.append(r)
                continue
            n_filled += 1
            row_got[key] = row_got.get(key, 0) + 1
            rec = {k: (r.get(k) or "").strip() for k in VALUE_HDR if k in r}
            rec.update({k: "" for k in VALUE_HDR if k not in r})
            rec["value_as_published"] = va
            rec["edited_by"] = rec.get("edited_by") or ""
            prev = prev_dates.get((rec.get("batch_id", ""), str(rec.get("row_id", "")),
                                   rec.get("param", "")), {})
            rec["edited_by"] = rec.get("edited_by") or prev.get("edited_by", "")
            rec["entry_date"] = rec.get("entry_date") or prev.get("entry_date", "") or TODAY
            rec["era"] = era_of(rec.get("scenario", ""), rec.get("notes", ""))

            parsed = parse_cell(va) if va else {}
            val = None
            if cens:
                rec["censored"] = cens
                parsed.setdefault("NE", True)
            if parsed.get("NE"):
                rec["values_available"] = "TRUE"
                rec["is_real_value"] = "FALSE"
            elif parsed:
                val = parsed.get("value") if parsed.get("value") is not None else None
                if val is None and v:
                    try:
                        val = float(v)
                    except ValueError:
                        val = None
                if val is not None:
                    rec["value"] = f"{val:g}"
                    if parsed.get("value_max") is not None:
                        rec["value_max"] = f"{parsed['value_max']:g}"
                    if parsed.get("flags"):
                        rec["sd_or_note"] = ";".join(parsed["flags"]) + (
                            ";" + rec["sd_or_note"] if rec.get("sd_or_note") else "")
                    rec["values_available"] = "TRUE"
                    rec["is_real_value"] = "TRUE"
                    unit = (r.get("unit_published") or "").strip()
                    element = ELEMENT.get(r.get("param", ""))
                    if element and unit:
                        try:
                            newv, rule = harmonize(element, val, unit)
                            if rule:
                                conv_log.append((r["batch_id"], r["param"], va, rule, f"{newv:g}"))
                                rec["unit_rule_id"] = rule
                                val = newv
                                rec["value"] = f"{newv:g}"
                                if parsed.get("value_max") is not None:
                                    vmax2, _ = harmonize(element, parsed["value_max"], unit)
                                    rec["value_max"] = f"{vmax2:g}"
                        except Exception as e:  # noqa: BLE001
                            rec["notes"] = (rec.get("notes", "") + f" | harmonize-error:{e}").strip(" |")
                else:
                    rec["values_available"] = "FALSE"
                    rec["is_real_value"] = "FALSE"
            else:
                rec["values_available"] = "FALSE"
                rec["is_real_value"] = "FALSE"
            if val is not None:
                flag = qa(r.get("param", ""), val)
                if flag:
                    qa_flags.append((r["batch_id"], r["row_id"], r["param"], rec["value"], flag))
                if rec["value_max"]:
                    try:
                        if float(rec["value_max"]) < float(rec["value"]):
                            qa_flags.append((r["batch_id"], r["row_id"], r["param"],
                                             f"{rec['value']}>{rec['value_max']}",
                                             "CHECK: value>value_max (malformed as-printed range)"))
                    except ValueError:
                        pass
            filled_rows.append(rec)

    # stageB_done per row = ALL template cells of that row resolved
    for rec in filled_rows:
        key = (rec["batch_id"], str(rec["row_id"]))
        rec["stageB_done"] = "TRUE" if row_got.get(key, 0) == row_tot.get(key, 0) else "FALSE"

    # write values CSV
    lines = [",".join(VALUE_HDR)]
    for rec in filled_rows:
        lines.append(",".join(f'"{rec[k]}"' if ("," in rec[k] or ";" in rec[k] or '"' in rec[k])
                              else rec[k] for k in VALUE_HDR))
    atomic_write(OUT_VALUES, "\n".join(lines) + "\n")

    # conversion log
    log_lines = ["batch_id,param,raw,rule,normalized"] + [",".join(f'"{c}"' if "," in str(c) else str(c)
                                                                   for c in row) for row in conv_log]
    atomic_write(OUT_LOG, "\n".join(log_lines) + "\n")

    # QA report
    done_rows = sum(1 for k, tot in row_tot.items() if row_got.get(k, 0) == tot)
    tot_rows = len(row_tot)
    q = [f"# Stage-B QA report ({TODAY})", "",
         f"- cells total: {n_cells} | filled: {n_filled} ({n_filled / n_cells:.1%}) | missing: {len(missing_rows)}",
         f"- rows with all cells resolved: {done_rows} / {tot_rows}",
         f"- unit conversions logged: {len(conv_log)}", f"- QA range flags: {len(qa_flags)}", ""]
    if missing_rows:
        q.append("## Missing cells (by paper; top 20)")
        from collections import Counter
        c = Counter(r["batch_id"] for r in missing_rows)
        q += [f"- {k}: {v}" for k, v in sorted(c.items())[:20]]
        q.append("")
    if qa_flags:
        q.append("## QA flags (out-of-range — review)")
        q.append("| batch_id | row | param | value | flag |")
        q.append("|---|---|---|---|---|")
        q += [f"| {a} | {b} | {c} | {d} | {e} |" for a, b, c, d, e in qa_flags]
    atomic_write(OUT_QA, "\n".join(q) + "\n")

    print(f"cells {n_filled}/{n_cells} filled | rows done {done_rows}/{tot_rows} | "
          f"conversions {len(conv_log)} | QA flags {len(qa_flags)}")
    print("wrote:", OUT_VALUES, OUT_LOG, OUT_QA, sep="\n  ")


def selftest():
    cases = [("7.2±0.3", "DO (mg/L)", "mg/L"), ("23-28", "TSS (mg/L)", "mg/L"),
             ("<0.001", "Pb (µg/L)", "µg/L"), ("0.5", "Pb (µg/L)", "mg/L"),
             ("12.5", "Temperature (°C)", "°C"), ("", "pH", "")]
    ok = True
    for va, param, unit in cases:
        p = parse_cell(va)
        el = ELEMENT.get(param)
        if el and p.get("value") is not None and unit:
            v, rule = harmonize(el, p["value"], unit)
            print(f"  {va!r:12s} {param:12s} -> {p} | harmonize: {v} {rule}")
        else:
            print(f"  {va!r:12s} {param:12s} -> {p}")
    print("selftest PASS (parse+hrmonize path executes)")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        raise SystemExit(selftest())
    main()
