#!/usr/bin/env python3
"""verdict_derivation.py — implement FROZEN verdict rubric v1.0 (D74) for P1 (D47).

Derives the 589 paper-unlabeled verdicts (blank / TBD-StageB) from Stage-B VALUES.
Reads:  docs/extraction/stageA/matrix_presence_v2.csv  (verdict, cause_effect_note targets)
        docs/family_map.csv                           (row_id <-> presence index bridge)
        docs/extraction/stageB/matrix_values_v2.csv   (harmonized values)
Writes: presence csv (verdict + cause_effect_note)  [atomic]
        results/verdict_derivation_log.csv          [atomic]

Rules implemented ONLY as specified in docs/verdict_rubric.md v1.0 (frozen).
Idempotent: never overrides non-candidate verdicts; note-marker guards prevent double-append.
"""
from __future__ import annotations
import csv, os, re, sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRES = os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv")
FMAP = os.path.join(ROOT, "docs/family_map.csv")
MATRIX = os.path.join(ROOT, "docs/extraction/stageB/matrix_values_v2.csv")
LOG = os.path.join(ROOT, "results/verdict_derivation_log.csv")
RULE_VERSION = "v1.0"

STD_RE = re.compile(r"baku\s*mutu|limit|standard|who|nwqs|criteria|kelas|class\s|doe|ccme|mmwqs", re.I)
LETHAL_RE = re.compile(r"lc\s*[-]?\s*50|mortality|96\s*h|lethal|death", re.I)

METAL_STD = {"Cd (µg/L)": 10.0, "Cu (µg/L)": 20.0, "Pb (µg/L)": 30.0, "Cr (µg/L)": 50.0}  # GR22 (Q6)
CITE = {"DO (mg/L)": "PP 22/2021 DO", "BOD (mg/L)": "PP 22/2021 BOD", "COD (mg/L)": "PP 22/2021 COD",
        "pH": "PP 22/2021 pH", "NO3 (mg/L)": "PP 22/2021 NO3", "NH4 (mg/L)": "PP 22/2021 NH4",
        "PO4 (mg/L)": "KepMenLH 51/2004 PO4", "Total Coliform (CFU/100mL)": "PP 22/2021 coliform",
        "Cd (µg/L)": "PP 22/2021 Cd", "Cu (µg/L)": "PP 22/2021 Cu", "Pb (µg/L)": "PP 22/2021 Pb",
        "Cr (µg/L)": "PP 22/2021 Cr"}
CORE = {"DO (mg/L)", "pH", "Temperature (°C)"}


def band(param: str, x: float):
    if param == "DO (mg/L)": return 0 if x >= 5 else 1 if x >= 4 else 2 if x >= 2 else 3
    if param == "BOD (mg/L)": return 0 if x <= 3 else 1 if x <= 6 else 2 if x <= 12 else 3
    if param == "COD (mg/L)": return 0 if x <= 25 else 1 if x <= 40 else 2 if x <= 80 else 3
    if param == "pH": return 0 if 6.5 <= x <= 8.5 else 1 if (6 <= x < 6.5 or 8.5 < x <= 9) else 2 if (5.5 <= x < 6 or 9 < x <= 9.5) else 3
    if param == "Temperature (°C)": return 0 if 26 <= x <= 32 else 1 if (22 <= x < 26 or 32 < x <= 35) else 2
    if param == "NO3 (mg/L)": return 0 if x <= 1 else 1 if x <= 5 else 2 if x <= 10 else 3
    if param == "NH4 (mg/L)": return 0 if x <= 0.3 else 1 if x <= 1 else 2 if x <= 3 else 3
    if param == "PO4 (mg/L)": return 0 if x <= 0.1 else 1 if x <= 0.5 else 2 if x <= 1 else 3
    if param == "Total Coliform (CFU/100mL)": return 0 if x <= 1000 else 1 if x <= 5000 else 2 if x <= 10000 else 3
    if param == "TSS (mg/L)": return 0 if x <= 25 else 1
    return None


def append_note(note: str, add: str, marker: str) -> str:
    if marker in (note or ""):
        return note
    return ((note or "").strip() + " | " + add).strip(" |")


def atomic_write(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def classify(scen: str, stype: str, pr: dict, batch_id: str = "") -> dict:
    """Frozen rubric v1.0 decision for one candidate row (pure; no I/O, no side effects).

    D197: extracted verbatim from main() so that the shipped producer and the
    build-time replay (code/verdict_rubric_replay.py) share ONE implementation —
    the S14 claim "the rubric re-implementation reproduces N of the 338 stored
    verdicts" is then an end-to-end assertion instead of a hand-typed constant.

    Returns dict(verdict, route, bands, harmed, low, citation, note):
      verdict is None for the insufficient-values branch (caller writes the note).
    """
    if stype == "standard" or STD_RE.search(scen):
        return dict(verdict="Reference standard", route="standard", bands=[], harmed=[], low=False,
                    citation=scen,
                    note=f"[derived {RULE_VERSION}: mechanical standard route — citation: {scen}]")
    era = next(iter(pr.values()))["era"]
    notes_all = " ".join(r.get("notes", "") for r in pr.values())
    if era == "classic-tox":
        v = "Lethal/Mortality risk" if LETHAL_RE.search(scen + " " + notes_all) else "Stressful"
        cite = f"study {batch_id} ({scen})"
        return dict(verdict=v, route="bioassay", bands=[], harmed=[], low=False, citation=cite,
                    note=f"[derived {RULE_VERSION}: bioassay route — {cite}]")
    # value rules
    bands, harmed, low, core = [], [], False, False
    for p, r in pr.items():
        v = (r.get("value") or "").strip()
        c = (r.get("censored") or "").strip()
        if r.get("extraction_confidence") == "low":
            low = True
        if p in METAL_STD:
            if not v:
                continue
            b = 0 if c == "lt" else (3 if c == "gt" else (0 if float(v) / METAL_STD[p] < 0.1 else 1 if float(v) / METAL_STD[p] < 0.5 else 2 if float(v) / METAL_STD[p] <= 1 else 3))
            bands.append((p, b))
            core = True
            if b == 3:
                harmed.append(CITE[p])
            continue
        b = None
        if v:
            b = band(p, float(v))
            # worst-case for ranged cells (rubric §5.2): max of endpoint bands —
            # covers max-direction params (PO4/NH4/NO3/TSS…) banded at value_max too (D91)
            vm = (r.get("value_max") or "").strip()
            if vm:
                try:
                    b2 = band(p, float(vm))
                    if b2 is not None and b2 > b:
                        b = b2
                except ValueError:
                    pass
        elif c == "lt":
            b = 0
        elif c == "gt":
            b = 3
        if b is None:
            continue
        bands.append((p, b))
        if p in CORE:
            core = True
        if b == 3:
            harmed.append(CITE.get(p, ""))
    if len(bands) < 2 or not core:
        return dict(verdict=None, route="insufficient", bands=bands, harmed=harmed, low=low,
                    citation="", note="")
    mb = max(b for _, b in bands)
    do = next((float(r["value"]) for p, r in pr.items() if p == "DO (mg/L)" and (r.get("value") or "").strip()), None)
    ph = next((float(r["value"]) for p, r in pr.items() if p == "pH" and (r.get("value") or "").strip()), None)
    col = next((float(r["value"]) for p, r in pr.items() if p == "Total Coliform (CFU/100mL)" and (r.get("value") or "").strip()), None)
    allB0 = all(b == 0 for _, b in bands)
    if allB0 and do is not None and do >= 6 and (col is None or col <= 1000) and (ph is not None and 7 <= ph <= 8.5):
        verdict = "Favorable"
    elif allB0:
        verdict = "Stable"
    elif mb == 1:
        verdict = "Mild stress"
    elif mb == 2:
        verdict = "Stressful"
    else:
        verdict = "Harmful" if harmed else "Stressful"
    bstr = "; ".join(f"{p}->B{b}" for p, b in bands)
    rv = f"[derived {RULE_VERSION}: {bstr}" + (f"; cite={','.join(sorted(set(filter(None, harmed))))}" if harmed else "") + (f"; low-confidence" if low else "") + "]"
    return dict(verdict=verdict, route="values", bands=bands, harmed=harmed, low=low,
                citation=",".join(sorted(set(filter(None, harmed)))), note=rv)


def main():
    pres = list(csv.DictReader(open(PRES, encoding="utf-8")))
    fmap = list(csv.DictReader(open(FMAP, encoding="utf-8-sig")))
    mrows = defaultdict(dict)
    for r in csv.DictReader(open(MATRIX, encoding="utf-8")):
        mrows[(r["batch_id"], r["row_id"])][r["param"]] = r
    header = list(pres[0].keys())

    log_rows, summary = [], Counter()
    for fr in fmap:
        rid = str(fr["row_id"])
        i = int(rid) - 1
        if i < 0 or i >= len(pres):
            summary["skip:bad_index"] += 1
            continue
        row = pres[i]
        cur = (row.get("verdict") or "").strip()
        if cur not in ("", "TBD-StageB"):
            continue  # never touch labeled rows
        if fr["row_kind"] != "data":
            summary["skip:row_kind"] += 1
            continue
        key = (fr["batch_id_filled"], rid)
        if key not in mrows:
            summary["no_values"] += 1
            continue
        scen = fr["scenario"]
        stype = (row.get("scenario_type") or "").strip()
        res = classify(scen, stype, mrows[key], batch_id=key[0])
        if res["verdict"] is None:
            row["cause_effect_note"] = append_note(row.get("cause_effect_note"), "insufficient values for verdict (v1.0)", "insufficient values for verdict (v1.0)")
            summary["insufficient"] += 1
            continue
        row["verdict"] = res["verdict"]
        row["cause_effect_note"] = append_note(row.get("cause_effect_note"), res["note"], "[derived v1.0")
        log_rows.append(dict(batch_id=key[0], row_id=rid, scenario=scen, verdict=res["verdict"],
                             route=res["route"], bands="; ".join(f"{p}->B{b}" for p, b in res["bands"]),
                             citation=res["citation"], low_confidence="TRUE" if res["low"] else "",
                             n_contrib=len(res["bands"]) if res["route"] == "values" else ""))
        summary[res["verdict"]] += 1

    atomic_write(PRES, header, pres)
    if log_rows:
        atomic_write(LOG, list(log_rows[0].keys()), log_rows)
    print("derived:", dict((k, v) for k, v in sorted(summary.items())))
    print("log rows:", len(log_rows), "->", LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
