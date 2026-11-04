#!/usr/bin/env python3
"""verdict_rubric_replay.py — build-time replay of the frozen verdict rubric (D197).

Reads (read-only):  results/verdict_derivation_log.csv   (the 338 stored verdicts)
                    docs/extraction/stageA/matrix_presence_v2.csv
                    docs/family_map.csv
                    docs/extraction/stageB/matrix_values_v2.csv
Writes:             results/verdict_replay_agreement.csv

Re-derives every stored verdict with the SHIPPED decision function
(`verdict_derivation.classify`) and reports how many agree. Because both the
producer and this replay call the same function, the resulting count is an
end-to-end assertion that the frozen log is what the frozen rubric produces —
replacing the hand-typed "336 of the 338" constant formerly printed by Table S14.

Deterministic (no randomness). Read-only on every frozen layer.
Run: python3 code/verdict_rubric_replay.py
"""
from __future__ import annotations
import csv
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from verdict_derivation import classify  # noqa: E402

PRES = os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv")
FMAP = os.path.join(ROOT, "docs/family_map.csv")
MATRIX = os.path.join(ROOT, "docs/extraction/stageB/matrix_values_v2.csv")
LOG = os.path.join(ROOT, "results/verdict_derivation_log.csv")
OUT = os.path.join(ROOT, "results/verdict_replay_agreement.csv")

FIELDS = ["batch_id", "row_id", "field", "stored", "replayed"]


def main() -> int:
    pres = list(csv.DictReader(open(PRES, encoding="utf-8")))
    fmap = list(csv.DictReader(open(FMAP, encoding="utf-8-sig")))
    mrows: dict = defaultdict(dict)
    for r in csv.DictReader(open(MATRIX, encoding="utf-8")):
        mrows[(r["batch_id"], r["row_id"])][r["param"]] = r
    log = list(csv.DictReader(open(LOG, encoding="utf-8-sig")))

    n_rows = n_full = 0
    mismatches, field_counts = [], defaultdict(int)
    for l in log:
        key = (l["batch_id"], l["row_id"])
        i = int(l["row_id"]) - 1
        if key not in mrows or not (0 <= i < len(pres)):
            mismatches.append(dict(batch_id=key[0], row_id=key[1], field="input",
                                   stored="present", replayed="MISSING"))
            field_counts["input"] += 1
            continue
        row = pres[i]
        res = classify(l["scenario"], (row.get("scenario_type") or "").strip(),
                       mrows[key], batch_id=key[0])
        got = dict(
            verdict=res["verdict"] or "",
            route=res["route"],
            bands="; ".join(f"{p}->B{b}" for p, b in res["bands"]),
            citation=res["citation"],
            low_confidence="TRUE" if res["low"] else "",
            n_contrib=str(len(res["bands"])) if res["route"] == "values" else "",
        )
        n_rows += 1
        row_ok = True
        for f in ("verdict", "route", "bands", "citation", "low_confidence", "n_contrib"):
            if (l.get(f) or "") != got[f]:
                row_ok = False
                field_counts[f] += 1
                mismatches.append(dict(batch_id=key[0], row_id=key[1], field=f,
                                       stored=l.get(f) or "", replayed=got[f]))
        n_full += 1 if row_ok else 0

    tmp = OUT + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for m in mismatches:
            w.writerow(m)
    os.replace(tmp, OUT)

    print(f"[verdict-replay] rows replayed: {n_rows}")
    print(f"[verdict-replay] rows reproduced on all 6 fields: {n_full}")
    if mismatches:
        for m in mismatches[:10]:
            print(f"  MISMATCH {m['batch_id']}/{m['row_id']} {m['field']}: "
                  f"stored={m['stored']!r} replayed={m['replayed']!r}")
        print(f"[verdict-replay] {len(mismatches)} field mismatches -> {OUT}")
    else:
        print(f"[verdict-replay] 0 mismatches -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
