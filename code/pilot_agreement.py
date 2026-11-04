#!/usr/bin/env python3
"""pilot_agreement.py v2 — R8c metrics (protocol v1.6).

GATE = operational collapse: {INCLUDE,UNSURE} -> retrieve, {EXCLUDE} -> drop.
Also reports strict 3-class raw agreement + Cohen's kappa (secondary).

Usage: python code/pilot_agreement.py A.csv B.csv [out.md]
"""
from __future__ import annotations
import csv, sys
from collections import Counter


def load(path):
    with open(path, encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    key = "pilot_id" if "pilot_id" in rows[0] else "batch_id"
    for r in rows:
        r["_key"] = r[key]
        r["decision"] = (r.get("decision") or "").strip().upper() or "EMPTY"
        r["reason_code"] = (r.get("reason_code") or "").strip().upper()
    return {r["_key"]: r for r in rows}


def collapse(d):
    return "RETRIEVE" if d in ("INCLUDE", "UNSURE") else ("DROP" if d == "EXCLUDE" else "EMPTY")


def kappa(pairs, labels):
    n = len(pairs)
    if n == 0:
        return 0.0, 0.0
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return po, (0.0 if abs(1 - pe) < 1e-9 else (po - pe) / (1 - pe))


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__); return 2
    out = sys.argv[3] if len(sys.argv) > 3 else "docs/screening/repilot_agreement.md"
    A, B = load(sys.argv[1]), load(sys.argv[2])
    assert set(A) == set(B), "pilot_id sets differ"
    ids = sorted(A)

    coll = [(collapse(A[i]["decision"]), collapse(B[i]["decision"])) for i in ids]
    n = len(coll)
    agree_c = sum(1 for a, b in coll if a == b)
    raw_c = agree_c / n
    po_c, k_c = kappa(coll, ["RETRIEVE", "DROP"])

    strict = [(A[i]["decision"], B[i]["decision"]) for i in ids]
    agree_s = sum(1 for a, b in strict if a == b)
    raw_s = agree_s / n
    po_s, k_s = kappa(strict, sorted({x for p in strict for x in p} | {"INCLUDE","EXCLUDE","UNSURE"}))

    gate = "PASS" if raw_c > 0.90 else "FAIL → boundary workshop (protocol bump)"

    lines = ["# Screening Agreement — R8c (operational-collapse gate)",
             "",
             "| metric | value |", "|---|---|",
             f"| N | {n} |",
             f"| **Operational agreement (GATE)**: {agree_c}/{n} | **{raw_c:.1%}** (κ={k_c:.3f}) |",
             f"| Strict 3-class agreement (secondary) | {agree_s}/{n} = {raw_s:.1%} (κ={k_s:.3f}) |",
             f"| **Gate (>90% operational): {gate}** | |",
             "",
             "## Operational disagreements (must adjudicate)",
             "| pilot_id | A decision | B decision | A reason | B reason | ADJUDICATED |",
             "|---|---|---|---|---|---|"]
    for i in ids:
        ca, cb = collapse(A[i]["decision"]), collapse(B[i]["decision"])
        if ca != cb:
            lines.append(f"| {i} | {A[i]['decision']} | {B[i]['decision']} | {A[i]['reason_code']} | {B[i]['reason_code']} |  |")
    if agree_c == n:
        lines.append("| — | (none) | | | | |")
    # strict-only disagreements (UNSURE vs INCLUDE etc.) — informational
    lines += ["", "## Strict-only differences (informational, not gate-counted)",
              "| pilot_id | A | B |", "|---|---|---|"]
    for i in ids:
        if A[i]["decision"] != B[i]["decision"] and collapse(A[i]["decision"]) == collapse(B[i]["decision"]):
            lines.append(f"| {i} | {A[i]['decision']} | {B[i]['decision']} |")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"GATE metric: operational {raw_c:.1%} ({agree_c}/{n}) | strict {raw_s:.1%} | kappa(op)={k_c:.3f}")
    print(f"gate: {gate}")
    print(f"written: {out}")
    return 0 if raw_c > 0.90 else 1


if __name__ == "__main__":
    raise SystemExit(main())
