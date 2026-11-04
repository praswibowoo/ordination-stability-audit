#!/usr/bin/env python3
"""registry_check.py — H-number / DOI collision checks (F11, R5 mitigation).

Usage:
    python code/registry_check.py papers.csv

`papers.csv` columns (header row required):
    h_number,doi,title
Checks:
  1. Duplicate h_number values (must be unique).
  2. Duplicate DOI values (case-insensitive, must be unique).
  3. h_number format: ^H\\d{3}$ and sequence sanity (report min/max).
  4. Optional cross-check against a `known` H-number list (v1: H001..H036)
     — fails if a v1 number is re-used.

Exit code 0 = clean, 1 = problems found. Prints a report to stdout.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter

V1_RESERVED = {f"H{i:03d}" for i in range(1, 37)}  # v1 H001–H036: never reuse
PATTERN = re.compile(r"^H\d{3}$")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    rows = list(csv.DictReader(open(sys.argv[1])))
    problems: list[str] = []

    for i, r in enumerate(rows, start=2):
        h = (r.get("h_number") or "").strip()
        doi = (r.get("doi") or "").strip()
        if not PATTERN.match(h):
            problems.append(f"line {i}: bad H-format '{h}' (expect H###)")
        if h in V1_RESERVED:
            problems.append(f"line {i}: H-number {h} collides with v1 reserved range")
        if not doi:
            problems.append(f"line {i}: empty DOI for {h}")

    hc = Counter((r.get("h_number") or "").strip() for r in rows)
    dc = Counter((r.get("doi") or "").strip().lower() for r in rows)
    for h, n in hc.items():
        if n > 1 and h:
            problems.append(f"duplicate h_number: {h} ×{n}")
    for d, n in dc.items():
        if n > 1 and d:
            problems.append(f"duplicate DOI: {d} ×{n}")

    nums = sorted(int(r["h_number"][1:]) for r in rows if PATTERN.match((r.get("h_number") or "").strip()))
    span = f"{nums[0]}..{nums[-1]}" if nums else "none"
    gaps = [n for n in range(nums[0], nums[-1] + 1) if n not in nums] if nums else []

    print(f"rows: {len(rows)}")
    print(f"H-number span: {span}")
    print(f"gaps: {gaps if gaps else 'none'}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nCLEAN — no collisions, formats valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
