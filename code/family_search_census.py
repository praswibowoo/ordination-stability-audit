#!/usr/bin/env python3
"""family_search_census.py — persist the exhaustive family-search counts.

The manuscript reports that the F6 search returned 34,206 rule-compliant
subsets (21,041 maximal). `explore_families.py` computed those counts but
printed them only, so the numbers had no reproducible artifact. This script
uses the same rule (3-5 parameters, at least 10 data rows complete for every
parameter, all 44 parameters) and writes the census.

Maximality is exact for sizes 3-5: a subset is maximal iff none of its
one-parameter extensions is itself rule-compliant. That is equivalent to the
`explore_families.py` pairwise-superset test but avoids its O(V^2) scan.

Outputs
-------
results/family_search_census.csv   every rule-compliant subset
results/family_search_summary.csv  counts and rule constants

Run: python3 code/family_search_census.py
"""
from __future__ import annotations

import csv
import itertools
import os
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRES = os.path.join(ROOT, "docs/extraction/stageA/matrix_presence_v2.csv")
RESULTS = os.path.join(ROOT, "results")
MIN_PARAMS, MAX_PARAMS, MIN_ROWS = 3, 5, 10


def load():
    with open(PRES, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    params = list(rows[0].keys())[7:51]
    own = ""
    data = []
    for r in rows:
        if r["batch_id"].strip():
            own = r["batch_id"]
        if r["row_kind"] != "data":
            continue
        mask = 0
        for j, p in enumerate(params):
            if r[p].strip() in ("Y", "NE"):
                mask |= 1 << j
        data.append((mask, own))
    return params, data


def main() -> int:
    params, data = load()
    n = len(data)
    # Per-parameter bitsets over rows: bit i is set when row i has the parameter.
    col = [0] * len(params)
    for i, (mask, _) in enumerate(data):
        for j in range(len(params)):
            if mask >> j & 1:
                col[j] |= 1 << i

    # key = parameter bitmask; value = (size, row bitset of complete rows)
    valid: dict[int, tuple[int, int]] = {}
    for k in range(MIN_PARAMS, MAX_PARAMS + 1):
        for combo in itertools.combinations(range(len(params)), k):
            key = 0
            rows_mask = (1 << n) - 1
            for j in combo:
                key |= 1 << j
                rows_mask &= col[j]
            if rows_mask.bit_count() >= MIN_ROWS:
                valid[key] = (k, rows_mask)

    maximal = []
    for key, (k, _) in valid.items():
        if k == MAX_PARAMS:
            maximal.append(key)
            continue
        if not any((key | (1 << j)) in valid
                   for j in range(len(params)) if not (key >> j & 1)):
            maximal.append(key)

    def describe(key: int) -> tuple[int, int, str]:
        rows_mask = (1 << n) - 1
        for j in range(len(params)):
            if key >> j & 1:
                rows_mask &= col[j]
        hits = [i for i in range(n) if rows_mask >> i & 1]
        papers = len({data[i][1] for i in hits})
        names = [params[j] for j in range(len(params)) if key >> j & 1]
        return len(hits), papers, "; ".join(names)

    os.makedirs(RESULTS, exist_ok=True)
    maxset = set(maximal)
    out = os.path.join(RESULTS, "family_search_census.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["size", "rows", "papers", "maximal", "parameters"])
        for key in sorted(valid):
            k, _ = valid[key]
            rows_n, papers, names = describe(key)
            w.writerow([k, rows_n, papers, int(key in maxset), names])
    summ = os.path.join(RESULTS, "family_search_summary.csv")
    with open(summ, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["min_params", "max_params", "min_rows", "parameters",
                    "compliant_subsets", "maximal_subsets"])
        w.writerow([MIN_PARAMS, MAX_PARAMS, MIN_ROWS, len(params),
                    len(valid), len(maximal)])
    by_size = Counter(v[0] for v in valid.values())
    print(f"[family-search] {len(valid)} compliant / {len(maximal)} maximal "
          f"| sizes {dict(sorted(by_size.items()))}")
    print(f"[family-search] wrote {os.path.relpath(out, ROOT)} + summary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
