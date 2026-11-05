#!/usr/bin/env python3
"""check_highlights.py — the Elsevier Highlights gate.

Environmental Modelling & Software (Elsevier) asks for 3 to 5 Highlights, each at most
85 characters including spaces, as a separate submission file. The bullets quote numbers the claims checker already
verifies, so the risk this gate closes is editorial drift: a bullet is reworded, grows past
the limit, or the character count in the record stops matching the sentence.

Reads `manuscript/highlights.md` (a table with the bullet, its stated count, and its source),
checks the count and the limit, and compares each bullet's characters against the number the
file claims. Skips, rather than fails, when the manuscript sources are absent, which is the
case inside the public code mirror.

Run: python3 code/check_highlights.py     (also part of `make check`)
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "manuscript", "highlights.md")
LIMIT = 85
MIN, MAX = 3, 5


def main() -> int:
    if not os.path.exists(PATH):
        print("[highlights] SKIPPED — the manuscript sources are not part of this tree "
              "(the public code mirror ships no LaTeX manuscript)")
        return 0

    rows = []
    for line in open(PATH, encoding="utf-8"):
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not cells[0].isdigit():
            continue
        rows.append((int(cells[0]), cells[1], cells[2]))

    fails = []
    if not MIN <= len(rows) <= MAX:
        fails.append(f"{len(rows)} bullets, the journal asks for {MIN} to {MAX}")
    for n, text, stated in rows:
        chars = len(text)
        if chars > LIMIT:
            fails.append(f"bullet {n} is {chars} characters, the limit is {LIMIT}")
        m = re.search(r"\d+", stated)
        if m and int(m.group()) != chars:
            fails.append(f"bullet {n} states {m.group()} characters but has {chars}")
        if text.endswith(".") and not text[0].isupper():
            fails.append(f"bullet {n} starts in lower case")

    if fails:
        for f in fails:
            print(f"FAIL  {f}")
        print(f"[highlights] {len(fails)} problem(s)")
        return 1
    print(f"[highlights] OK — {len(rows)} bullets, longest {max(len(t) for _, t, _ in rows)} "
          f"characters of {LIMIT}, stated counts match")
    return 0


if __name__ == "__main__":
    sys.exit(main())
