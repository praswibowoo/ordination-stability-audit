#!/usr/bin/env python3
"""check_si_refs.py — guard the hand-written S-numbers in the main text.

Why this exists
---------------
The main text points at supplementary floats with literal strings ("Table S7",
"Figs. S2--S6") rather than \\ref, because the two documents compile separately
and the SI numbers its floats manually (S12a/S12b use \\renewcommand{\\thetable}).
LaTeX therefore cannot warn when the SI is reordered: the string "Table S7"
would silently point at a different table.

What it checks
--------------
1. Every S-token cited in main_rev2.tex resolves to an SI float of the same
   kind (a "Table S7" must match \\label{tab:S7}, a "Fig. S2" \\label{fig:S2}).
   Ranges ("S2--S6") and enumerations ("S12a and S12b") are expanded.
2. Every SI float label is cited at least once in the main text (no orphans).
3. SI document order is ascending within each float kind
   (S1 < ... < S12a < S12b < S13 < S14).

Scope: main_rev2a.tex (default; override with --main) and supplementary_main.tex
(the submission pair), plus a label-parity check against supplementary_rev2.tex
(the prose source the submission SI is hand-extracted from).
Read-only. Exit 1 on any finding.
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAIN = os.path.join(ROOT, "manuscript", "latex", "main_rev2a.tex")
SI = os.path.join(ROOT, "manuscript", "latex", "supplementary_main.tex")
SRC = os.path.join(ROOT, "manuscript", "latex", "supplementary_rev2.tex")

TOKEN = re.compile(r"S(\d{1,2})([ab])?")
LABEL = re.compile(r"\\label\{(tab:S[^}]*|fig:S[^}]*)\}")
CITE = re.compile(
    r"(Tables?|Figs?\.?|Figures?)\s*~?\s*"
    r"(S\d{1,2}[ab]?(?:(?:\s*(?:,|and|--|\u2013|-)\s*)+S\d{1,2}[ab]?)*)"
)


def key(name: str):
    m = re.fullmatch(r"S(\d{1,2})([ab])?", name)
    if not m:
        return None
    return (int(m.group(1)), m.group(2) or "")


def cited_refs(main_tex: str) -> set[str]:
    """Full labels ("tab:S7") cited in the main text, ranges expanded."""
    cited: set[str] = set()
    for m in CITE.finditer(main_tex):
        kind = "tab" if m.group(1).startswith("Table") else "fig"
        body = m.group(2)
        toks = [(t.start(), t.end(), t.group(0)) for t in TOKEN.finditer(body)]
        for i, (_, end, name) in enumerate(toks):
            cited.add(f"{kind}:{name}")
            if i + 1 < len(toks):
                nxt_start, _, nxt = toks[i + 1]
                sep = body[end:nxt_start]
                if re.fullmatch(r"\s*(?:--|\u2013|-)\s*", sep):
                    lo, hi = key(name), key(nxt)
                    if lo and hi and lo[1] == "" and hi[1] == "" and lo[0] < hi[0]:
                        for n in range(lo[0] + 1, hi[0]):
                            cited.add(f"{kind}:S{n}")
    return cited


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=MAIN,
                    help="main manuscript .tex (default: main_rev2a.tex)")
    args = ap.parse_args()
    main_path = args.main if os.path.isabs(args.main) else os.path.join(ROOT, args.main)
    if not (os.path.exists(main_path) and os.path.exists(SI)):
        print("[si-refs] SKIPPED — manuscript sources are not part of this tree")
        return 0
    main_tex = open(main_path, encoding="utf-8").read()
    # The blanket "Figures S1--S8 and Tables S1--S20 are provided" sentence must
    # not satisfy every float; each SI item needs a specific in-text citation.
    main_tex = main_tex.split("\\section*{Supplementary material}")[0]
    si_tex = open(SI, encoding="utf-8").read()

    labels = LABEL.findall(si_tex)
    src_labels = LABEL.findall(open(SRC, encoding="utf-8").read()) if os.path.exists(SRC) else []
    if not labels:
        print("[si-refs] FAIL — no S-labels found in the supplementary")
        return 1

    cited = cited_refs(main_tex)

    def satisfied(name: str) -> bool:
        # a range such as "Tables S1--S14" covers the S12 family, which the SI
        # splits into S12a and S12b
        if name in labels:
            return True
        base = name[:-1] if name[-1] in "ab" and name[-2].isdigit() else name
        return any(re.fullmatch(re.escape(base) + r"[ab]", lab) for lab in labels)

    missing = sorted(c for c in cited if not satisfied(c))

    def base_label(lab: str) -> str:
        # S12a/S12b collapse to S12; S16 must NOT collapse to S1.
        kind, name = lab.split(":", 1)
        if len(name) > 2 and name[-1] in "ab" and name[-2].isdigit():
            name = name[:-1]
        return f"{kind}:{name}"

    orphans = [lab for lab in labels if lab not in cited and base_label(lab) not in cited]

    out_of_order = []
    for kind in ("fig", "tab"):
        seq = [lab.split(":", 1)[1] for lab in labels if lab.startswith(kind + ":")]
        keys = [key(n) for n in seq]
        if keys != sorted(keys):
            out_of_order.append((kind, seq))

    parity_bad = sorted(set(src_labels) ^ set(labels))

    # Rows labelled "S1 & ...", "S2 & ..." inside an SI table collide with the
    # S-float numbering ("Table S1"). Panel labels use V1/V2/V3 instead.
    row_collisions = []
    for path, text in (("source", open(SRC, encoding="utf-8").read() if os.path.exists(SRC) else ""),
                       ("submission", si_tex)):
        for n in re.findall(r"(?m)^S\d+\s*&", text):
            row_collisions.append((path, n.strip()))

    if missing or orphans or out_of_order or parity_bad or row_collisions:
        for c in missing:
            print(f"[si-refs] FAIL — main cites {c}, no such SI label")
        for lab in orphans:
            print(f"[si-refs] FAIL — SI float {lab} is never cited in the main text")
        for kind, seq in out_of_order:
            print(f"[si-refs] FAIL — {kind} labels out of order: {seq}")
        for lab in parity_bad:
            print(f"[si-refs] FAIL — label {lab} is not in both SI source and submission SI")
        for path, tok in row_collisions:
            print(f"[si-refs] FAIL — {path} SI table row starts with '{tok}', which collides with S-float numbering")
        return 1

    print(
        f"[si-refs] OK — {len(cited)} S-references resolve; "
        f"{len(labels)} SI floats all cited; document order ascending; "
        f"source/submission labels match ({len(src_labels)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
