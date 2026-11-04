#!/usr/bin/env python3
"""House-style gate for the canonical manuscript (D192; D207: tex scope).

Guards the regression class found in the D192 audit: readability/deep-cut edits
re-introduced prose semicolons after the D175-D181 passes had cleared them.

D207: scope is the main manuscript body (abstract + Sections 1-5) of the
canonical hand-edited source `manuscript/latex/main_rev2a.tex`, scanned as a
tex-aware prose view. Exempt registers established in earlier D-rows: float
captions, table bodies, the algorithm block (its `\\;` are line terminators),
math spans, and back matter (declarations, CRediT, availability statements).

Checks:
  1. prose semicolons outside the exempt registers
  2. `\\textbf{}` bold emphasis left in body prose
  3. balance of parentheses per prose line

Line numbers in reports are relative to the scanned scope.
Run: python3 code/check_style.py   (also invoked by `make style` / `make all`)
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = "manuscript/latex/main_rev2a.tex"
START = "\\begin{abstract}"
END = "\\section*{Supplementary material}"


def strip_math(text: str) -> str:
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    return re.sub(r"\$[^$]*\$", lambda m: " " * len(m.group(0)), text)


def strip_braced(text: str, cmd: str) -> str:
    """Remove `cmd{...}` including balanced-brace content."""
    out, i = [], 0
    while True:
        j = text.find(cmd + "{", i)
        if j < 0:
            out.append(text[i:])
            break
        out.append(text[i:j])
        k = j + len(cmd) + 1
        depth = 1
        while k < len(text) and depth:
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
            k += 1
        i = k
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--main", default=TEX)
    args, _ = ap.parse_known_args()
    tex = args.main
    if not os.path.exists(os.path.join(ROOT, tex)):
        print(f"[style] SKIPPED — {tex} is not in this tree (the manuscript sources "
              f"are not part of the public code archive)")
        return 0
    raw = open(os.path.join(ROOT, tex), encoding="utf-8").read()
    a = raw.find(START)
    b = raw.find(END)
    if a < 0 or b < 0:
        print(f"[style] FAIL — scope markers not found in {tex}")
        return 1
    body = raw[a:b]
    body = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", " ", body, flags=re.S)
    body = strip_braced(body, "\\caption")
    body = strip_math(body)

    problems = []
    for ln, line in enumerate(body.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("\\"):
            continue
        for sm in re.finditer(";", line):
            ctx = line[max(0, sm.start() - 60):sm.start() + 60].strip()
            problems.append(f"{tex}:{ln} (scope-relative): prose semicolon … {ctx} …")
        for bm in re.finditer(r"\\textbf\{", line):
            problems.append(f"{tex}:{ln}: bold emphasis … {line[bm.start():bm.start() + 60]} …")
        if line.count("(") != line.count(")"):
            problems.append(
                f"{tex}:{ln}: unbalanced parentheses ({line.count('(')} open / {line.count(')')} close)"
            )
    if problems:
        print("[style] FAIL")
        for p in problems:
            print("  " + p)
        return 1
    print(f"[style] OK — {tex} body (abstract + Sections 1-5): "
          "0 prose semicolons, 0 bold emphasis, balanced parentheses")
    return 0


if __name__ == "__main__":
    sys.exit(main())
