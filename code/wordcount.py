#!/usr/bin/env python3
"""wordcount.py — abstract word count (P8F, D173; D207: reads the canonical tex).

Counts the abstract body of manuscript/latex/main_rev2a.tex under a
documented, reproducible, deliberately conservative method.

The target is venue-specific: Environmental Modelling & Software caps the
abstract at 150 words (guide for authors, verified 2026-11-04). Change TARGET
when the target journal changes.

  1. Body = text between \\begin{abstract} and \\end{abstract}.
  2. Markdown emphasis markers (** * `) removed; LaTeX math spans ($...$) kept
     with delimiters removed (inner commands/braces left as-is, so each math
     token still counts at least once).
  3. Two counts are reported:
       strict    = every whitespace-separated token (Word-like: operators count)
       alnum     = tokens containing at least one letter or digit
  4. The TARGET gate is applied to the STRICT count (conservative upper bound).

Exit code 1 if strict > TARGET. Cite this output in the abstract footer and
manuscript/journal_checklist.md.
"""
from __future__ import annotations
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABSTRACT = os.path.join(ROOT, "manuscript", "latex", "main_rev2a.tex")
TARGET = 150  # EMS abstract limit; was 250 for Ecological Informatics


def extract_body(text: str) -> str:
    a = text.find("\\begin{abstract}")
    b = text.find("\\end{abstract}")
    if a < 0 or b < 0:
        raise SystemExit("abstract environment not found")
    return text[a + len("\\begin{abstract}"):b]


def clean(body: str) -> str:
    body = body.replace("\\%", "%")                 # LaTeX percent -> %
    body = re.sub(r"`([^`]*)`", r"\1", body)        # code spans -> content
    body = re.sub(r"\*\*([^*]+)\*\*", r"\1", body)  # bold
    body = re.sub(r"\*([^*]+)\*", r"\1", body)      # emphasis
    body = re.sub(r"\$([^$]*)\$", r" \1 ", body)    # keep math content
    return body


def main() -> int:
    if not os.path.exists(os.path.join(ROOT, ABSTRACT)):
        print("[wordcount] SKIPPED — the manuscript sources are not part of this tree "
              "(the public code mirror ships no LaTeX manuscript)")
        return 0
    text = open(ABSTRACT, encoding="utf-8").read()
    body = clean(extract_body(text))
    tokens = body.split()
    strict = len(tokens)
    alnum = sum(1 for t in tokens if re.search(r"[A-Za-z0-9]", t))
    print(f"abstract words: strict(token-conservative) = {strict} | alnum-words = {alnum} | target <= {TARGET}")
    if strict > TARGET:
        print(f"[FAIL] strict count {strict} exceeds target {TARGET}")
        return 1
    print(f"[PASS] strict count {strict} <= {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
