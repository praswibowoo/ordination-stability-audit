#!/usr/bin/env python3
"""Duplicate-phrase gate (D196; D207: canonical tex scope).

Guards the "same sentence in two copies" failure class (D187/D191/D195): a
sentence is authored once but rendered in multiple layers (main drafts, SI
extended analyses, table notes, captions), and a later fix touches only one copy.

D207: the md->tex pipeline is retired; the scan now reads the canonical
hand-edited pair `manuscript/latex/main_rev2a.tex` +
`manuscript/latex/supplementary_rev2.tex` through a tex-aware prose view
(float bodies, algorithms, math and command scaffolding removed) plus the
builder-generated `manuscript/tables/*.md` (still the guarded table-note layer).

Rule A (hard): a normalized sentence must not appear twice in the SAME file.
Rule B (report): the same normalized sentence appearing in TWO OR MORE files
is listed; designated capsule mirrors can be allowlisted in ALLOW below with a
recorded justification, everything else is treated as accidental duplication.

Normalization: lowercase, strip citations and math, mask all numbers, drop
punctuation. Sentences < 8 tokens after masking are ignored.

Run: python3 code/check_dup.py            (make style / make all)
     python3 code/check_dup.py --report   (list all cross-file groups, never fails)
"""
from __future__ import annotations

import argparse
import collections
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABBR = ["e.g.", "i.e.", "et al.", "cf.", "Fig.", "Figs.", "vs.", "Sec.", "No.", "Eq.", "approx.", "Dr.", "St."]
CITATION = re.compile(r"\[@[^\]]*\]")
MATH = re.compile(r"\$[^$]*\$")
NUM = re.compile(r"\d[\d,\.]*")

# Deliberate cross-file mirrors (capsule register). Add with a recorded reason only.
ALLOW: dict[str, str] = {}


def tex_view(text: str) -> str:
    """Tex-aware prose view for the canonical manuscript files (D207).

    Drops the preamble, float bodies (figures, tables incl. their notes,
    algorithms, tikz), display math and command scaffolding, keeping the
    prose lines only. Line numbers in reports are relative to this view.
    """
    if "\\begin{document}" in text:
        text = text.split("\\begin{document}", 1)[1]
    for stop in ("\\bibliography{", "\\section*{Supplementary material}"):
        j = text.find(stop)
        if j >= 0:
            text = text[:j]
    text = re.sub(r"\\begin\{(figure|table|algorithm|tikzpicture)\}.*?\\end\{\1\}", " ", text, flags=re.S)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.S)
    text = re.sub(r"\$[^$]*\$", " ", text)
    text = re.sub(r"\\[a-zA-Z@]+\*?", " ", text)   # command names; brace content stays
    return text


def sources(main: str = "manuscript/latex/main_rev2a.tex") -> list[tuple[str, int, str]]:
    files = [os.path.join(ROOT, main),
             os.path.join(ROOT, "manuscript/latex/supplementary_rev2.tex")]
    files += sorted(glob.glob(os.path.join(ROOT, "manuscript/tables/*.md")))
    out = []
    for path in files:
        if not os.path.exists(path):
            continue          # the manuscript sources are not part of this tree
        rel = os.path.relpath(path, ROOT)
        # Cross-platform: os.path.relpath returns backslashes on Windows, which
        # made the "manuscript/tables/" branch below unreachable; every table row
        # was then scanned as prose and the S12a/S12b shared rows failed the gate.
        rel = rel.replace(os.sep, "/")
        if rel.endswith(".tex"):
            for ln, line in enumerate(tex_view(open(path, encoding="utf-8").read()).splitlines(), 1):
                s = line.strip()
                if s:
                    out.append((rel, ln, s))
            continue
        raw = open(path, encoding="utf-8").read().splitlines()
        start = 0
        for ln, line in enumerate(raw[start:], start + 1):
            s = line.strip()
            if not s:
                continue
            if re.match(r"^(paper|title|status|authors|target|style|word count)\s*:", s, re.I):
                continue
            if rel.startswith("manuscript/tables/"):
                if not s.startswith("_"):        # table notes only
                    continue
            elif s.startswith(("#", "**", "<!--")) and "caption" not in rel:
                continue
            if "wordcount.py" in s:
                continue
            out.append((rel, ln, s))
    return out


def sentences(text: str):
    t = text
    for a in ABBR:
        t = t.replace(a, a.replace(".", "\x01"))
    parts = re.split(r"(?<=[.!?])\s+", t)
    return [p.replace("\x01", ".") for p in parts if p.strip()]


def norm(s: str) -> str:
    s = s.lower()
    s = CITATION.sub(" ", s)
    s = MATH.sub(" ", s)
    s = NUM.sub("#", s)
    s = re.sub(r"[^a-z#\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--main", default="manuscript/latex/main_rev2a.tex")
    ap.add_argument("--report", action="store_true")
    args, _ = ap.parse_known_args()
    if not os.path.exists(os.path.join(ROOT, args.main)):
        print("[dup] SKIPPED — the manuscript sources are not part of this tree "
              "(the public code archive ships no LaTeX manuscript); the generated "
              "tables are still checked for duplication")
    report = args.report
    groups: dict[str, list[tuple[str, int, str]]] = collections.defaultdict(list)
    for rel, ln, line in sources(args.main):
        for sent in sentences(line):
            key = norm(sent)
            if len(key.split()) >= 8:
                groups[key].append((rel, ln, " ".join(sent.split())[:110]))
    same_file, cross_file = [], []
    for key, members in groups.items():
        if len(members) < 2:
            continue
        files = {m[0] for m in members}
        if len(files) == 1:
            same_file.append((key, members))
        elif key not in ALLOW:
            cross_file.append((key, members))
    # phrase layer: 12-gram windows across files (warnings)
    phr: dict[str, set] = collections.defaultdict(set)
    windows: dict[str, list] = collections.defaultdict(list)
    for rel, ln, line in sources(args.main):
        for sent in sentences(line):
            toks = norm(sent).split()
            for i in range(len(toks) - 11):
                key = " ".join(toks[i:i + 12])
                phr[key].add(rel)
                if len(windows[key]) < 4:
                    windows[key].append((rel, ln, " ".join(sent.split())[:90]))
    phrase_groups = {k: v for k, v in windows.items() if len(phr[k]) >= 2}
    if report:
        print(f"[dup --report] same-file groups: {len(same_file)} | cross-file sentences (non-allowlisted): {len(cross_file)} | cross-file 12-gram phrases: {len(phrase_groups)}")
        n = 0
        for key, members in phrase_groups.items():
            n += 1
            if n > 30:
                break
            print(f"\nPHRASE-12: {key}")
            for rel, ln, prev in members:
                print(f"   {rel}:{ln}: {prev}")
        return 0
        for tag, lst in (("SAME-FILE", same_file), ("CROSS-FILE", cross_file)):
            for key, members in lst:
                print(f"\n{tag}: {key[:96]}")
                for rel, ln, prev in members:
                    print(f"   {rel}:{ln}: {prev}")
        return 0
    problems = list(same_file) + list(cross_file)
    for key, members in cross_file:
        print(f"[dup] CROSS-FILE (not allowlisted): {key[:90]}")
        for rel, ln, prev in members:
            print(f"      {rel}:{ln}: {prev}")
    if same_file:
        print(f"[dup] {len(same_file)} SAME-FILE duplicate group(s) — hard fail")
        for key, members in same_file:
            print(f"  {key[:90]}")
            for rel, ln, prev in members:
                print(f"      {rel}:{ln}: {prev}")
    if problems:
        return 1
    print("[dup] OK — 0 same-file duplicates; 0 non-allowlisted cross-file sentence duplicates")
    return 0


if __name__ == "__main__":
    sys.exit(main())
