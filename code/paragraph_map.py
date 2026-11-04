#!/usr/bin/env python3
"""paragraph_map.py — index the manuscript paragraph by paragraph, for line-by-line review.

For each paragraph of Sections 1 to 5 it reports: an identifier (S3.4-2), the word count on
two measures, the sentence count, the cross-references it makes, and the numbers the claims
checker pins to the manuscript. It also flags the mechanical defects a reading pass looks for
(over-long sentences, over-long paragraphs, repeated sentence openers, evaluative "-ing"
tails, comparative density), so the reviewer spends attention on content rather than on
counting.

Not a gate. Writes `docs/paragraph_map.md` and prints a section summary.

Run: python3 code/paragraph_map.py
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEX = os.path.join(ROOT, "manuscript", "latex", "main_rev2a.tex")
OUT = os.path.join(ROOT, "docs", "paragraph_map.md")

spec = importlib.util.spec_from_file_location(
    "wcb", os.path.join(ROOT, "code", "wordcount_body.py"))
wcb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wcb)

FLOATS = ("table", "figure", "algorithm", "tabular", "tabularx")


def strip_floats(t: str) -> tuple[str, list[str]]:
    """Replace display environments with a marker line, so paragraph order survives."""
    marks = []

    def repl(m):
        kind = re.match(r"\\begin\{(\w+)\}", m.group(0)).group(1)
        label = re.search(r"\\label\{([^}]*)\}", m.group(0))
        marks.append(f"[{kind} float, label {label.group(1) if label else 'none'}]")
        return "\n\n" + marks[-1] + "\n\n"

    for env in FLOATS:
        t = re.sub(r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}", repl, t,
                   flags=re.S)
    return t, marks


def pinned_values() -> set[str]:
    """Numbers the checker pins to the manuscript, from its own verbose run."""
    try:
        out = subprocess.run([sys.executable, os.path.join(ROOT, "code", "check_numbers.py"),
                              "-v"], capture_output=True, text=True, cwd=ROOT).stdout
    except Exception:
        return set()
    vals = set()
    for line in out.splitlines():
        if not line.startswith("PASS"):
            continue
        if not re.search(r"main_rev2a?\.tex", line):
            continue
        m = re.search(r"=\s*([-\d.]+)", line)
        if m:
            v = m.group(1)
            vals.add(v)
            # the checker prints counts as floats ("761.0"); the prose writes "761"
            if re.fullmatch(r"-?\d+\.0", v):
                vals.add(v.split(".")[0])
    return vals


def sentences(p: str) -> list[str]:
    body = re.sub(r"\$[^$]*\$", " M ", p)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(\\])", body)
    return [s.strip() for s in parts if len(wcb.words(s)) >= 3]


def main() -> int:
    if not os.path.exists(TEX):
        print("[paragraph-map] SKIPPED — manuscript sources are not part of this tree")
        return 0
    raw = open(TEX, encoding="utf-8").read()
    raw = "\n".join(l for l in raw.splitlines() if not l.lstrip().startswith("%"))
    body = wcb.body_span(wcb.strip_frontmatter(raw))
    body, _ = strip_floats(body)
    pins = pinned_values()

    rows, section, idx, sec_words = [], "front", 0, 0
    summary = []
    sec_no, sub_no = 0, 0
    NEUTRAL = {"including", "using", "following", "assuming", "corresponding", "according",
               "building", "testing", "applying", "meaning", "starting", "depending",
               "containing", "having", "being", "doing", "showing", "making", "taking"}
    for chunk in re.split(r"\n\s*\n", body):
        c = chunk.strip()
        if not c:
            continue
        m = re.match(r"\\(section|subsection)\{(.*?)\}", c)
        if m:
            section = m.group(2).strip()
            idx = 0
            if m.group(1) == "section":
                sec_no += 1
                sub_no = 0
            else:
                sub_no += 1
            summary.append([section, 0, 0])
            rows.append(["", f"**{m.group(1)}: {section}**", "", "", "", "", ""])
            continue
        if c.startswith("[") and c.endswith("]"):
            rows.append(["", f"*{c}*", "", "", "", "", ""])
            continue
        idx += 1
        w_words = len(wcb.words(wcb.clean(c, "words", True)))
        w_rem = len(wcb.words(wcb.clean(c, "removed", True)))
        sents = sentences(c)
        refs = re.findall(r"\\ref\{([^}]*)\}", c)
        cites = len(re.findall(r"\\cite[a-z]*\{", c))
        nums = sorted({v for v in pins if re.search(r"(?<![\d.])" + re.escape(v) + r"(?![\d])", c)})

        flags = []
        long_s = [len(wcb.words(s)) for s in sents if len(wcb.words(s)) > 35]
        if long_s:
            flags.append(f"{len(long_s)} sentence(s) over 35 words (max {max(long_s)})")
        if w_words > 170:
            flags.append(f"paragraph {w_words} words")
        opens = [s.split()[0] for s in sents]
        for o in dict.fromkeys(opens):
            if opens.count(o) >= 3:
                flags.append(f"opener '{o}' ×{opens.count(o)}")
        tail = [w for s_ in sents
                for w in [m2.group(1) for m2 in [re.search(r",\s+(\w+ing)\b[^.]*\.$", s_)] if m2]
                if w not in NEUTRAL]
        if tail:
            flags.append(f"evaluative tail '{tail[0]}'")
        piv = len(re.findall(r"rather than|not only|, not ", c))
        if piv >= 3:
            flags.append(f"{piv} comparatives")

        tag = (f"§{sec_no}-{idx}" if sub_no == 0 else f"§{sec_no}.{sub_no}-{idx}")
        rows.append([tag, c.split()[0:7] and
                     " ".join(wcb.words(re.sub(r"\$[^$]*\$", "M", c))[:7]) + " …",
                     str(w_words), str(w_rem), str(len(sents)),
                     ", ".join(dict.fromkeys(refs))[:40] or "—",
                     (f"{len(nums)} pinned: " + ", ".join(nums[:4])) if nums else "—",
                     "; ".join(flags) or "—"])
        summary[-1][1] += w_words
        summary[-1][2] += w_rem

    hdr = ("| id | opens with | words (math as words) | words (math removed) | sentences | refs | pinned numbers | flags |\n"
           "|---|---|---|---|---|---|---|---|\n")
    lines = ["# Manuscript paragraph map (Sections 1 to 5)", "",
             "Generated by `code/paragraph_map.py` for the paragraph-by-paragraph review pass.",
             "Word counts are per paragraph on two measures. `pinned numbers` are values the claims",
             "checker binds to this document; they are the ones an edit must not disturb.", "",
             hdr]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    lines += ["", "## Section totals", "", "| section | words (math as words) | words (math removed) |", "|---|---|---|"]
    for name, a, b in summary:
        if name != "front":
            lines.append(f"| {name} | {a} | {b} |")
    text = "\n".join(lines) + "\n"
    if "--check" in sys.argv:
        try:
            current = open(OUT, encoding="utf-8").read()
        except FileNotFoundError:
            print(f"[map-check] {os.path.relpath(OUT, ROOT)} is missing; run code/paragraph_map.py")
            return 1
        if current != text:
            print(f"[map-check] STALE — {os.path.relpath(OUT, ROOT)} no longer matches the "
                  f"manuscript; run python3 code/paragraph_map.py")
            return 1
        print("[map-check] OK — paragraph map matches the manuscript")
        return 0
    open(OUT, "w", encoding="utf-8").write(text)

    print(f"wrote {os.path.relpath(OUT, ROOT)}")
    for name, a, b in summary:
        if name != "front":
            print(f"  {name:52s} {a:5d} {b:5d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
