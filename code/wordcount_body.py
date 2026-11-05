#!/usr/bin/env python3
"""wordcount_body.py — body word count for the canonical manuscript (v1).

The D189 deep cut was stamped with an ad-hoc session counter ("6,800
math-removed") that was never preserved as a script or a stored output, so
the historical stamp cannot be reproduced from the tree. This tool replaces
it with a defined, re-runnable instrument and prints the count on every
scope the journal checklist needs.

Body scope: Sections 1-5 of manuscript/latex/main_rev2a.tex (Introduction
through Conclusion). The abstract, the keyword block, the end-matter
(availability, ethics, CRediT, AI declaration), the reference list and the
supplementary material are excluded, matching "body excluding refs/abstract".

Measures (each printed with its scope):
  prose_math_removed   prose only (no tables, captions, algorithm, display
                       equations); inline and display math removed
  prose_math_as_words  the same prose with math content kept and counted
                       as words
  body_math_removed    the whole body including table bodies, figure
                       captions and the algorithm, math removed
  body_math_as_words   the same full body with math content kept and counted
                       as words (the same basis as prose_math_as_words)

Word = whitespace-delimited token containing a letter or a digit, after
LaTeX markup is stripped; hyphenated compounds count as one word.

Exit code 0 always (this is a report, not a gate).

Run: python3 code/wordcount_body.py [--tex manuscript/latex/main_rev2a.tex]
"""
import argparse
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_TEX = os.path.join(ROOT, "manuscript", "latex", "main_rev2a.tex")


def strip_frontmatter(t):
    i = t.find("\\begin{document}")
    t = t[i + len("\\begin{document}"):] if i >= 0 else t
    j = t.find("\\bibliography")
    t = t[:j] if j > 0 else t
    for env in ("abstract", "keyword"):
        m = re.search(r"\\begin\{" + env + r"\}.*?\\end\{" + env + r"\}", t, re.S)
        if m:
            t = t[:m.start()] + t[m.end():]
    return t


def body_span(t):
    """Sections 1-5: Introduction .. end of Conclusion."""
    i = t.find("\\section{Introduction}")
    if i < 0:
        i = 0
    j = t.find("\\section*{Supplementary material}")
    if j < 0:
        j = t.find("\\end{document}")
    return t[i:j if j > 0 else len(t)]


def clean(t, math, prose_only):
    if prose_only:
        t = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", t, flags=re.S)
        t = re.sub(r"\\caption\{.*?\}", " ", t, flags=re.S)
        t = re.sub(r"\\begin\{algorithm\}.*?\\end\{algorithm\}", " ", t, flags=re.S)
        t = re.sub(r"\\begin\{equation\}.*?\\end\{equation\}", " ", t, flags=re.S)
    if math == "removed":
        t = re.sub(r"\$\$.*?\$\$", " ", t, flags=re.S)
        t = re.sub(r"\$[^$]*\$", " ", t)
    else:
        t = re.sub(r"\$([^$]*)\$", r" \1 ", t)
    # citations and cross-references: drop the keys, keep no words
    t = re.sub(r"\\(cite[a-z]*|ref|label|nocite)\s*(\[[^\]]*\])?\{[^}]*\}", " ", t)
    # environments and commands
    t = re.sub(r"\\begin\{[a-z*]+\}(\[[^\]]*\])?", " ", t)
    t = re.sub(r"\\end\{[a-z*]+\}", " ", t)
    t = re.sub(r"\\(includegraphics|documentclass|usepackage|graphicspath|biboptions|"
               r"affiliation|author|frontmatter|title|keyword|bibliographystyle|"
               r"bibliography|emergencystretch|SetAlgoLined|DontPrintSemicolon|"
               r"KwIn|KwOut|ForEach|ElseIf|Else|EndFor|EndWhile|Return|State|Print|"
               r"Input|Output|Function|Procedure)\b", " ", t)
    t = re.sub(r"\\[a-zA-Z]+\s*(\[[^\]]*\])?", " ", t)
    t = re.sub(r"[{}$&\\]", " ", t)
    return t


def words(t):
    return [w for w in t.split() if re.search(r"[A-Za-z0-9]", w)]


def measure(tex_path):
    if not os.path.exists(tex_path):
        print(f"[wordcount-body] SKIPPED — {tex_path} is not in this tree (the "
              f"manuscript sources are not part of the public code mirror)")
        return 0
    raw = open(tex_path, encoding="utf-8").read()
    body = body_span(strip_frontmatter(raw))
    out = {}
    for label, math, prose_only in (
            ("prose_math_removed", "removed", True),
            ("prose_math_as_words", "words", True),
            ("body_math_removed", "removed", False),
            ("body_math_as_words", "words", False)):
        out[label] = len(words(clean(body, math, prose_only)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tex", default=DEFAULT_TEX)
    args = ap.parse_args()
    path = args.tex if os.path.isabs(args.tex) else os.path.join(ROOT, args.tex)
    res = measure(path)
    print(f"[wordcount-body] {os.path.relpath(path, ROOT)}")
    for k, v in res.items():
        print(f"  {k:22s} {v:6d}")
    print("  scope: Sections 1-5 (Introduction..Conclusion), excluding the abstract,")
    print("         keyword block, end-matter, references and supplementary material.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
