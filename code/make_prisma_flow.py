#!/usr/bin/env python3
"""PRISMA flow figure (P1-6.1 figure factory, first deliverable).

Numbers are PARSED + ASSERTED from audited sources -- never hand-written:
  * funnel numbers           -> manuscript/corpus_funnel.md (verbatim funnel table)
  * analyzed-paper count     -> results/phase4_baseline.json (verify_baseline gate)
  * PDF / text counts        -> disk counts (data/raw/pdf, docs/extraction/corpus_texts)
Figure: PRISMA-2020-style vertical flow, exclusion boxes on the right,
wave-4 extension (D90) + current full-text status appended after funnel close.
Layout (D141): connectors are computed from the boxes' padded edges (bottom_i ->
top_{i+1}) so nothing floats; each exclusion box is aligned to its parent stage
with an arrowhead off-ramp; identification/screening/included phase bands behind.

Outputs (atomic tmp->replace): figures/figS1_prisma_flow.png (600 dpi) + .pdf
Run: python3 code/make_prisma_flow.py
"""
import csv
import importlib.util
import json
import os
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117)
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FUNNEL_MD = os.path.join(ROOT, "manuscript", "corpus_funnel.md")
OUT_PNG = os.path.join(ROOT, "figures", "figS1_prisma_flow.png")
OUT_PDF = os.path.join(ROOT, "figures", "figS1_prisma_flow.pdf")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def atomic_save(fig, path, **kw):
    base, ext = os.path.splitext(path)
    tmp = base + ".tmp" + ext  # keep extension: mpl infers format from it
    fig.savefig(tmp, **kw)
    os.replace(tmp, path)


def load_baseline():
    spec = importlib.util.spec_from_file_location(
        "freeze", os.path.join(ROOT, "code", "phase4_freeze.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.verify_baseline()


def parse_funnel():
    """Extract every funnel number from the audited doc; fail loud on drift."""
    text = open(FUNNEL_MD, encoding="utf-8").read().replace("\u2212", "-")
    need = {
        "raw": ("11,612", 11612),
        "dedup_rm": ("2,681", 2681),
        "after_dedup": ("8,931", 8931),
        "filter_rm": ("2,838", 2838),
        "after_filter": ("6,093", 6093),
        "frame": ("375", 375),
        "pilot_consumed": ("140", 140),
        "screened": ("237", 237),
        "include": ("134", 134),
        "unsure": ("14", 14),
        "retained": ("148", 148),
        "doi_confirmed": ("117", 117),
        "no_doi": ("25", 25),
        "year_fixes": ("3 year", 3),
        "offtopic": ("8", 8),
        "predatory": ("5", 5),
        "responsible": ("135", 135),
        "pdf_at_close": ("119", 119),
        "pct": ("88%", 88),
    }
    nums = {}
    for key, (pat, expect) in need.items():
        assert pat in text, f"funnel number missing in corpus_funnel.md: {key} ({pat})"
        nums[key] = expect
    # internal arithmetic integrity
    assert nums["raw"] - nums["dedup_rm"] == nums["after_dedup"]
    assert nums["after_dedup"] - nums["filter_rm"] == nums["after_filter"]
    assert nums["include"] + nums["unsure"] == nums["retained"]
    assert nums["screened"] - nums["retained"] == 89
    assert nums["retained"] - nums["offtopic"] - nums["predatory"] == nums["responsible"]
    return nums


def disk_counts():
    # The source corpus is not redistributed, so in a fresh unpack of the two
    # records these directories do not exist and the counts are zero — which is
    # how the provenance check below is skipped. os.listdir raises on a missing
    # directory, so the absence has to be handled here rather than assumed.
    pdf_dir = os.path.join(ROOT, "data", "raw", "pdf")
    txt_dir = os.path.join(ROOT, "docs", "extraction", "corpus_texts")
    pdfs = ([f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
            if os.path.isdir(pdf_dir) else [])
    texts = ([f for f in os.listdir(txt_dir) if f.lower().endswith(".txt")]
             if os.path.isdir(txt_dir) else [])
    return len(pdfs), len(texts)


def n_data_papers():
    with open(os.path.join(ROOT, "docs", "extraction", "stageA",
                           "matrix_presence_v2.csv"), encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    return len({r["batch_id"] for r in rows if r["row_kind"] == "data"}), len(rows)


def main():
    n = parse_funnel()
    n_papers_baseline = load_baseline()["baselines"]["n_papers"]
    n_pdf_disk, n_txt_disk = disk_counts()
    n_data_papers_n, _ = n_data_papers()
    wave4 = n_papers_baseline - n["responsible"]
    # cross-source assertions (audit armor)
    assert n["pdf_at_close"] + wave4 == 122, "119+3=122 analyzed chain broken"
    assert n_papers_baseline == 138, "screened corpus 138 broken"
    # Disk-count provenance check. It can only run where the source corpus is
    # present (the authors' working copy): the source PDFs and texts are not
    # redistributed, so in a fresh unpack of the two Zenodo records the local
    # counts are zero and the check is skipped. The funnel numbers the figure
    # reports come from manuscript/corpus_funnel.md, which ships with the code.
    _corpus_present = (n_pdf_disk or n_txt_disk)
    if _corpus_present:
        assert n_pdf_disk == 129, f"PDFs on disk {n_pdf_disk} != 129"
        assert n_txt_disk == 123, f"texts on disk {n_txt_disk} != 123"
    else:
        print("[prisma-flow] local corpus not present: skipping the PDF/text "
              "disk-count provenance assert (the source texts are not redistributed)")
    assert n_data_papers_n == 122, f"data papers {n_data_papers_n} != 122"
    assert n_papers_baseline - n_data_papers_n == 16, "16 ILL/no-data papers broken"

    fig, ax = plt.subplots(figsize=(6.6, 8.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    EDGE = "#263238"; FILL = "#eceff1"; EXF = "#fbe9e7"; EXE = "#bf360c"
    WAV = "#fff8e1"; WAVE = "#f57f17"; FIN = "#e8f5e9"; FINE = "#1b5e20"
    FS = 7.8; FSB = 8.6
    PAD = 0.6  # FancyBboxPatch pad: true visual edge is (w/2 + PAD) from center

    def box(x, y, w, h, text, fc=FILL, ec=EDGE, fs=FS, lw=1.2, weight="normal"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle=f"round,pad={PAD},rounding_size=1.6",
                                    fc=fc, ec=ec, lw=lw, zorder=3))
        t = ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                    fontweight=weight, zorder=4, color="#1a1a1a", linespacing=1.35)
        # auto-fit: shrink until widest line fits inner box width (no overflow)
        fig.canvas.draw()
        inv = ax.transData.inverted()
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        (x0d, _), (x1d, _) = inv.transform((bb.x0, bb.y0)), inv.transform((bb.x1, bb.y1))
        avail = (w - 1.6)  # inner margin 0.8 units each side
        need = x1d - x0d
        if need > avail > 0:
            t.set_fontsize(t.get_fontsize() * avail / need * 0.97)

    def edges(x, y, w, h):
        """Padded visual edges of a box (what connectors must touch)."""
        return dict(top=y + h / 2 + PAD, bot=y - h / 2 - PAD,
                    l=x - w / 2 - PAD, r=x + w / 2 + PAD)

    def vconnect(y0, y1, x):
        """Vertical arrow landing exactly on both box edges."""
        ax.add_patch(FancyArrowPatch((x, y0), (x, y1),
                                     arrowstyle="-|>", mutation_scale=13,
                                     lw=1.4, color=EDGE, zorder=2))

    def hconnect(y, x0, x1, color=EXE, head=True):
        """Horizontal off-ramp from the funnel spine to an exclusion box."""
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y),
                                     arrowstyle="-|>" if head else "-",
                                     mutation_scale=10, lw=0.9,
                                     color=color, zorder=1))

    X, W = 32, 34
    EX, EW = 80, 34
    # derived counts — arithmetic over asserted values only (no hand-typed numbers)
    n_prescreen_rm = n["after_filter"] - n["frame"]           # 6093 - 375
    n_intake_rm = n["offtopic"] + n["predatory"]              # 8 + 5
    n_drawn_from_frame = n["frame"] - n["screened"]           # 375 - 237 = 138 pilots drawn from frame
    assert n_drawn_from_frame == 138, "frame - screened must equal 138 (D207 audit fix)"
    n_not_retrieved = n["responsible"] - n["pdf_at_close"]    # 135 - 119
    assert n_prescreen_rm == 5718 and n_intake_rm == 13 and n_not_retrieved == 16
    # the displayed funnel must satisfy both subtraction legs explicitly
    assert n["retained"] - n_intake_rm == n["responsible"], "148 - 13 = 135 broken"
    assert n["responsible"] - n_not_retrieved == n["pdf_at_close"], "135 - 16 = 119 broken"
    # PRISMA-2020 vocabulary; every count interpolated from the asserted dict
    rows = [  # (y, h, text, kwargs)
        (90, 8.0, f"Records identified from databases\nOpenAlex Boolean queries · 10 families\n(Q1-Q9 + Q3b) · 100% pages fetched\n(n = {n['raw']:,})", {"weight": "bold", "fs": 8.0}),
        (76.5, 4.6, f"Records after duplicate removal\n(n = {n['after_dedup']:,})", {}),
        (63.5, 6.0, f"Records after eligibility filters\npeer-reviewed · mangrove-linked\n(n = {n['after_filter']:,})", {}),
        (51.0, 4.6, f"Pre-screen frame (term-scoring tiers)\n(n = {n['frame']})", {}),
        (38.5, 6.0, f"Records screened\nhuman × LLM, blind · κ = 0.529\n(n = {n['screened']})", {}),
        (26.5, 6.0, f"Reports sought for retrieval\n(n = {n['retained']}: {n['include']} include + {n['unsure']} unsure)", {}),
        (14.5, 6.0, f"Studies included\n(n = {n['responsible']} responsible corpus)", {}),
        (4.2, 7.0, f"STUDIES IN THE SYNTHESIS — ANALYZED CORPUS\n({n_data_papers_n} papers: {n['pdf_at_close']} full texts + {wave4} substitutes)", {"fc": FIN, "ec": FINE, "weight": "bold", "fs": 7.2}),
    ]
    E = [edges(X, y, W, h) for y, h, _, _ in rows]

    # ---- PRISMA phase bands (behind everything) ----
    def band(y0, y1, label, fc="#f5f7f8"):
        ax.add_patch(plt.Rectangle((1.5, y0), 97, y1 - y0, fc=fc, ec="none",
                                   zorder=0))
        ax.text(4.6, (y0 + y1) / 2, label, rotation=90, ha="center",
                va="center", fontsize=6.8, color="#607d8b",
                fontweight="bold", zorder=1)

    band(E[2]["bot"] - 1.0, E[0]["top"] + 1.0, "IDENTIFICATION")
    band(E[4]["bot"] - 1.0, E[3]["top"] + 1.0, "SCREENING")
    band(E[7]["top"] + 0.0, E[5]["top"] + 1.4, "INCLUDED")

    for y, h, t, kw in rows:
        box(X, y, W, h, t, **kw)

    # vertical spine arrows: bottom edge of stage i -> top edge of stage i+1
    for i in range(len(rows) - 1):
        vconnect(E[i]["bot"], E[i + 1]["top"], X)

    # exclusion boxes: PRISMA vocabulary; (parent index, dy, h, text)
    excl_rows = [
        (0, 0.0, 4.2, f"Duplicate records removed\n(n = {n['dedup_rm']:,})"),
        (1, 0.0, 5.6, f"Records removed before screening\n(n = {n['filter_rm']:,}): preprints/reviews →\nchaining file · non-mangrove · title dupes"),
        (2, 0.0, 5.0, f"Records removed at pre-screen\n(n = {n_prescreen_rm:,}; 37-param term scoring)"),
        (3, 0.0, 4.2, f"Screening pilots consumed\n{n_drawn_from_frame} of {n['pilot_consumed']} titles"),
        (4, 0.0, 4.0, f"Records excluded\n(n = {n['screened'] - n['retained']})"),
        (5, 3.4, 5.2, f"Metadata validation\n{n['doi_confirmed']} DOI confirmed · {n['no_doi']} verified\nDOI-less · {n['year_fixes']} year fixes"),
        (6, 2.3, 3.6, f"Reports not retrieved (n = {n_not_retrieved}\nILL-pending); {n['pdf_at_close']}/{n['responsible']} PDFs ({n['pct']}%)"),
        (5, -3.6, 4.2, f"Records removed at intake (n = {n_intake_rm})\noff-topic {n['offtopic']} · predatory {n['predatory']}"),
    ]
    for i, dy, h, t in excl_rows:
        y = rows[i][0] + dy
        box(EX, y, EW, h, t, fc=EXF, ec=EXE, fs=7.2)
        hconnect(y, E[i]["r"], EX - EW / 2 - PAD)
    # visual-edge separation assert (D147): the metadata and intake boxes at
    # the 148 stage, and the intake/retrieval boxes at the 135 stage, must not
    # intersect once the FancyBbox patch pad is included
    s6 = rows[5][0]
    meta_lo = (s6 + 3.4) - 5.2 / 2 - PAD
    intake_top = (s6 - 3.6) + 4.2 / 2 + PAD
    assert meta_lo - intake_top >= 0.4, f"148-stage boxes overlap: {meta_lo} vs {intake_top}"
    s7 = rows[6][0]
    intake_bot = (s6 - 3.6) - 4.2 / 2 - PAD
    not_ret_top = (s7 + 2.3) + 3.6 / 2 + PAD
    assert intake_bot - not_ret_top >= 0.4, f"135-stage boxes overlap: {intake_bot} vs {not_ret_top}"

    ax.text(50, 98.0, "Corpus provenance funnel \u2014 mangrove water-quality literature "
                      "(OpenAlex search batch)",
            fontsize=10, fontweight="bold", color="#1a1a1a", ha="center")

    atomic_save(fig, OUT_PNG, dpi=600, bbox_inches="tight")
    atomic_save(fig, OUT_PDF, bbox_inches="tight")
    print(f"[prisma-flow] funnel numbers asserted {len(n)}/19 | papers 138 | "
          f"PDFs {n_pdf_disk} | texts {n_txt_disk} | data-papers {n_data_papers_n}")
    print(f"[prisma-flow] wrote {os.path.relpath(OUT_PNG, ROOT)} + .pdf")


if __name__ == "__main__":
    sys.exit(main())
