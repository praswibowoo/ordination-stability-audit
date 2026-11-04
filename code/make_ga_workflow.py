#!/usr/bin/env python3
"""Graphical abstract — deletion-stability audit workflow.

Standalone submission item (Elsevier / Environmental Modelling & Software). This is not a
main-text figure and is not referenced in the manuscript.

Elsevier graphical-abstract canvas: 13 x 5 cm landscape. The PDF is written at
that exact page size (no tight bounding box), and the PNG is the 300 dpi twin.

Outputs (atomic tmp->replace):
  figures/figGA_workflow.png (300 dpi)
  figures/figGA_workflow.pdf (13 x 5 cm)

Run: python3 code/make_ga_workflow.py
"""
import os

os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twin

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PNG = os.path.join(ROOT, "figures", "figGA_workflow.png")
OUT_PDF = os.path.join(ROOT, "figures", "figGA_workflow.pdf")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def atomic_save(fig, path, **kw):
    base, ext = os.path.splitext(path)
    tmp = base + ".tmp" + ext
    fig.savefig(tmp, **kw)
    os.replace(tmp, path)


# 13 x 5 cm at the journal's requested canvas. Coordinates are centimetres.
W, H = 13.0, 5.0

STAGES = [
    (
        "1  Inputs",
        "764 rows · 122 papers\n44 parameters",
        "#E8F1FA",
    ),
    (
        "2  Deletion",
        "LOPO vs volume-matched\nscattered control",
        "#EAF6EC",
    ),
    (
        "3  Diagnostics",
        "Procrustes drift \u03b4\nneighbour churn \u03c7",
        "#FFF4E5",
    ),
    (
        "4  Families",
        "Jaccard \u2192 Ward\nfamily restriction",
        "#F3EAF8",
    ),
    (
        "5  Decision",
        "absolute ratio vs\nrandom families\n(tested, not assumed)",
        "#FDECEC",
    ),
]


def main() -> None:
    fig = plt.figure(figsize=(W / 2.54, H / 2.54))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")

    ax.text(
        W / 2, 4.74,
        "Deletion-stability audit for literature-derived meta-matrices",
        ha="center", va="center", fontsize=7.4, fontweight="bold", color="#1F3448",
    )
    ax.text(
        W / 2, 4.40,
        "Paper-level resampling  ·  matched-volume controls",
        ha="center", va="center", fontsize=5.2, color="#5A6B7B",
    )
    ax.text(
        W / 2, 4.03,
        "In simulation, row-level tests inflate false positives; paper-level tests stay near nominal.\n"
        "One paper (0.65% of rows) moves 20% of neighbours through ties; blocks add 42% drift.",
        ha="center", va="center", fontsize=4.6, color="#1F3448", linespacing=1.15,
    )

    box_w, box_h = 2.15, 2.35
    left = (W - (5 * box_w + 4 * 0.25)) / 2
    y0 = 1.35
    cy = y0 + box_h / 2
    for i, (title, body, face) in enumerate(STAGES):
        x = left + i * (box_w + 0.25)
        ax.add_patch(
            FancyBboxPatch(
                (x, y0), box_w, box_h,
                boxstyle="round,pad=0.04,rounding_size=0.08",
                linewidth=0.65, edgecolor="#2F4B63", facecolor=face,
            )
        )
        ax.text(
            x + box_w / 2, y0 + box_h - 0.42,
            title, ha="center", va="center",
            fontsize=6.1, fontweight="bold", color="#1F3448",
        )
        ax.text(
            x + box_w / 2, y0 + 0.92,
            body, ha="center", va="center",
            fontsize=4.55, color="#33475B", linespacing=1.35,
        )
        if i < len(STAGES) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x + box_w + 0.03, cy), (x + box_w + 0.22, cy),
                    arrowstyle="-|>", mutation_scale=6,
                    linewidth=0.8, color="#5A6B7B",
                )
            )

    ax.text(
        W / 2, 0.50,
        "Confirmatory value-layer tests: 10 value-complete rows and 10 papers per family.",
        ha="center", va="center", fontsize=4.25, color="#5A6B7B",
    )

    atomic_save(fig, OUT_PNG, dpi=300, facecolor="white")
    atomic_save(fig, OUT_PDF, facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    print(f"wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
