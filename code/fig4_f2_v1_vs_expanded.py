#!/usr/bin/env python3
"""fig4_f2_v1_vs_expanded.py — Fig 4: F2 metals v1 (frozen 49-row pilot) vs expanded corpus."""
import csv, os, re, html, zipfile
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")  # byte-stable PDF twins (D117)
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, "data/raw/mangrove_water_parameters_v1_frozen.xlsx")
FIGS = os.path.join(ROOT, "figures")

DPI = int(os.environ.get("FIG_DPI", "600"))  # publication grade 600 DPI (P1-6.1)

def _save(fig, path):
    """PNG at DPI + vector PDF twin, tight bbox (D117)."""
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    fig.savefig(os.path.splitext(path)[0] + ".pdf", bbox_inches="tight")



METAL_HEADERS = None


def v1_metal_counts():
    z = zipfile.ZipFile(XLSX)
    ss = z.read("xl/sharedStrings.xml").decode("utf-8")
    SST = [html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", s, re.S)))
           for s in re.findall(r"<si>(.*?)</si>", ss, re.S)]
    sh = z.read("xl/worksheets/sheet3.xml").decode("utf-8")
    rows = re.findall(r"<row[^>]*>(.*?)</row>", sh, re.S)
    hdr_row = rows[0]
    hdr = {}
    for c in re.finditer(r'<c r="([A-Z]+)\d+"[^>]*t="s"[^>]*><v>(\d+)</v>', hdr_row):
        hdr[c.group(1)] = SST[int(c.group(2))]
    metals = {v: k for k, v in hdr.items()
              if v and any(m in v for m in ("As (", "Cd (", "Cr (", "Cu (", "Fe (",
                                            "Hg (", "Mn (", "Ni (", "Pb (", "Zn ("))}
    counts = defaultdict(int)
    papers = defaultdict(set)
    for rx in rows[1:]:
        cells = {}
        for c in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*)>(.*?)</c>', rx, re.S):
            col, attrs, inner = c.group(1), c.group(2), c.group(3)
            v = re.search(r"<v>([^<]*)</v>", inner)
            if not v:
                continue
            cells[col] = SST[int(v.group(1))] if 't="s"' in attrs else v.group(1)
        lab = cells.get("A", "")
        for name, col in metals.items():
            if str(cells.get(col, "")).strip() not in ("", "0"):
                counts[name] += 1
                papers[name].add(lab)
    return counts, papers


def current_metal_counts():
    pres = list(csv.DictReader(open(os.path.join(
        ROOT, "docs/extraction/stageA/matrix_presence_v2.csv"), encoding="utf-8-sig")))
    counts = defaultdict(int)
    papers = defaultdict(set)
    own = ""
    for r in pres:
        b = (r["batch_id"] or "").strip()
        if b:
            own = b
        if r["row_kind"] != "data":
            continue
        for p in ("As (µg/L)", "Cd (µg/L)", "Cr (µg/L)", "Cu (µg/L)", "Fe (µg/L)",
                  "Hg (µg/L)", "Mn (µg/L)", "Ni (µg/L)", "Pb (µg/L)", "Zn (µg/L)"):
            if r[p].strip() in ("Y", "NE"):
                counts[p] += 1
                papers[p].add(own)
    return counts, papers


def main():
    v1c, v1p = v1_metal_counts()
    c2c, c2p = current_metal_counts()
    order = ["As (µg/L)", "Cd (µg/L)", "Cr (µg/L)", "Cu (µg/L)", "Fe (µg/L)",
             "Hg (µg/L)", "Mn (µg/L)", "Ni (µg/L)", "Pb (µg/L)", "Zn (µg/L)"]
    x = np.arange(len(order))
    v1 = [v1c.get(p, 0) for p in order]
    cur = [c2c.get(p, 0) for p in order]
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    ax.bar(x - 0.2, v1, 0.4, label="v1 pilot (49 rows / 19 papers)", color="#90a4ae")
    ax.bar(x + 0.2, cur, 0.4, label="expanded corpus (764 rows / 122 papers)", color="#4c72b0")
    ax.set_xticks(x)
    ax.set_xticklabels([p.split(" (")[0] for p in order], fontsize=8.2)
    ax.set_ylabel("rows reporting the metal", fontsize=8.2)
    for i, p in enumerate(order):
        ax.annotate(f"{len(v1p.get(p, set()))}p", (i - 0.2, v1[i]), fontsize=6.8, ha="center",
                    va="bottom", xytext=(0, 2), textcoords="offset points")
        ax.annotate(f"{len(c2p.get(p, set()))}p", (i + 0.2, cur[i]), fontsize=6.8, ha="center",
                    va="bottom", xytext=(0, 2), textcoords="offset points")
        if v1[i] > 0:
            ax.annotate(f"×{cur[i] / v1[i]:.0f}", (i + 0.2, cur[i]), fontsize=7.2,
                        ha="center", va="bottom", xytext=(0, 11),
                        textcoords="offset points", color="#1b4f72", fontweight="bold")
    ax.legend(loc="upper center", framealpha=0.92, fontsize=8.0)
    ax.set_title("F2 metals: pilot vs expanded corpus (labels above bars: distinct papers 'p')",
                 fontsize=8.4, pad=5)
    ax.set_ylim(0, max(max(v1), max(cur)) * 1.36)  # headroom for ×fold labels
    fig.tight_layout()
    _save(fig, os.path.join(FIGS, "fig4_f2_v1_vs_expanded.png"))
    print("fig4 saved | v1 counts:", dict(v1c), "| current:", dict(c2c))


if __name__ == "__main__":
    main()
