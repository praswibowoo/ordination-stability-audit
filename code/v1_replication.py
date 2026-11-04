#!/usr/bin/env python3
"""v1_replication.py — reproduce the v1 pilot's F1 PCA structure from the frozen xlsx (D110).

v1 claim (Analysis_Design B4): "F1 PC1 60.2%, PC2 27.9% from v1 becomes the pilot".
Method: parse the frozen v1 workbook 'Matrix space' sheet (presence 0/1), take the F1
parameter set, run PCA(2) — same conventions as the Phase-4 pipeline. Also report F2 metals
for the record. Output: results/v1_replication.csv
"""
from __future__ import annotations
import csv, html, os, re, zipfile
import numpy as np
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XLSX = os.path.join(ROOT, "data/raw/mangrove_water_parameters_v1_frozen.xlsx")
F1 = ["DO (mg/L)", "pH", "Salinity (ppt/\u2030)", "Temperature (\u00b0C)"]


def parse_sheet3():
    z = zipfile.ZipFile(XLSX)
    ss = z.read("xl/sharedStrings.xml").decode("utf-8")
    SST = [html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", s, re.S)))
           for s in re.findall(r"<si>(.*?)</si>", ss, re.S)]
    sh = z.read("xl/worksheets/sheet3.xml").decode("utf-8")
    rows = re.findall(r"<row[^>]*>(.*?)</row>", sh, re.S)
    def parse_row(rx):
        out = {}
        for c in re.finditer(r'<c r="([A-Z]+)\d+"([^>]*)>(.*?)</c>', rx, re.S):
            col, attrs, inner = c.group(1), c.group(2), c.group(3)
            v = re.search(r"<v>([^<]*)</v>", inner)
            if not v:
                continue
            out[col] = SST[int(v.group(1))] if 't="s"' in attrs else v.group(1)
        return out
    parsed = [parse_row(r) for r in rows]
    hdr = parsed[0]
    col_of = {name: col for col, name in hdr.items()}
    return parsed[1:], col_of, len(parsed) - 1


def main():
    data, col_of, n_rows = parse_sheet3()
    out = []
    for label, params in (("F1_physical", F1),
                          ("F2_metals", ["Zn (\u00b5g/L)", "Cr (\u00b5g/L)", "Pb (\u00b5g/L)"])):
        cols = [col_of[p] for p in params if p in col_of]
        if len(cols) < 2:
            print(f"{label}: columns missing in v1 xlsx"); continue
        X = np.asarray([[float(r.get(c, "0") or 0) for c in cols] for r in data])
        pca = PCA(n_components=2).fit(X)
        evr = pca.explained_variance_ratio_
        out.append(dict(set="+".join(p.split(" (")[0].split()[0] for p in params if p in col_of),
                        n_rows=len(X), n_params=len(cols),
                        PC1=round(float(evr[0]), 4), PC2=round(float(evr[1]), 4)))
        print(f"{label}: n={len(X)}x{len(cols)} → PC1 {evr[0]*100:.1f}%  PC2 {evr[1]*100:.1f}%")
    # comparison vs the recorded v1 claim
    f1 = out[0] if out else None
    if f1:
        d1 = abs(f1["PC1"] - 0.602); d2 = abs(f1["PC2"] - 0.279)
        verdict = "MATCH" if (d1 <= 0.015 and d2 <= 0.015) else "DEVIATION"
        out.append(dict(set="v1_claim_reference", n_rows="", n_params="",
                        PC1=0.602, PC2=0.279))
        print(f"vs recorded v1 claim (60.2% / 27.9%): {verdict} (dPC1={d1*100:.1f}pp, dPC2={d2*100:.1f}pp)")
    ap = os.path.join(ROOT, "results/v1_replication.csv")
    with open(ap, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["set", "n_rows", "n_params", "PC1", "PC2"])
        w.writeheader(); w.writerows(out)
    print("written results/v1_replication.csv")


if __name__ == "__main__":
    main()
