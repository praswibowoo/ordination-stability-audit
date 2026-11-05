#!/usr/bin/env python3
"""archive_preflight.py — second-scale checks on the shipping code archive.

`make reproduce` inside a fresh unpack takes about two hours, so it is a
deposit-time gate. Every defect that has actually reached the archive was
visible without running the numerics:

  * a script the shipped Makefile invokes but the bundle does not carry
    (D212 item 1: family_tiers.py and rho_paper_bootstrap.py were missing);
  * a target ordered before the producers of its inputs (D212 item 2:
    family-tiers ran before absratio-variants and familyexpansion);
  * a file inventory that no longer matches the deposited zip;
  * a checksum or a manifest that no longer matches the bundle.

This gate checks those in seconds, so the two-hour run can be reserved for the
deposit. It never runs the pipeline and never writes inside the archive.

Scope: the order check covers the tier and table targets (family-tiers,
tableS13, tableM), which is where the ordering defect occurred. It is not a
general dependency solver.

Run:  make archive-preflight
      python3 code/archive_preflight.py --bundle release/p1_family_pca_code_v1.1
Exit 0 = every check passed.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import py_compile
import re
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VERSION = os.environ.get("CODE_RELEASE_VERSION", "v1.4")
DEFAULT_BUNDLE = os.path.join(ROOT, "release", f"p1_family_pca_code_{VERSION}")
ZIP = os.path.join(ROOT, "release", "zenodo_deposition", f"p1_family_pca_code_{VERSION}.zip")
PREFIX = f"p1_family_pca_code_{VERSION}"

# artifact -> the `reproduce` prerequisite that produces it. Derived from the
# generators (experiment_G writes exG_absratio.csv, experiment_K the exK pair,
# experiment_M the exM suite metrics and the matched benchmark files) and from
# code/family_tiers.py for the tier table. Add an entry when a new artifact
# enters the tier or table chain; the check fails loudly on an unlisted one.
PRODUCERS = [
    ("exG_absratio.csv", "corrections"),          # experiment_G via `absratio`
    ("exK_absratio_summary.csv", "absratio-variants"),
    ("exK_absratio_variants.csv", "absratio-variants"),
    ("exM_benchmark_matched_*.csv", "familyexpansion"),   # --benchmark-matched
    ("exM_benchmark_matched_summary.csv", "familyexpansion"),
    ("exM_suite_metrics.csv", "familyexpansion"),         # default run
    ("family_tiers.csv", "family-tiers"),
]

# reproduce-level target -> the artifacts its script reads
CONSUMERS = {
    "family-tiers": ["exK_absratio_summary.csv", "exG_absratio.csv",
                     "exM_benchmark_matched_*.csv"],
    "tableS13": ["exK_absratio_summary.csv", "exG_absratio.csv"],
    "tableM": ["family_tiers.csv", "exM_suite_metrics.csv",
               "exM_benchmark_matched_summary.csv"],
}

# a consumer must not be pulled in by an earlier top-level target
MUST_NOT_BE_INSIDE = {"family-tiers": ["corrections", "figures", "tables"],
                      "tableM": ["corrections", "figures"],
                      "tableS13": ["corrections", "figures"]}


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bundle_files(bundle: str) -> list[str]:
    out = []
    ignore = {".DS_Store", "desktop.ini", "Thumbs.db"}
    for base, dirs, names in os.walk(bundle):
        dirs[:] = [d for d in sorted(dirs) if d != "__pycache__"]
        for n in sorted(names):
            if n in ignore:
                continue
            out.append(os.path.relpath(os.path.join(base, n), bundle).replace(os.sep, "/"))
    return sorted(out)


def recipes(makefile: str) -> dict[str, list[str]]:
    """target -> prerequisite list, for single-line `target: prereq ...` rules."""
    rules: dict[str, list[str]] = {}
    text = open(makefile, encoding="utf-8").read()
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+):\s*([^=].*)$", line)
        if not m:
            continue
        target, rest = m.group(1), m.group(2).strip()
        if rest.startswith("#"):
            rest = ""
        rules.setdefault(target, rest.split()) if rest else rules.setdefault(target, [])
    return rules


def main() -> int:
    # Run from inside an unpacked archive and check that archive; from the
    # project tree, check the built bundle.
    if os.path.exists(os.path.join(os.getcwd(), "CHECKSUMS.txt")) and \
            os.path.isdir(os.path.join(os.getcwd(), "code")):
        DEFAULT_BUNDLE_RUNTIME = os.getcwd()
    else:
        DEFAULT_BUNDLE_RUNTIME = DEFAULT_BUNDLE
    bundle = DEFAULT_BUNDLE_RUNTIME
    argv = sys.argv[1:]
    if "--bundle" in argv:
        bundle = argv[argv.index("--bundle") + 1]
    prefix = os.path.basename(os.path.normpath(bundle))
    zip_path = ZIP
    if "--zip" in argv:
        zip_path = argv[argv.index("--zip") + 1]

    fails: list[str] = []
    notes: list[str] = []

    if not os.path.isdir(bundle):
        print(f"FAIL  bundle not found: {bundle}\n"
              f"      build it first: python3 code/make_code_release.py")
        return 2

    mk = os.path.join(bundle, "Makefile")
    if not os.path.exists(mk):
        print(f"FAIL  the bundle carries no Makefile: {mk}")
        return 2
    rules = recipes(mk)

    # ---- 1. inventory: every script the shipped Makefile invokes ships ------
    invoked = set(re.findall(r"code/([A-Za-z0-9_]+\.py)",
                             open(mk, encoding="utf-8").read()))
    present = set(os.listdir(os.path.join(bundle, "code"))) \
        if os.path.isdir(os.path.join(bundle, "code")) else set()
    absent = sorted(invoked - present)
    if absent:
        fails.append(f"inventory: the Makefile invokes scripts the bundle does not "
                     f"carry: {absent}")
    else:
        print(f"ok    inventory: {len(invoked)} Makefile-invoked scripts all present")

    # ---- 2. every shipped script compiles ----------------------------------
    broken = []
    with tempfile.TemporaryDirectory() as tmp:
        for f in sorted(present):
            if not f.endswith(".py"):
                continue
            try:
                py_compile.compile(os.path.join(bundle, "code", f),
                                   cfile=os.path.join(tmp, f + "c"),
                                   doraise=True)
            except py_compile.PyCompileError as exc:
                broken.append(f"{f}: {exc.msg.splitlines()[-1][:120]}")
    if broken:
        fails.append("compile: " + "; ".join(broken))
    else:
        print(f"ok    compile: {len(present)} shipped scripts compile")

    # ---- 3. the deposited zip carries exactly the bundle -------------------
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path) as z:
            in_zip = sorted(n[len(prefix) + 1:] for n in z.namelist()
                            if n.startswith(prefix + "/") and not n.endswith("/"))
        in_dir = bundle_files(bundle)
        if in_zip != in_dir:
            only_zip = sorted(set(in_zip) - set(in_dir))[:5]
            only_dir = sorted(set(in_dir) - set(in_zip))[:5]
            fails.append(f"zip parity: the deposited zip and the bundle differ "
                         f"(zip-only {only_zip}, bundle-only {only_dir})")
        else:
            print(f"ok    zip parity: {len(in_zip)} files, zip == bundle")
    else:
        notes.append(f"zip not found at {zip_path}; parity check skipped")

    # ---- 4. checksums ------------------------------------------------------
    sums = os.path.join(bundle, "CHECKSUMS.txt")
    if not os.path.exists(sums):
        fails.append("checksums: CHECKSUMS.txt missing")
    else:
        bad = []
        for line in open(sums, encoding="utf-8"):
            if not line.strip():
                continue
            digest, _, name = line.rstrip("\n").partition("  ")
            full = os.path.join(bundle, name)
            if not os.path.exists(full):
                bad.append(f"{name} (missing)")
            elif sha256(full) != digest:
                bad.append(f"{name} (hash)")
        if bad:
            fails.append(f"checksums: {len(bad)} mismatched: {bad[:4]}")
        else:
            print(f"ok    checksums: {sum(1 for l in open(sums) if l.strip())} files verify")

    # ---- 5. manifest drift -------------------------------------------------
    man_path = os.path.join(bundle, "archive_manifest.json")
    if not os.path.exists(man_path):
        notes.append("no archive_manifest.json; drift check skipped "
                     "(rebuild with code/make_code_release.py to record one)")
    else:
        man = json.load(open(man_path, encoding="utf-8"))
        recorded = man.get("files", {})
        current = {f: sha256(os.path.join(bundle, f)) for f in bundle_files(bundle)
                   if f not in ("CHECKSUMS.txt", "archive_manifest.json")}
        drift = sorted(set(recorded) ^ set(current))
        changed = sorted(k for k in set(recorded) & set(current)
                         if recorded[k] != current[k])
        if drift or changed:
            fails.append(f"manifest: bundle drifted from the recorded build "
                         f"(added/removed {drift[:5]}, changed {changed[:5]})")
        else:
            print(f"ok    manifest: {len(current)} files match the recorded build "
                  f"({man.get('counts', {}).get('scripts', '?')} scripts, "
                  f"claims {man.get('claims', {}).get('total', '?')})")

    # ---- 6. producer/consumer order inside `reproduce` ---------------------
    order = rules.get("reproduce", [])
    if not order:
        fails.append("order: no `reproduce` prerequisite list to check")
    else:
        def produced_by(artifact: str) -> str | None:
            for pat, target in PRODUCERS:
                if fnmatch.fnmatch(artifact, pat) or artifact == pat:
                    return target
            return None

        problems = []
        for consumer, artifacts in CONSUMERS.items():
            if consumer not in order:
                problems.append(f"{consumer} is not a `reproduce` prerequisite")
                continue
            ci = order.index(consumer)
            for art in artifacts:
                prod = produced_by(art)
                if prod is None:
                    problems.append(f"{art} has no producer entry (add one to PRODUCERS)")
                    continue
                if prod not in order:
                    problems.append(f"{art} is produced by `{prod}`, which is not in "
                                    f"`reproduce`")
                    continue
                if order.index(prod) > ci:
                    problems.append(f"{consumer} reads {art} but `{prod}` runs after it")
            for outer in MUST_NOT_BE_INSIDE.get(consumer, []):
                if consumer in rules.get(outer, []):
                    problems.append(f"{consumer} is a prerequisite of `{outer}`, which "
                                    f"runs earlier")
        if problems:
            fails.append("order: " + "; ".join(problems))
        else:
            print(f"ok    order: {len(CONSUMERS)} tier/table targets read only artifacts "
                  f"produced earlier in `reproduce`")

    # ---- 7. verification record: is this the build a run validated? --------
    rec = os.path.join(ROOT, "release", "archive_verification.json")
    if not os.path.exists(rec):
        notes.append("no release/archive_verification.json; run `make archive-verify` "
                     "before depositing")
    else:
        v = json.load(open(rec, encoding="utf-8"))
        rec_files = v.get("files", {})
        cur = {f: sha256(os.path.join(bundle, f)) for f in bundle_files(bundle)}
        new = sorted(set(cur) - set(rec_files))
        gone = sorted(set(rec_files) - set(cur))
        changed = sorted(k for k in set(cur) & set(rec_files) if cur[k] != rec_files[k])
        when = v.get("verified", {}).get("finished", "?")
        if not (new or gone or changed):
            print(f"ok    verification: this bundle is the build a run validated ({when})")
        else:
            notes.append(
                f"verification: this bundle is NOT the build a full run validated "
                f"({when}) — {len(new)} new, {len(changed)} changed, {len(gone)} removed "
                f"shipped files; run `make archive-verify` before depositing")

    for n in notes:
        print(f"note  {n}")
    if fails:
        print()
        for f in fails:
            print(f"FAIL  {f}")
        print(f"\n[archive-preflight] {len(fails)} check(s) failed")
        return 1
    print("\n[archive-preflight] ALL CHECKS PASS (seconds-scale; "
          "the 2 h `make reproduce` remains the deposit gate)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
