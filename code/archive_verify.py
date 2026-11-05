#!/usr/bin/env python3
"""archive_verify.py — run, or record, the full archive reproduction.

`code/archive_preflight.py` answers "is the archive internally coherent?" in
seconds. It cannot answer "does the pipeline still regenerate the shipped
artifacts?", which needs the two-hour run. This script performs that run in a
scratch directory and writes `release/archive_verification.json`: the exit code,
the gate line, the per-group artifact comparison and the hash of every shipped
file, so a later session can tell which build a run actually validated.

  make archive-verify                                             # full run (~2 h)
  python3 code/archive_verify.py --from-log LOG --dir UNPACKED    # record an existing run

The record lives in the repository, not in the archive. Stamping it into the
bundle would change the bytes it describes, and the bundle already carries
`archive_manifest.json` and `CHECKSUMS.txt` for a reader.

Run from the project root. Exit 0 = the run passed and the record was written.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIPDIR = os.path.join(ROOT, "release", "zenodo_deposition")
VERSION = os.environ.get("CODE_RELEASE_VERSION", "v1.4")
DATA_VERSION = os.environ.get("DATA_RELEASE_VERSION", "v1.1")
CODE_ZIP = os.path.join(ZIPDIR, f"p1_family_pca_code_{VERSION}.zip")
DATA_ZIP = os.path.join(ZIPDIR, f"p1_family_pca_data_{DATA_VERSION}.zip")
BUNDLE = os.path.join(ROOT, "release", f"p1_family_pca_code_{VERSION}")
RECORD = os.path.join(ROOT, "release", "archive_verification.json")
ENV_RECORD = os.path.join(ROOT, "results", "env_versions.json")
PREFIX = f"p1_family_pca_code_{VERSION}"
PIN_KEYS = ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "seaborn")
ENV_GOT: dict = {}

# artifacts the pipeline writes, compared against the project tree
GROUPS = {"tables": "manuscript/tables", "figures": "figures", "results": "results"}


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def interpreter_versions(py: str) -> dict:
    """Versions of the interpreter that `make` will actually use (D261)."""
    code = (
        "import json, platform, importlib.metadata as m;"
        "d={'python': platform.python_version()};"
        "d.update({p: m.version(p) for p in "
        "['numpy','pandas','scipy','scikit-learn','matplotlib','seaborn']});"
        "print(json.dumps(d))"
    )
    return json.loads(subprocess.check_output(
        [py, "-c", code], text=True, stderr=subprocess.DEVNULL))


def check_environment(py: str) -> tuple[dict, dict]:
    """Fail before the 3 h run if the interpreter is not the pinned one."""
    want = json.load(open(ENV_RECORD, encoding="utf-8"))
    try:
        got = interpreter_versions(py)
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise SystemExit(f"[archive-verify] cannot read versions from {py}: {exc}") from exc
    bad = {
        k: (want.get(k), got.get(k))
        for k in ("python",) + PIN_KEYS
        if str(want.get(k)) != str(got.get(k))
    }
    if bad:
        detail = ", ".join(f"{k}: want {w}, got {g}" for k, (w, g) in bad.items())
        raise SystemExit(f"[archive-verify] environment mismatch: {detail}")
    return got, want


def content_key(path: str) -> str:
    """Hash of the content, ignoring the gzip container (header mtime, level)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as fh:
            return sha256_bytes(fh.read())
    return sha256_file(path)


def shipped_hashes(bundle: str) -> dict[str, str]:
    out = {}
    for base, dirs, names in os.walk(bundle):
        dirs[:] = [d for d in sorted(dirs) if d != "__pycache__"]
        for n in sorted(names):
            p = os.path.join(base, n)
            out[os.path.relpath(p, bundle)] = sha256_file(p)
    return out


def compare_artifacts(run_dir: str) -> dict:
    groups = {}
    for name, rel in GROUPS.items():
        d = os.path.join(run_dir, rel)
        if not os.path.isdir(d):
            groups[name] = {"present": False}
            continue
        files = [f for f in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, f))]
        identical, differing, only_run = [], [], []
        for f in files:
            tree = os.path.join(ROOT, rel, f)
            if not os.path.exists(tree):
                only_run.append(f)
            elif content_key(os.path.join(d, f)) == content_key(tree):
                identical.append(f)
            else:
                differing.append(f)
        groups[name] = {"present": True, "files": len(files),
                        "identical": len(identical), "differing": differing,
                        "only_in_run": only_run}
    return groups


def gate_line(log: str) -> str | None:
    hits = re.findall(r"^\[check-numbers\].*$", log, re.M)
    return hits[-1].strip() if hits else None


def exit_code(log: str) -> int | None:
    m = re.findall(r"ARCHIVE REPRODUCE EXIT=(\d+)", log)
    if m:
        return int(m[-1])
    if "[make] REPRODUCE complete" in log:
        return 0
    return None


def write_record(payload: dict) -> None:
    with open(RECORD, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")


def build_payload(log: str, run_dir: str, started: str | None) -> dict:
    text = open(log, encoding="utf-8", errors="replace").read()
    run_shipped = {}
    present = set()
    for base, dirs, names in os.walk(run_dir):
        dirs[:] = [d for d in sorted(dirs) if d != "__pycache__"]
        for n in sorted(names):
            present.add(os.path.relpath(os.path.join(base, n), run_dir))
    bundle = shipped_hashes(BUNDLE) if os.path.isdir(BUNDLE) else {}
    for rel in sorted(bundle):
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            run_shipped[rel] = sha256_file(p)

    differing = sorted(k for k, v in run_shipped.items() if bundle.get(k) != v)
    missing = sorted(set(bundle) - set(run_shipped))

    payload = {
        "schema": 1,
        "archive": f"p1_family_pca_code_{VERSION}",
        "verified": {
            "log": os.path.relpath(log, ROOT) if log.startswith(ROOT) else log,
            "run_dir": run_dir,
            "environment": ENV_GOT,
            "started": started,
            "finished": time.strftime("%Y-%m-%dT%H:%M:%S",
                                      time.localtime(os.path.getmtime(log))),
            "exit_code": exit_code(text),
            "reproduce_complete": "[make] REPRODUCE complete" in text,
            "gate": gate_line(text),
            "artifacts": compare_artifacts(run_dir),
            # md5 of the archives on disk when this record was written, not of
            # the archive that ran; the `files` map below identifies that build
            "code_zip_md5_at_record_time": hashlib.md5(open(CODE_ZIP, "rb").read()).hexdigest()
            if os.path.exists(CODE_ZIP) else None,
            "data_zip_md5_at_record_time": hashlib.md5(open(DATA_ZIP, "rb").read()).hexdigest()
            if os.path.exists(DATA_ZIP) else None,
        },
        # the shipped files as they were in the archive that ran, so a later
        # build can be compared with the build a run actually validated
        "files": dict(sorted(run_shipped.items())),
        "not_in_run": missing,
        "differing_at_record_time": differing,
    }
    return payload


def main() -> int:
    argv = sys.argv[1:]
    if "--run" in argv:
        global ENV_GOT
        py = (argv[argv.index("--python") + 1] if "--python" in argv
              else os.environ.get("PY", sys.executable))
        ENV_GOT, _ = check_environment(py)
        print(f"[archive-verify] env {ENV_GOT['python']} | "
              f"numpy {ENV_GOT['numpy']} | sklearn {ENV_GOT['scikit-learn']}")
        if shutil.which("make") is None:
            raise SystemExit("[archive-verify] make is not on PATH")
        run_env = os.environ.copy()
        run_env["PY"] = py.replace("\\", "/")
        if os.name == "nt" and not run_env.get("SHELL"):
            for cand in (r"C:\Program Files\Git\bin\bash.exe",
                         r"C:\Program Files (x86)\Git\bin\bash.exe"):
                if os.path.exists(cand):
                    run_env["SHELL"] = cand
                    break
            if not run_env.get("SHELL"):
                raise SystemExit("[archive-verify] no POSIX shell found; set SHELL")
        started = time.strftime("%Y-%m-%dT%H:%M:%S")
        t0 = time.time()
        tmp = tempfile.mkdtemp(prefix="p1archive-")
        print(f"[archive-verify] unpacking into {tmp}")
        for z in (CODE_ZIP, DATA_ZIP):
            if not os.path.exists(z):
                raise SystemExit(f"[archive-verify] missing archive: {z}")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(tmp)
        run_dir = os.path.join(tmp, PREFIX)
        log_path = os.path.join(tmp, "reproduce.log")
        with open(log_path, "w", encoding="utf-8") as log:
            for cmd in (["make", "stage-layers"], ["make", "reproduce"]):
                log.write(f"$ {' '.join(cmd)}\n")
                log.flush()
                rc = subprocess.call(cmd, cwd=run_dir, stdout=log,
                                     stderr=subprocess.STDOUT, env=run_env)
                if rc:
                    log.write(f"\nARCHIVE REPRODUCE EXIT={rc}\n")
                    break
            else:
                log.write("\nARCHIVE REPRODUCE EXIT=0\n")
        payload = build_payload(log_path, run_dir, started)
        payload["verified"]["duration_s"] = round(time.time() - t0)
        write_record(payload)
        print(f"[archive-verify] {payload['verified']['duration_s']} s | "
              f"exit {payload['verified']['exit_code']} | "
              f"{payload['verified']['gate']}")
        print(f"[archive-verify] record → {os.path.relpath(RECORD, ROOT)}")
        if "--strict-artifacts" in argv:
            bad = (payload["verified"]["artifacts"]["results"]["differing"]
                   + payload["verified"]["artifacts"]["tables"]["differing"])
            if bad:
                print(f"[archive-verify] STRICT ARTIFACT FAIL: "
                      f"{len(bad)} result/table files differ")
                return 1
        return 0 if payload["verified"]["exit_code"] == 0 else 1

    if "--from-log" in argv and "--dir" in argv:
        log_path = argv[argv.index("--from-log") + 1]
        run_dir = argv[argv.index("--dir") + 1]
        started = argv[argv.index("--started") + 1] if "--started" in argv else None
        for p in (log_path, run_dir):
            if not os.path.exists(p):
                raise SystemExit(f"[archive-verify] not found: {p}")
        payload = build_payload(log_path, run_dir, started)
        write_record(payload)
        v = payload["verified"]
        print(f"[archive-verify] recorded | exit {v['exit_code']} | {v['gate']}")
        print(f"[archive-verify] shipped files recorded: {len(payload['files'])} | "
              f"new since that build: {len(payload['not_in_run'])} | "
              f"changed: {len(payload['differing_at_record_time'])}")
        print(f"[archive-verify] record → {os.path.relpath(RECORD, ROOT)}")
        return 0 if v["exit_code"] == 0 else 1

    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
