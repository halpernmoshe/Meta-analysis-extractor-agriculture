#!/usr/bin/env python3
"""Check the whole deposited workflow with one command.

    python verify.py

Steps, in order:
  1. Extraction records: every JSON file under 1_extraction/records/ is listed in
     SHA256SUMS.txt with the same SHA-256 digest and byte size, no file is missing or
     extra, and every file parses as JSON.
  2. AI-side conversion: the five decoders in 2_conversion/ai_side/ are run into a
     temporary directory. The generated key CSVs must have the same file names as the
     deposited tables in 2_conversion/keys/<dataset>/ai/ and the same content after
     line endings are normalised (CRLF -> LF).
  2r. Reference-side conversion: the four reference-side builders in
     2_conversion/reference_side/ are run on the published files in 0_sources/raw/ into a
     temporary directory and compared with 2_conversion/keys/<dataset>/gt/ after line-ending
     normalisation. Loladze and Li J must match in full. Boldorini must match except the two
     documented Garfinkel 2020 rows, which may differ only in `crop` and `evidence`. Hui is
     SKIPPED unless the workbook has been downloaded into 0_sources/raw/hui/. Biochar cannot
     be rebuilt by a script; for it (and for the other four) a trace check confirms that every
     reference key row appears once in row_trace/ with a match flag and the same means.
  3. Evaluation: every analysis script is run and its printed output is compared as text
     (after line-ending normalisation) with the expected-output file.
  4. Figures: every figure script is run and must produce its PNG. The printed values are
     compared with the expected-output files, and the PNG is compared with the deposited
     figure (identical file, else identical decoded pixels). A different image is a failure.
     The figures were produced with matplotlib 3.10.8; other versions can render the same
     numbers slightly differently. In that case run
         python verify.py --allow-image-differences
     which reports image differences as warnings and relies on the printed-value checks
     (these cover Figures 2, 3 and S1, but not Figure 1).

Steps 3 and 4 run in a temporary copy of the repository, from an unrelated working
directory, so the repository itself is not modified. Generated figures and coverage tables
are deleted from the copy before the scripts run, so each one is shown to be regenerated.

Exit status is 0 only if no check fails (SKIPPED is not a failure). Uses the standard
library, matplotlib for the figures, openpyxl for the Loladze, Li J and Hui reference builders,
and pandas for the Hui reference builder.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECORDS = ROOT / "1_extraction" / "records"
MANIFEST = RECORDS / "SHA256SUMS.txt"
# Manifest paths are relative to the repository root.
MANIFEST_PREFIX = "1_extraction/records/"
N_RECORDS = 178
KEYS = ROOT / "2_conversion" / "keys"
EXPECTED = ROOT / "3_evaluation" / "expected_outputs"
FIG_EXPECTED = ROOT / "4_figures" / "expected_outputs"

# decoder script, folder it writes under DECODER_OUTPUT_ROOT, deposited key folder
DECODERS = [
    ("2_conversion/ai_side/decode_biochar.py", "biochar", "biochar_v2"),
    ("2_conversion/ai_side/decode_boldorini_march_v11.py", "boldorini", "boldorini"),
    ("2_conversion/ai_side/decode_hui.py", "hui_strict", "hui_v4"),
    ("2_conversion/ai_side/decode_li_j.py", "li_j", "li2022_v2"),
    ("2_conversion/ai_side/decode_loladze.py", "loladze", "loladze_v2"),
]

# analysis script, expected-output file (in 3_evaluation/expected_outputs/)
ANALYSES = [
    ("3_evaluation/line_by_line_scope_aware.py", "EXPECTED_OUTPUT_LINEBYLINE.txt"),
    ("3_evaluation/scope_aware_paired_tost.py", "EXPECTED_OUTPUT_PAIRED_TOST.txt"),
    ("3_evaluation/scope_aware_aggregate_tost.py", "EXPECTED_OUTPUT_AGGREGATE_TOST.txt"),
    ("3_evaluation/biochar_native_control_tost.py", "EXPECTED_OUTPUT_BIOCHAR_NATIVE_CONTROL.txt"),
    ("3_evaluation/reconciliation_analysis.py", "EXPECTED_OUTPUT_RECONCILIATION.txt"),
    ("3_evaluation/make_bland_altman.py", "EXPECTED_OUTPUT_BLAND_ALTMAN.txt"),
    ("3_evaluation/round2_additional_analysis/coverage_structural_complexity.py",
     "EXPECTED_OUTPUT_COVERAGE.txt"),
]

# Tables written by the coverage script; compared with the deposited copies.
COVERAGE_OUTPUTS = [
    "3_evaluation/round2_additional_analysis/coverage_source_format_summary.csv",
    "3_evaluation/round2_additional_analysis/coverage_structural_complexity_associations.csv",
    "3_evaluation/round2_additional_analysis/coverage_structural_complexity_cells.csv",
    "3_evaluation/round2_additional_analysis/coverage_structural_complexity_papers.csv",
    "3_evaluation/round2_additional_analysis/coverage_structural_complexity_summary.csv",
    "3_evaluation/round2_additional_analysis/coverage_structural_complexity_report.md",
]

# figure script (or None if produced by an analysis script above), PNG,
# expected printed values (in 4_figures/expected_outputs/) or None
FIGURES = [
    ("4_figures/make_fig1_fidelity.py", "4_figures/figures/fig1_fidelity.png", None),
    ("4_figures/make_fig2_equivalence.py", "4_figures/figures/fig2_equivalence.png",
     "EXPECTED_OUTPUT_FIG2_EQUIVALENCE.txt"),
    ("4_figures/make_fig3_reconciliation.py", "4_figures/figures/fig3_reconciliation.png",
     "EXPECTED_OUTPUT_FIG3_RECONCILIATION.txt"),
    (None, "4_figures/figures/figS1_bland_altman.png", None),  # written by make_bland_altman.py
]

RESULTS: list[tuple[str, str, str]] = []  # (check, status, detail)


def record(check: str, ok: bool, detail: str, warn: bool = False, skip: bool = False) -> None:
    status = "SKIPPED" if skip else ("PASS" if ok else ("WARN" if warn else "FAIL"))
    RESULTS.append((check, status, detail))
    print(f"[{status}] {check}: {detail}")


def text(data: bytes) -> str:
    return data.decode("utf-8").replace("\r\n", "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def child_env(**extra: str) -> dict:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"  # some scripts print non-ASCII author names
    env["MPLBACKEND"] = "Agg"
    env.update(extra)
    return env


def run(script: Path, cwd: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(script), *args], cwd=str(cwd), env=env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)


# ---------------------------------------------------------------- step 1
def check_records() -> None:
    entries = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        digest, size, rel = line.split(None, 2)
        if not rel.startswith(MANIFEST_PREFIX):
            record("1 records manifest", False, f"unexpected manifest path {rel}")
            return
        entries[rel[len(MANIFEST_PREFIX):]] = (digest, int(size))
    on_disk = {p.relative_to(RECORDS).as_posix(): p for p in RECORDS.glob("*/*.json")}
    missing = sorted(entries.keys() - on_disk.keys())
    extra = sorted(on_disk.keys() - entries.keys())
    bad = []
    for rel, path in sorted(on_disk.items()):
        if rel in entries:
            digest, size = entries[rel]
            if path.stat().st_size != size or sha256(path) != digest:
                bad.append(rel)
        try:
            with path.open(encoding="utf-8-sig") as stream:
                json.load(stream)
        except ValueError:
            bad.append(rel + " (not valid JSON)")
    ok = not (missing or extra or bad) and len(entries) == N_RECORDS
    detail = (f"{len(on_disk)} JSON files; SHA-256 and size match the manifest"
              if ok else f"missing={missing} extra={extra} mismatched={bad} listed={len(entries)}")
    record("1 records manifest", ok, detail)


# ---------------------------------------------------------------- step 2
def csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return sum(1 for _ in csv.DictReader(stream))


def check_decoders() -> None:
    with tempfile.TemporaryDirectory(prefix="verify-decoders-") as tmp:
        out = Path(tmp)
        env = child_env(DECODER_OUTPUT_ROOT=str(out))
        for script, _gen, _dep in DECODERS:
            done = run(ROOT / script, Path(tmp), env, "--quiet")
            if done.returncode:
                hint = ("\n    On Windows, a FileNotFoundError here usually means a full file path "
                        "exceeds 260 characters; move the repository to a shorter path.")
                record(f"2 decode {Path(script).name}", False,
                       "decoder exited with an error:\n" + text(done.stderr + done.stdout) + hint)
        total_files = total_rows = 0
        for script, gen, dep in DECODERS:
            got = {p.name: p for p in (out / gen).glob("*.csv")}
            want = {p.name: p for p in (KEYS / dep / "ai").glob("*.csv")}
            missing = sorted(want.keys() - got.keys())
            extra = sorted(got.keys() - want.keys())
            changed = sorted(n for n in got.keys() & want.keys()
                             if got[n].read_bytes().replace(b"\r\n", b"\n")
                             != want[n].read_bytes().replace(b"\r\n", b"\n"))
            ok = bool(want) and not (missing or extra or changed)
            rows = sum(csv_rows(p) for p in got.values())
            total_files += len(got)
            total_rows += rows
            detail = (f"{len(got)} CSVs, {rows} rows; content-identical to keys/{dep}/ai "
                      "after line-ending normalisation" if ok else
                      f"missing={missing} extra={extra} changed={changed}")
            record(f"2 decode {dep}", ok, detail)
        print(f"    decoder total: {total_files} CSV files, {total_rows} rows")


# ---------------------------------------------------------------- step 2r
REFERENCE = ROOT / "2_conversion" / "reference_side"
RAW = ROOT / "0_sources" / "raw"
# dataset, builder, deposited key folder, raw file
REF_BUILDERS = [
    ("loladze", "build_reference_keys_loladze.py", "loladze_v2", "loladze/CO2 Dataset.xlsx"),
    ("li_j", "build_reference_keys_li_j.py", "li2022_v2", "li_j/Data_Sheet_2.XLSX"),
    ("boldorini", "build_reference_keys_boldorini.py", "boldorini", "boldorini/rspb20232522_si_005.csv"),
    ("hui", "build_reference_keys_hui.py", "hui_v4", "hui/41467_2025_58397_MOESM5_ESM.xlsx"),
]
# Documented difference: two Boldorini rows had their crop label corrected by hand after the
# conversion script ran; the script does not encode that edit.
BOLDORINI_ALLOWED = {"gt__B06_Garfinkel_2020__id_44": {"crop", "evidence"},
                     "gt__B06_Garfinkel_2020__id_45": {"crop", "evidence"}}
TRACE_FLAGS = {"MATCH", "MATCH_EXACT_UNIQUE", "MATCH_EXACT_MULTIPLE_RAW_ROWS",
               "MATCH_UP_TO_UNIT_FACTOR_0.001"}
TRACES = [("loladze", "loladze_v2"), ("li_j", "li2022_v2"), ("hui", "hui_v4"),
          ("boldorini", "boldorini"), ("biochar", "biochar_v2")]


def key_rows(folder: Path) -> dict:
    out = {}
    for path in sorted(folder.glob("*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                out[row["row_id"]] = row
    return out


def check_reference_side() -> None:
    with tempfile.TemporaryDirectory(prefix="verify-reference-") as tmp:
        for ds, script, dep, raw in REF_BUILDERS:
            label = f"2r rebuild reference {dep}"
            if not (RAW / raw).exists():
                if ds == "hui":
                    record(label, False, "workbook not present; to run this check, download it as "
                           "described in 0_sources/raw/hui/DOWNLOAD.txt", skip=True)
                else:
                    record(label, False, f"published file missing: 0_sources/raw/{raw}")
                continue
            out = Path(tmp) / ds
            done = run(REFERENCE / script, Path(tmp), child_env(), "--raw", str(RAW / raw),
                       "--out", str(out))
            if done.returncode:
                record(label, False, "builder failed:\n" + text(done.stderr + done.stdout))
                continue
            want_dir = KEYS / dep / "gt"
            got = {p.name: p.read_bytes().replace(b"\r\n", b"\n") for p in out.glob("*.csv")}
            want = {p.name: p.read_bytes().replace(b"\r\n", b"\n") for p in want_dir.glob("*.csv")}
            changed = sorted(n for n in want if n in got and got[n] != want[n])
            missing, extra = sorted(want.keys() - got.keys()), sorted(got.keys() - want.keys())
            rows_want, rows_got = key_rows(want_dir), key_rows(out)
            if ds != "boldorini":
                ok = not (missing or extra or changed)
                detail = (f"{len(got)} files, {len(rows_got)} rows; content-identical to keys/{dep}/gt "
                          "after line-ending normalisation" if ok else
                          f"missing={missing} extra={extra} changed={changed}")
                record(label, ok, detail)
                continue
            diffs = {}
            for rid, row in rows_want.items():
                other = rows_got.get(rid)
                cols = {c for c in row if other is None or row[c] != other.get(c)}
                if cols:
                    diffs[rid] = cols
            ok = (not missing and not extra and len(rows_got) == len(rows_want)
                  and diffs == BOLDORINI_ALLOWED)
            same = len(rows_want) - len(diffs)
            detail = (f"{len(got)} files, {len(rows_got)} rows; {same} rows identical; the 2 documented "
                      "Garfinkel 2020 rows differ only in crop and evidence" if ok else
                      f"unexpected differences: missing={missing} extra={extra} rows={diffs}")
            record(label, ok, detail)
    for ds, dep in TRACES:
        trace_path = REFERENCE / "row_trace" / f"{ds}_row_trace.csv"
        with trace_path.open(encoding="utf-8", newline="") as stream:
            trace = list(csv.DictReader(stream))
        by_id = {}
        for row in trace:
            by_id.setdefault(row["row_id"], []).append(row)
        keys = key_rows(KEYS / dep / "gt")
        bad = [rid for rid, row in keys.items()
               if len(by_id.get(rid, [])) != 1
               or by_id[rid][0]["match_flag"] not in TRACE_FLAGS
               or by_id[rid][0]["stored_treatment_mean"] != row["treatment_mean"]
               or by_id[rid][0]["stored_control_mean"] != row["control_mean"]]
        extra = sorted(set(by_id) - set(keys))
        ok = not bad and not extra
        detail = (f"trace check (not a rebuild): all {len(keys)} reference key rows appear once in "
                  f"row_trace/{trace_path.name} with a match flag and the same means" if ok else
                  f"{len(bad)} key rows without a clean trace (first: {bad[:3]}); "
                  f"{len(extra)} extra trace rows")
        record(f"2r trace reference {dep}", ok, detail)


# ---------------------------------------------------------------- steps 3 and 4
def copy_repo(dest: Path) -> Path:
    ignore = shutil.ignore_patterns(".git", "__pycache__", "generated_keys", "*.pyc")
    shutil.copytree(ROOT, dest, ignore=ignore)
    for _script, png, _exp in FIGURES:
        (dest / png).unlink()
    for rel in COVERAGE_OUTPUTS:
        (dest / rel).unlink()
    return dest


def check_analyses(copy: Path, workdir: Path) -> None:
    env = child_env()
    for script, expected in ANALYSES:
        name = Path(script).name
        done = run(copy / script, workdir, env)
        if done.returncode:
            record(f"3 {name}", False, "script failed:\n" + text(done.stderr))
            continue
        want = text((EXPECTED / expected).read_bytes())
        got = text(done.stdout)
        if got == want:
            record(f"3 {name}", True, f"output matches {expected}")
        else:
            diff = [f"    expected: {a!r}\n    actual:   {b!r}"
                    for a, b in zip(want.splitlines(), got.splitlines()) if a != b]
            if len(want.splitlines()) != len(got.splitlines()):
                diff.append("    (line counts differ)")
            record(f"3 {name}", False, f"output differs from {expected}\n" + "\n".join(diff[:10]))
    for rel in COVERAGE_OUTPUTS:
        new, old = copy / rel, ROOT / rel
        ok = new.exists() and text(new.read_bytes()) == text(old.read_bytes())
        record(f"3 {Path(rel).name}", ok,
               "regenerated; content identical to deposited file" if ok
               else "not regenerated or content differs from deposited file")


def same_pixels(a: Path, b: Path) -> bool | None:
    try:
        import matplotlib.image as mpimg
        import numpy as np
    except ImportError:
        return None
    x, y = mpimg.imread(str(a)), mpimg.imread(str(b))
    return x.shape == y.shape and bool(np.array_equal(x, y))


def check_figures(copy: Path, workdir: Path, allow_image_differences: bool) -> None:
    env = child_env()
    for script, png, expected in FIGURES:
        label = f"4 {Path(png).name}"
        if script is not None:
            done = run(copy / script, workdir, env)
            if done.returncode:
                record(label, False, "script failed:\n" + text(done.stderr))
                continue
            if expected is not None:
                # The first printed line is "wrote <absolute path>"; it depends on location.
                got = "".join(l for l in text(done.stdout).splitlines(True)
                              if not l.startswith("wrote "))
                want = text((FIG_EXPECTED / expected).read_bytes())
                record(f"4 {Path(script).name} values", got == want,
                       f"printed values match {expected}" if got == want
                       else f"printed values differ from {expected}")
        new, old = copy / png, ROOT / png
        if not new.exists():
            record(label, False, "figure was not produced")
            continue
        if new.read_bytes() == old.read_bytes():
            record(label, True, "produced; file identical to deposited figure")
            continue
        pixels = same_pixels(new, old)
        if pixels:
            record(label, True, "produced; decoded pixels identical to deposited figure")
        elif allow_image_differences:
            record(label, False,
                   "produced, but the image differs from the deposited figure "
                   "(allowed by --allow-image-differences)", warn=True)
        else:
            record(label, False,
                   "produced, but the image differs from the deposited figure. If your "
                   "matplotlib version differs from 3.10.8, rerun with "
                   "--allow-image-differences")


def main() -> int:
    # Never fail on printing non-ASCII names (for example on a Windows cp1252 console).
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--allow-image-differences", action="store_true",
                        help="report figure image differences as warnings, not failures")
    parser.add_argument("--keep-temp", action="store_true",
                        help="keep the temporary repository copy and print its location")
    args = parser.parse_args()

    print(f"Repository: {ROOT.name}  Python {sys.version.split()[0]}\n")
    check_records()
    check_decoders()
    check_reference_side()

    tmp = Path(tempfile.mkdtemp(prefix="verify-run-"))
    try:
        copy = copy_repo(tmp / "repo")
        workdir = tmp / "elsewhere"
        workdir.mkdir()
        check_analyses(copy, workdir)
        check_figures(copy, workdir, args.allow_image_differences)
    except OSError as exc:
        record("3-4 temporary copy", False,
               f"could not copy or run the repository in a temporary folder: {exc}. On Windows "
               "this usually means a full file path exceeds 260 characters.")
    finally:
        if args.keep_temp:
            print(f"\nTemporary copy kept at {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)

    width = max(len(c) for c, _s, _d in RESULTS)
    print("\n" + "=" * (width + 12))
    print(f"{'check':<{width}}  result")
    print("-" * (width + 12))
    for check, status, _detail in RESULTS:
        print(f"{check:<{width}}  {status}")
    print("=" * (width + 12))
    failed = [c for c, s, _d in RESULTS if s == "FAIL"]
    warned = [c for c, s, _d in RESULTS if s == "WARN"]
    skipped = [c for c, s, _d in RESULTS if s == "SKIPPED"]
    notes = []
    if skipped:
        notes.append(f"{len(skipped)} skipped")
    if warned:
        notes.append(f"{len(warned)} figure image warning(s)")
    note = f" ({'; '.join(notes)}; see above)" if notes else ""
    if failed:
        print(f"FAIL: {len(failed)} of {len(RESULTS)} checks failed{note}.")
        return 1
    ran = len(RESULTS) - len(skipped)
    print(f"PASS: all {ran} checks that ran passed{note}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
