#!/usr/bin/env python3
"""Verify finalized JSON source records and reproduce the frozen AI key tables."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "source_records"
MANIFEST = SOURCE_ROOT / "SHA256SUMS.txt"

DATASETS = {
    "biochar": (34, "biochar", ROOT / "runs/biochar_v2/keys/ai", 28, 446),
    "boldorini": (18, "boldorini", ROOT / "runs/boldorini/keys/ai", 18, 80),
    "hui": (37, "hui_strict", ROOT / "runs/hui_v4/keys/ai", 29, 515),
    "li_j": (49, "li_j", ROOT / "runs/li2022_v2/keys/ai", 49, 464),
    "loladze": (46, "loladze", ROOT / "runs/loladze_v2/keys/ai", 46, 1646),
}

DECODERS = (
    "decoders/decode_biochar.py",
    "decoders/decode_hui.py",
    "decoders/decode_li_j.py",
    "decoders/decode_loladze.py",
    "decoders/decode_boldorini_march_v11.py",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_files() -> list[Path]:
    return sorted(SOURCE_ROOT.glob("*/*.json"), key=lambda p: p.as_posix())


def manifest_text() -> str:
    lines = ["# SHA-256, byte size, and repository-relative path for 184 finalized JSON records."]
    for path in source_files():
        rel = path.relative_to(ROOT).as_posix()
        lines.append(f"{sha256(path)}  {path.stat().st_size}  {rel}")
    return "\n".join(lines) + "\n"


def verify_manifest() -> None:
    expected = manifest_text()
    actual = MANIFEST.read_text(encoding="utf-8")
    if actual != expected:
        raise AssertionError("source_records/SHA256SUMS.txt is stale or incorrect")


def count_csv_rows(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            total += sum(1 for _ in csv.DictReader(stream))
    return total


def normalized_csv_sha256(path: Path) -> str:
    """Hash CSV content independent of the historical CRLF/LF convention."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def compare_csv_trees(generated: Path, frozen: Path) -> tuple[int, int]:
    got = {p.name: normalized_csv_sha256(p) for p in generated.glob("*.csv")}
    want = {p.name: normalized_csv_sha256(p) for p in frozen.glob("*.csv")}
    if got != want:
        missing = sorted(want.keys() - got.keys())
        extra = sorted(got.keys() - want.keys())
        changed = sorted(k for k in got.keys() & want.keys() if got[k] != want[k])
        raise AssertionError(
            f"decoder mismatch for {generated.name}: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    files = sorted(generated.glob("*.csv"))
    return len(files), count_csv_rows(files)


def verify_sources() -> tuple[int, int]:
    paths = source_files()
    if len(paths) != 184:
        raise AssertionError(f"expected 184 JSON files, found {len(paths)}")
    total_bytes = 0
    for dataset, (expected_count, *_rest) in DATASETS.items():
        records = sorted((SOURCE_ROOT / dataset).glob("*.json"))
        if len(records) != expected_count:
            raise AssertionError(
                f"{dataset}: expected {expected_count} JSON files, found {len(records)}"
            )
        for path in records:
            with path.open(encoding="utf-8-sig") as stream:
                json.load(stream)
            total_bytes += path.stat().st_size
    verify_manifest()
    return len(paths), total_bytes


def verify_decoders() -> tuple[int, int]:
    with tempfile.TemporaryDirectory(prefix="source-record-verify-") as tmp:
        output_root = Path(tmp)
        env = os.environ.copy()
        env["DECODER_OUTPUT_ROOT"] = str(output_root)
        for decoder in DECODERS:
            completed = subprocess.run(
                [sys.executable, str(ROOT / decoder), "--quiet"],
                cwd=ROOT,
                env=env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if completed.returncode:
                raise RuntimeError(f"{decoder} failed:\n{completed.stdout}")

        file_total = 0
        row_total = 0
        for dataset, (_n, generated_name, frozen, expected_files, expected_rows) in DATASETS.items():
            files, rows = compare_csv_trees(output_root / generated_name, frozen)
            if (files, rows) != (expected_files, expected_rows):
                raise AssertionError(
                    f"{dataset}: expected {expected_files} files/{expected_rows} rows, "
                    f"found {files} files/{rows} rows"
                )
            print(
                f"{dataset}: {files} generated CSVs, {rows} rows, content-identical "
                "to frozen AI keys after CSV newline normalization"
            )
            file_total += files
            row_total += rows
        return file_total, row_total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="rewrite source_records/SHA256SUMS.txt from the current JSON files",
    )
    parser.add_argument("--skip-decoders", action="store_true")
    args = parser.parse_args()

    if args.write_manifest:
        MANIFEST.write_text(manifest_text(), encoding="utf-8", newline="\n")

    files, size = verify_sources()
    print(f"source records: {files} valid JSON files, {size} bytes, manifest verified")
    if not args.skip_decoders:
        csv_files, rows = verify_decoders()
        print(f"decoder rebuild: {csv_files} CSV files and {rows} rows verified")


if __name__ == "__main__":
    main()
