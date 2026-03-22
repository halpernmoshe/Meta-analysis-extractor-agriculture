#!/usr/bin/env python
"""
reproduce_everything.py -- Master script to reproduce all validation results.

Runs, in order:
  1. Checks that all required data files exist
  2. reproduce_all.py          (stats verification against paper claims)
  3. formal_stats_all_datasets.py  (formal statistical analysis)
  4. generate_figures.py       (publication figures)
  5. build_docx.py             (Word document)

Each step is run in-process.  If one step fails the others still execute.
A summary report is printed at the end.

Usage:
    python reproduce_everything.py
    ./venv/Scripts/python.exe reproduce_everything.py
"""
import sys
import os
import time
import traceback
from pathlib import Path
from datetime import datetime

# ── Windows encoding fix ──────────────────────────────────────────────────
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent

# ── Required data files ───────────────────────────────────────────────────
# Each entry: (relative_path_from_BASE_DIR, description, critical?)
REQUIRED_FILES = [
    # Loladze 2014
    ("data/loladze_agent_replication/validation_llm_10pp.json",
     "Loladze LLM-aligned matched observations (413 obs)", True),

    # Hui 2023
    ("data/hui2023_full_35/validation_matches_improved.csv",
     "Hui improved matched observations (319 obs)", True),
    ("data/hui2023_full_35/validation_matches.csv",
     "Hui original matches (paper_id column)", False),

    # Li 2022
    ("data/li2022_combined/validation_matches_effect_first.csv",
     "Li 2022 effect-first matched observations (117 obs)", True),

    # Biochar
    ("data/biochar_extraction/validation_results.json",
     "Biochar validation results (254 obs)", True),

    # Boldorini
    ("data/boldorini_extraction/validation_results.json",
     "Boldorini validation results (46 obs)", True),

    # Boldorini GT (needed by reproduce_all.py which re-matches)
    ("data/boldorini_gt/boldorini_gt.csv",
     "Boldorini ground-truth CSV", True),

    # Boldorini extraction JSONs (needed by reproduce_all.py)
    ("data/boldorini_extraction/B01_Ali_2018.json",
     "Boldorini extraction JSON (sample)", True),

    # Paper markdown (needed by build_docx.py)
    ("PAPER_FINAL_v23.md",
     "Paper markdown v23", True),
]


def check_data_files():
    """Check all required data files and report status."""
    print("=" * 70)
    print("  STEP 0: Checking required data files")
    print("=" * 70)

    missing_critical = []
    missing_optional = []

    for rel_path, description, critical in REQUIRED_FILES:
        full_path = (BASE_DIR / rel_path).resolve()
        exists = full_path.exists()
        status = "OK" if exists else ("MISSING (critical)" if critical else "MISSING (optional)")
        icon = "  " if exists else ">>"
        print(f"  {icon} [{status:>20s}] {rel_path}")
        if not exists:
            if critical:
                missing_critical.append((rel_path, description))
            else:
                missing_optional.append((rel_path, description))

    print()
    if missing_critical:
        print(f"  WARNING: {len(missing_critical)} critical file(s) missing.")
        print("  Some steps may fail or produce incomplete results.")
        for p, d in missing_critical:
            print(f"    - {p}  ({d})")
    else:
        print("  All critical data files present.")

    if missing_optional:
        print(f"  Note: {len(missing_optional)} optional file(s) missing (non-fatal).")

    print()
    return len(missing_critical) == 0


def run_step(step_num, name, func):
    """Run a step, capturing success/failure and elapsed time."""
    print()
    print("=" * 70)
    print(f"  STEP {step_num}: {name}")
    print("=" * 70)
    t0 = time.time()
    try:
        func()
        elapsed = time.time() - t0
        print(f"\n  Step {step_num} completed in {elapsed:.1f}s")
        return True, elapsed, None
    except SystemExit as e:
        elapsed = time.time() - t0
        # reproduce_all.py uses sys.exit(0/1)
        if e.code == 0:
            print(f"\n  Step {step_num} completed in {elapsed:.1f}s")
            return True, elapsed, None
        else:
            msg = f"Exited with code {e.code}"
            print(f"\n  Step {step_num} FAILED ({msg}) in {elapsed:.1f}s")
            return False, elapsed, msg
    except Exception:
        elapsed = time.time() - t0
        msg = traceback.format_exc()
        print(f"\n  Step {step_num} FAILED in {elapsed:.1f}s")
        print(msg)
        return False, elapsed, msg


# ── Step functions ────────────────────────────────────────────────────────

def step_reproduce_all():
    """Run reproduce_all.py (stats verification)."""
    script = BASE_DIR / "reproduce_all.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    # Import and run main()
    import importlib.util
    spec = importlib.util.spec_from_file_location("reproduce_all", str(script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def step_formal_stats():
    """Run formal_stats_all_datasets.py."""
    script = BASE_DIR / "formal_stats_all_datasets.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("formal_stats", str(script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def step_generate_figures():
    """Run generate_figures.py."""
    script = BASE_DIR / "generate_figures.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("generate_figures", str(script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def step_build_docx():
    """Run build_docx.py."""
    script = BASE_DIR / "build_docx.py"
    if not script.exists():
        raise FileNotFoundError(f"Script not found: {script}")
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_docx", str(script))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.build_docx()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print()
    print("#" * 70)
    print("#  REPRODUCE EVERYTHING -- Meta-Analysis Extractor Validation")
    print(f"#  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"#  Base directory: {BASE_DIR}")
    print("#" * 70)
    print()

    all_files_ok = check_data_files()

    steps = [
        (1, "reproduce_all.py  (stats verification)", step_reproduce_all),
        (2, "formal_stats_all_datasets.py  (formal stats)", step_formal_stats),
        (3, "generate_figures.py  (publication figures)", step_generate_figures),
        (4, "build_docx.py  (Word document)", step_build_docx),
    ]

    results = []
    for num, name, func in steps:
        ok, elapsed, err = run_step(num, name, func)
        results.append((num, name, ok, elapsed, err))

    # ── List output files ─────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  OUTPUT FILES CREATED")
    print("=" * 70)

    output_files = [
        "output/reproduction_results.json",
        "output/formal_stats_all_datasets.json",
        "figures/figure1_architecture.png",
        "figures/figure2_scatter_plots.png",
        "figures/figure3_cross_dataset_comparison.png",
        "figures/figure4_bland_altman.png",
        "figures/figure5_source_type_accuracy.png",
        "figures/figure6_aggregate_effects.png",
        "PAPER_FINAL_v23.docx",
    ]

    for rel in output_files:
        full = BASE_DIR / rel
        if full.exists():
            size = full.stat().st_size
            print(f"    {rel}  ({size:,} bytes)")
        else:
            print(f"    {rel}  [NOT FOUND]")

    # ── Summary ───────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    total_time = sum(e for _, _, _, e, _ in results)
    n_pass = sum(1 for _, _, ok, _, _ in results if ok)
    n_fail = sum(1 for _, _, ok, _, _ in results if not ok)

    for num, name, ok, elapsed, err in results:
        status = "PASS" if ok else "FAIL"
        print(f"  Step {num}: {status}  ({elapsed:.1f}s)  {name}")

    print()
    print(f"  Total: {n_pass}/{len(results)} passed, {n_fail} failed")
    print(f"  Total time: {total_time:.1f}s")

    if n_fail > 0:
        print()
        print("  FAILURES:")
        for num, name, ok, _, err in results:
            if not ok and err:
                short_err = err.strip().split("\n")[-1][:120]
                print(f"    Step {num}: {short_err}")

    print()
    if n_fail == 0:
        print("  ALL STEPS PASSED. Reproduction complete.")
    else:
        print(f"  {n_fail} step(s) failed. See output above for details.")

    print("=" * 70)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
