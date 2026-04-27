#!/usr/bin/env python
"""Orchestrate the row-probe sweep across (sigma × test_size).

For each sigma in JITTER_SIGMAS:
  1. LOO once on the local datasets.
  2. For each test_size in TEST_SIZES, proportional split on the OpenML datasets.
  3. Build the per-sigma report.

Edit the constants at the top of the file. Errors from any subprocess
abort the whole run (subprocess.run(..., check=True)).
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

# ── 控制是否强制重跑（True = 重做，False = 跳过已有结果）─────────────────
FRESH = False

LOCAL_DATASETS  = ["ship-all", "ship-tanker", "ship-selected"]
OPENML_DATASETS = ["airfoil-self-noise", "auction-verification",
                   "concrete-compressive-strength", "forest-fires"]

JITTER_SIGMAS = ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]
TEST_SIZES    = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6",
                 "0.7", "0.8", "0.9"]
SEEDS         = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K_LIST        = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

PY         = os.environ.get("PY", sys.executable)
SEEDS_CSV  = ",".join(str(s) for s in SEEDS)
FRESH_FLAG = ["--fresh"] if FRESH else []


def run(cmd: list[str]) -> None:
    print("  $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def banner(text: str, char: str = "#", width: int = 80) -> None:
    print()
    print(char * width)
    print(f"{char}  {text}")
    print(char * width)


def section(text: str, char: str = "=", width: int = 56) -> None:
    print()
    print(char * width)
    print(f"  {text}")
    print(char * width)


def main() -> int:
    local_args  = [a for d in LOCAL_DATASETS  for a in ("--dataset", d)]
    openml_args = [a for d in OPENML_DATASETS for a in ("--openml-preset", d)]
    report_args = (
        [a for d in LOCAL_DATASETS  for a in ("--dataset", d)] +
        [a for d in OPENML_DATASETS for a in ("--openml-preset", d)]
    )

    total = len(JITTER_SIGMAS) * len(TEST_SIZES)
    combo = 0

    for sigma in JITTER_SIGMAS:
        banner(f"SIGMA = {sigma}")

        print()
        print(f"  ---- LOO (once per sigma):  {' '.join(LOCAL_DATASETS)} ----")
        run([
            PY, "scripts/run_row_probe.py",
            *local_args,
            "--split-mode", "loo",
            "--k-list", K_LIST,
            "--seeds", SEEDS_CSV,
            "--jitter-sigma", sigma,
            *FRESH_FLAG,
            "-v",
        ])

        for test_size in TEST_SIZES:
            combo += 1
            section(
                f"({combo}/{total})  sigma={sigma}  "
                f"test_size={test_size}  seeds={SEEDS_CSV}"
            )

            print()
            print(
                f"  ---- PROP: {' '.join(OPENML_DATASETS)}  "
                f"test_size={test_size} ----"
            )
            run([
                PY, "scripts/run_row_probe.py",
                *openml_args,
                "--split-mode", "proportional",
                "--test-size", test_size,
                "--k-list", K_LIST,
                "--seeds", SEEDS_CSV,
                "--jitter-sigma", sigma,
                *FRESH_FLAG,
                "-v",
            ])

        print()
        print(f"  ---- REPORT (sigma={sigma},  all test_sizes) ----")
        run([
            PY, "scripts/build_report.py",
            *report_args,
            "--jitter-sigma", sigma,
            "-v",
        ])

    print()
    print("=================== ALL DONE ===================")
    print(f"Total combos: {total}  (sigma × test_size)")
    print(f"Seeds per combo: {SEEDS_CSV}  (N={len(SEEDS)})")
    print()
    print("Reports (one per sigma; pick test_size from the in-page dropdown):")
    for sigma in JITTER_SIGMAS:
        print(f"  results/sigma_{sigma}/report.html")
    return 0


if __name__ == "__main__":
    sys.exit(main())
