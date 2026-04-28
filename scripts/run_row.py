#!/usr/bin/env python
"""Orchestrate the row-probe sweep across (jitter_scale × sigma × test_size).

For each jitter_scale in JITTER_SCALES:
  For each sigma in JITTER_SIGMAS:
    1. LOO once on the local datasets.
    2. For each test_size in TEST_SIZES, proportional split on the OpenML datasets.
At the end, regenerate every PNG and print where to find the live report.

Edit the constants at the top of the file. Errors from any subprocess
abort the whole run (subprocess.run(..., check=True)).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

# ── 控制是否强制重跑（True = 重做，False = 跳过已有结果）─────────────────
FRESH = False

LOCAL_DATASETS = ["ship-all", "ship-tanker", "ship-selected"]
OPENML_DATASETS = [
    "airfoil-self-noise",
    "auction-verification",
    "concrete-compressive-strength",
    "forest-fires",
]

# Two jitter-scale ablations (see src/probing/row_probe.py::duplicate_context):
#   "absolute"    -> noise = N(0, σ²) for every numeric cell (legacy).
#   "per_col_std" -> noise per column scaled by that column's std on the
#                    untiled X — σ becomes a relative perturbation strength.
# Each scale lands in a disjoint subtree (jitter_<scale>/ for non-default),
# so the two coexist as independent ablation runs.
JITTER_SCALES = ["absolute", "per_col_std"]
JITTER_SIGMAS = ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]
TEST_SIZES = ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]
SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
K_LIST = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"

PY = os.environ.get("PY", sys.executable)
SEEDS_CSV = ",".join(str(s) for s in SEEDS)
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
    local_args = [a for d in LOCAL_DATASETS for a in ("--dataset", d)]
    openml_args = [a for d in OPENML_DATASETS for a in ("--openml-preset", d)]

    total = len(JITTER_SCALES) * len(JITTER_SIGMAS) * len(TEST_SIZES)
    combo = 0

    for scale in JITTER_SCALES:
        banner(f"JITTER_SCALE = {scale}", char="█")
        scale_args = ["--jitter-scale", scale]

        for sigma in JITTER_SIGMAS:
            banner(f"scale={scale}  ·  SIGMA = {sigma}")

            print()
            print(f"  ---- LOO (once per scale·sigma):  {' '.join(LOCAL_DATASETS)} ----")
            run(
                [
                    PY,
                    "scripts/run_row_probe.py",
                    *local_args,
                    "--split-mode",
                    "loo",
                    "--k-list",
                    K_LIST,
                    "--seeds",
                    SEEDS_CSV,
                    "--jitter-sigma",
                    sigma,
                    *scale_args,
                    *FRESH_FLAG,
                    "-v",
                ]
            )

            for test_size in TEST_SIZES:
                combo += 1
                section(
                    f"({combo}/{total})  scale={scale}  sigma={sigma}  "
                    f"test_size={test_size}  seeds={SEEDS_CSV}"
                )

                print()
                print(
                    f"  ---- PROP: {' '.join(OPENML_DATASETS)}  "
                    f"test_size={test_size} ----"
                )
                run(
                    [
                        PY,
                        "scripts/run_row_probe.py",
                        *openml_args,
                        "--split-mode",
                        "proportional",
                        "--test-size",
                        test_size,
                        "--k-list",
                        K_LIST,
                        "--seeds",
                        SEEDS_CSV,
                        "--jitter-sigma",
                        sigma,
                        *scale_args,
                        *FRESH_FLAG,
                        "-v",
                    ]
                )

    print()
    print("=" * 80)
    print("  Regenerating PNGs across all (scale × sigma) subtrees (single pass)")
    print("=" * 80)
    run([PY, "scripts/rebuild_reports.py", "-v"])

    print()
    print("=================== ALL DONE ===================")
    print(
        f"Total combos: {total}  "
        f"(jitter_scale × sigma × test_size = "
        f"{len(JITTER_SCALES)} × {len(JITTER_SIGMAS)} × {len(TEST_SIZES)})"
    )
    print(f"Jitter scales: {', '.join(JITTER_SCALES)}")
    print(f"Seeds per combo: {SEEDS_CSV}  (N={len(SEEDS)})")
    print()
    print("View the unified report (live, on-demand) with:")
    print("  python scripts/serve_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
