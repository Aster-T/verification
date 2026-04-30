#!/usr/bin/env python
"""LOO baseline sweep on the local ship-* datasets, one folder per model.

For each `datasets/ship-*/` directory with a `meta.json`, run row probe
in LOO mode with k=1 and seeds 1,2,3 — but split into THREE separate
passes per dataset, one per model variant, each landing in its own
top-level results folder so different models never share a metrics.jsonl:

    results/mlr/<dataset>/row/...           (MLR only,    --no-tabpfn)
    results/tabpfn_v2_6/<dataset>/row/...   (TabPFN v2.6, --no-mlr)
    results/tabpfn_v2/<dataset>/row/...     (TabPFN v2,   --no-mlr)

`k=1 + LOO` is the "no duplication" baseline. Anchor preservation makes
k=1+jitter numerically identical to exact, so we record exact only —
flip MODES to "exact,jitter" if you specifically want jitter rows.

Output flags used:
  MLR pass:   --out results/mlr        --no-tabpfn
  v2.6 pass:  --out results/tabpfn_v2_6 --no-mlr --tabpfn-weights v2_6
  v2 pass:    --out results/tabpfn_v2   --no-mlr --tabpfn-weights v2
              --no-weights-partition          (suppresses the auto
                                               weights_v2/ subdir so the
                                               output lands directly under
                                               results/tabpfn_v2/)

Edit the constants at the top to override. Errors abort the whole run
(`subprocess.run(..., check=True)`).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent

# ── 控制是否强制重跑(True = 重做,False = 跳过已有结果) ─────────────────
FRESH = False

# Auto-discover every `datasets/ship-*/meta.json` so adding a new ship
# folder is enough — no edit needed here. Sorted for stable run order.
SHIP_DATASETS: list[str] = sorted(
    p.parent.name
    for p in (REPO / "datasets").glob("ship-*/meta.json")
)

SEEDS = [1, 2, 3]
K_LIST = "1"
MODES = "exact"

# Three independent passes per dataset. Each entry:
#   (out-folder, model-flag, *extra args)
# The model-flag picks exactly one model; the extras pin the TabPFN
# checkpoint when applicable. Drop a row to skip that pass.
PASSES: list[tuple[str, list[str]]] = [
    ("mlr",          ["--no-tabpfn"]),
    ("tabpfn_v2_6",  ["--no-mlr", "--tabpfn-weights", "v2_6"]),
    ("tabpfn_v2",    ["--no-mlr", "--tabpfn-weights", "v2",
                      "--no-weights-partition"]),
]

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


def main() -> int:
    if not SHIP_DATASETS:
        print(
            "no ship-*/ datasets found under "
            f"{(REPO / 'datasets').as_posix()} — nothing to do.",
            file=sys.stderr,
        )
        return 1

    total = len(PASSES) * len(SHIP_DATASETS)
    pass_names = [name for name, _ in PASSES]
    banner(
        f"LOO baseline · k=1 · modes={MODES} · seeds={SEEDS_CSV} · "
        f"passes={','.join(pass_names)} · {total} run(s)"
    )
    print(f"  datasets: {', '.join(SHIP_DATASETS)}")

    combo = 0
    for out_name, extra_args in PASSES:
        banner(f"PASS = {out_name}", char="█", width=80)
        out_root = REPO / "results" / out_name
        for ds in SHIP_DATASETS:
            combo += 1
            banner(
                f"({combo}/{total})  {out_name}  ·  dataset={ds}",
                char="=", width=72,
            )
            run(
                [
                    PY, "scripts/run_row_probe.py",
                    "--dataset", ds,
                    "--split-mode", "loo",
                    "--k-list", K_LIST,
                    "--modes", MODES,
                    "--seeds", SEEDS_CSV,
                    "--out", str(out_root),
                    *extra_args,
                    *FRESH_FLAG,
                    "-v",
                ]
            )

    print()
    print("=================== ALL DONE ===================")
    print(f"Datasets ran: {', '.join(SHIP_DATASETS)}")
    print(f"Passes: {', '.join(pass_names)}")
    print(f"Seeds: {SEEDS_CSV}   k_list: {K_LIST}   modes: {MODES}")
    print()
    print("Results landed at (one folder per model, no record mixing):")
    for name, _ in PASSES:
        for ds in SHIP_DATASETS:
            print(f"  results/{name}/{ds}/row/metrics.jsonl")
    print()
    print("To regenerate PNGs and view in browser:")
    print("  python scripts/rebuild_reports.py -v")
    print("  python scripts/serve_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
