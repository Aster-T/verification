#!/usr/bin/env python
"""LOO baseline sweep on the local ship-* datasets.

For each `datasets/ship-*/` directory with a `meta.json`, run row probe in
LOO mode with k=1 and seeds 1,2,3. This is the "no duplication" baseline:
context is the original (N-1) rows for each held-out point, no jitter
matters (anchor preservation makes k=1+jitter numerically identical to
exact, so we skip jitter to save half the calls).

Outputs land at the unpartitioned default — `results/<dataset>/row/...` —
since no σ / test_size / jitter_scale / tabpfn_weights override is passed.
That keeps the baseline tidy and out of the per-σ ablation subtrees.

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
# k=1 + jitter == exact under the anchor-preservation rule in
# duplicate_context (the only "tile" IS the anchor, so no perturbation
# happens). Skip jitter to halve the per-fold work; flip to
# "exact,jitter" if you specifically want both modes recorded.
MODES = "exact"

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

    banner(
        f"LOO baseline · k=1 · modes={MODES} · seeds={SEEDS_CSV} · "
        f"{len(SHIP_DATASETS)} dataset(s)"
    )
    print(f"  datasets: {', '.join(SHIP_DATASETS)}")

    for ds in SHIP_DATASETS:
        banner(f"DATASET = {ds}", char="=", width=72)
        run(
            [
                PY, "scripts/run_row_probe.py",
                "--dataset", ds,
                "--split-mode", "loo",
                "--k-list", K_LIST,
                "--modes", MODES,
                "--seeds", SEEDS_CSV,
                *FRESH_FLAG,
                "-v",
            ]
        )

    print()
    print("=================== ALL DONE ===================")
    print(f"Datasets ran: {', '.join(SHIP_DATASETS)}")
    print(f"Seeds: {SEEDS_CSV}   k_list: {K_LIST}   modes: {MODES}")
    print()
    print("Results landed at (unpartitioned baseline tree):")
    for ds in SHIP_DATASETS:
        print(f"  results/{ds}/row/metrics.jsonl")
    print()
    print("To regenerate PNGs and view in browser:")
    print("  python scripts/rebuild_reports.py -v")
    print("  python scripts/serve_report.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
