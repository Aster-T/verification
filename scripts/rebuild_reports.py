#!/usr/bin/env python
"""Regenerate every PNG under results/ from the existing metrics.jsonl /
column.npz artefacts. Does NOT re-fit any model — pure visualization
refresh. Run this after changing anything in src/viz/curves.py or
src/viz/heatmap.py and you'll see the new charts on the next page reload
of `scripts/serve_report.py`.

Usage:
    python scripts/rebuild_reports.py [--results-root results]

The HTML report itself is no longer generated as a static file: start
`scripts/serve_report.py` and the page is served live.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from src.viz.curves import plot_row_curves  # noqa: E402
from src.viz.heatmap import plot_column_heatmaps  # noqa: E402

logger = logging.getLogger(__name__)


def discover_sigmas(results_root: Path) -> list[Path]:
    if not results_root.exists():
        return []
    return sorted(
        p for p in results_root.iterdir()
        if p.is_dir() and p.name.startswith("sigma_")
    )


def regen_one_sigma(sigma_root: Path) -> tuple[int, int, int]:
    """Regenerate every viz dir under sigma_root that has artefacts.
    Returns (row_curves_done, heatmaps_done, errors)."""
    rc, hm, err = 0, 0, 0
    sigma = sigma_root.name[len("sigma_"):]

    # 1) sigma-level: LOO row probe + column probe
    for child in sorted(sigma_root.iterdir()):
        if not child.is_dir() or child.name.startswith("test_size_"):
            continue
        ds = child.name
        viz_dir = child / "viz"
        loo_jsonl = child / "row" / "metrics.jsonl"
        column_dir = child / "column"

        if loo_jsonl.exists():
            try:
                viz_dir.mkdir(parents=True, exist_ok=True)
                plot_row_curves(ds, loo_jsonl, viz_dir)
                logger.info("[%s] %s LOO curves rebuilt", sigma, ds)
                rc += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] %s LOO curves failed: %s", sigma, ds, e)
                err += 1

        if (column_dir / "mlr.npz").exists():
            try:
                viz_dir.mkdir(parents=True, exist_ok=True)
                plot_column_heatmaps(ds, column_dir, viz_dir)
                logger.info("[%s] %s column heatmaps rebuilt", sigma, ds)
                hm += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s] %s column heatmaps failed: %s", sigma, ds, e)
                err += 1

    # 2) per-test_size: proportional row probe
    for ts_child in sorted(sigma_root.iterdir()):
        if not ts_child.is_dir() or not ts_child.name.startswith("test_size_"):
            continue
        ts = ts_child.name[len("test_size_"):]
        for ds_child in sorted(ts_child.iterdir()):
            if not ds_child.is_dir():
                continue
            ds = ds_child.name
            viz_dir = ds_child / "viz"
            prop_jsonl = ds_child / "row" / "metrics.jsonl"
            if not prop_jsonl.exists():
                continue
            try:
                viz_dir.mkdir(parents=True, exist_ok=True)
                plot_row_curves(ds, prop_jsonl, viz_dir)
                logger.info("[%s ts=%s] %s PROP curves rebuilt",
                            sigma, ts, ds)
                rc += 1
            except Exception as e:  # noqa: BLE001
                logger.warning("[%s ts=%s] %s PROP curves failed: %s",
                               sigma, ts, ds, e)
                err += 1

    return rc, hm, err


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=REPO / "results")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.results_root)
    sigmas = discover_sigmas(root)
    if not sigmas:
        sys.stderr.write(f"no sigma_* subtrees under {root}\n")
        return 1

    print(f"Rebuilding PNGs under {root.resolve()}")
    print(f"Found {len(sigmas)} sigma subtree(s): {[s.name for s in sigmas]}\n")

    total_rc = total_hm = total_err = 0
    for sigma_root in sigmas:
        print(f"=== {sigma_root.name} ===")
        rc, hm, err = regen_one_sigma(sigma_root)
        total_rc += rc
        total_hm += hm
        total_err += err
        print(f"  row-curve groups: {rc}  column heatmaps: {hm}  errors: {err}")

    print()
    print(f"DONE — row-curve groups: {total_rc}  "
          f"column heatmaps: {total_hm}  errors: {total_err}")
    print()
    print("Start the live report with:")
    print("  python scripts/serve_report.py")
    return 0 if total_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
