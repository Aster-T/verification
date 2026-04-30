#!/usr/bin/env python
"""Regenerate every PNG under results/ from the existing metrics.jsonl /
column.npz artefacts. Does NOT re-fit any model — pure visualization
refresh. Run this after changing anything in src/viz/curves.py or
src/viz/heatmap.py and you'll see the new charts on the next page reload
of `scripts/serve_report.py`.

Usage:
    python scripts/rebuild_reports.py [--results-root results]

How it walks the tree:
  Recursive rglob for `row/metrics.jsonl` and `column/mlr.npz`. The
  parent of `row/` (or `column/`) is treated as the dataset directory;
  PNGs go to that dir's sibling `viz/`. This makes the script agnostic
  to nesting depth, so every layout that run_row_probe.py /
  run_column_probe.py can produce works:

    results/<ds>/row/...                                       (legacy unpartitioned, e.g. ship LOO baseline)
    results/sigma_<σ>/<ds>/row/...                             (LOO with σ)
    results/sigma_<σ>/test_size_<ts>/<ds>/row/...              (PROP)
    results/sigma_<σ>/jitter_<scale>/<ds>/row/...              (jitter ablation)
    results/sigma_<σ>/weights_<w>/<ds>/row/...                 (weights ablation)
    results/weights_<w>/<ds>/row/...                           (weights-only baseline)
    results/sigma_<σ>/[weights_<w>/][jitter_<scale>/][test_size_<ts>/]<ds>/row/...

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


def _path_tag(rel_parts: tuple[str, ...]) -> str:
    """Make a human-readable label from the path components between
    results_root and the dataset directory. Last component (the dataset
    name) is dropped since the caller logs it separately. Empty tuple
    means the dataset sits directly under results_root (legacy
    unpartitioned layout)."""
    bits = list(rel_parts[:-1]) if rel_parts else []
    return " · ".join(bits) if bits else "(root)"


def _regen_dataset(
    ds_dir: Path, results_root: Path,
) -> tuple[int, int, int]:
    """Rebuild PNGs for a single dataset directory. Looks for
    row/metrics.jsonl and column/mlr.npz; either or both may be present.
    Returns (row_rebuilt, column_rebuilt, errors)."""
    rc = hm = err = 0
    ds = ds_dir.name
    rel = ds_dir.relative_to(results_root).parts
    tag = _path_tag(rel)
    viz_dir = ds_dir / "viz"

    row_jsonl = ds_dir / "row" / "metrics.jsonl"
    if row_jsonl.exists():
        try:
            viz_dir.mkdir(parents=True, exist_ok=True)
            plot_row_curves(ds, row_jsonl, viz_dir)
            logger.info("[%s] %s row curves rebuilt", tag, ds)
            rc += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] %s row curves failed: %s", tag, ds, e)
            err += 1

    column_dir = ds_dir / "column"
    if (column_dir / "mlr.npz").exists():
        try:
            viz_dir.mkdir(parents=True, exist_ok=True)
            plot_column_heatmaps(ds, column_dir, viz_dir)
            logger.info("[%s] %s column heatmaps rebuilt", tag, ds)
            hm += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[%s] %s column heatmaps failed: %s", tag, ds, e,
            )
            err += 1

    return rc, hm, err


def discover_dataset_dirs(results_root: Path) -> list[Path]:
    """Every `row/metrics.jsonl` and `column/mlr.npz` under results_root
    contributes its grandparent (the dataset directory). Dedupe and
    sort by full path so the log order is stable."""
    if not results_root.exists():
        return []
    found: set[Path] = set()
    for marker in ("row/metrics.jsonl", "column/mlr.npz"):
        for p in results_root.rglob(marker):
            # marker is "row/metrics.jsonl" -> p.parent is the row/ dir,
            # p.parent.parent is the dataset dir.
            found.add(p.parent.parent)
    return sorted(found)


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--results-root", type=Path, default=REPO / "results")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.results_root)
    if not root.exists():
        sys.stderr.write(f"results-root does not exist: {root}\n")
        return 1

    ds_dirs = discover_dataset_dirs(root)
    if not ds_dirs:
        sys.stderr.write(
            f"no row/metrics.jsonl or column/mlr.npz found under {root}\n"
            f"(did the probe runs land somewhere else? "
            f"check `--results-root`)\n"
        )
        return 1

    print(f"Rebuilding PNGs under {root.resolve()}")
    print(f"Found {len(ds_dirs)} dataset directory/ies with artefacts.\n")

    total_rc = total_hm = total_err = 0
    for ds_dir in ds_dirs:
        rel = ds_dir.relative_to(root).as_posix()
        rc, hm, err = _regen_dataset(ds_dir, root)
        total_rc += rc
        total_hm += hm
        total_err += err
        marks = []
        if rc:
            marks.append(f"row×{rc}")
        if hm:
            marks.append(f"col×{hm}")
        if err:
            marks.append(f"err×{err}")
        if marks:
            print(f"  {rel}   [{' '.join(marks)}]")

    print()
    print(
        f"DONE — row-curve groups: {total_rc}  "
        f"column heatmaps: {total_hm}  errors: {total_err}"
    )
    print()
    print("Start the live report with:")
    print("  python scripts/serve_report.py")
    return 0 if total_err == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
