"""CLI entry: render heatmaps + row curves + HTML report for given datasets."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from src.configs import add_openml_cli_args, resolve_openml_args  # noqa: E402
from src.viz.curves import plot_row_curves  # noqa: E402
from src.viz.heatmap import plot_column_heatmaps  # noqa: E402
from src.viz.report import build_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", default=[])
    add_openml_cli_args(p)
    p.add_argument("--results-root", type=Path, default=REPO / "results")
    p.add_argument("--out", type=Path, default=REPO / "results" / "report.html")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    datasets: list[str] = list(args.dataset) + resolve_openml_args(args)
    if not datasets:
        p.error("provide at least one of: --dataset / --openml-id / --openml-preset / --openml-all")

    results_root = Path(args.results_root)
    viz_dir = results_root / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    for ds in datasets:
        try:
            plot_column_heatmaps(ds, results_root / "column", viz_dir)
        except Exception as e:  # noqa: BLE001
            logging.warning("heatmaps failed for %s: %s", ds, e)
        jsonl = results_root / "row" / f"{ds}.jsonl"
        if jsonl.exists():
            try:
                plot_row_curves(ds, jsonl, viz_dir)
            except Exception as e:  # noqa: BLE001
                logging.warning("row curve failed for %s: %s", ds, e)

    build_report(datasets, results_root, args.out)
    logging.warning("Report written to %s", args.out)


if __name__ == "__main__":
    main()
