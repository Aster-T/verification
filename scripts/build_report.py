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


_DEFAULT_RESULTS_ROOT = REPO / "results"
_DEFAULT_OUT_HTML = REPO / "results" / "report.html"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", default=[])
    add_openml_cli_args(p)
    p.add_argument("--results-root", type=Path, default=_DEFAULT_RESULTS_ROOT)
    p.add_argument("--out", type=Path, default=_DEFAULT_OUT_HTML)
    p.add_argument(
        "--jitter-sigma", type=float, default=None,
        help="If set, look under <results-root>/sigma_<σ>/ (must match the "
             "value passed to run_row_probe.py). When --out is left at its "
             "default, the report is placed inside that same sigma subtree.",
    )
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
    out_html = Path(args.out)
    if args.jitter_sigma is not None:
        results_root = results_root / f"sigma_{args.jitter_sigma}"
        # If --out was left at the default, rehome it under the sigma subtree
        # so the HTML ends up next to the artefacts it references.
        if Path(args.out) == _DEFAULT_OUT_HTML:
            out_html = results_root / "report.html"
        logging.warning("jitter_sigma=%s -> reading %s, writing %s",
                        args.jitter_sigma, results_root, out_html)

    for ds in datasets:
        ds_root = results_root / ds
        viz_dir = ds_root / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)

        column_dir = ds_root / "column"
        if (column_dir / "mlr.npz").exists():
            try:
                plot_column_heatmaps(ds, column_dir, viz_dir)
            except Exception as e:  # noqa: BLE001
                logging.warning("heatmaps failed for %s: %s", ds, e)
        else:
            logging.warning("no column/ artefacts for %s; skipping heatmaps", ds)

        jsonl = ds_root / "row" / "metrics.jsonl"
        if jsonl.exists():
            try:
                plot_row_curves(ds, jsonl, viz_dir)
            except Exception as e:  # noqa: BLE001
                logging.warning("row curve failed for %s: %s", ds, e)

    build_report(datasets, results_root, out_html)
    logging.warning("Report written to %s", out_html)


if __name__ == "__main__":
    main()
