"""CLI entry: render heatmaps + row curves + a single multi-test_size HTML
report for given datasets.

Layout assumed (created by run_row_probe.py):

  results/sigma_<σ>/
    <dataset>/                       <- LOO + column probe artefacts (test_size-independent)
      row/metrics.jsonl
      column/mlr.npz, tabpfn.npz
      viz/...
    test_size_<ts>/<dataset>/        <- proportional split, per test_size
      row/metrics.jsonl
      viz/...
    report.html                      <- ONE report per sigma; dropdown selects test_size

The HTML always shows LOO + column blocks. PROP blocks (row_curve, facet PNGs,
metrics table) come in 9 versions inside each PROP dataset's section, one per
test_size, gated by a top-of-page dropdown.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from src.configs import add_openml_cli_args, resolve_openml_args, sigma_tag  # noqa: E402
from src.viz.curves import plot_row_curves  # noqa: E402
from src.viz.heatmap import plot_column_heatmaps  # noqa: E402
from src.viz.report import build_report  # noqa: E402


_DEFAULT_RESULTS_ROOT = REPO / "results"
_DEFAULT_OUT_HTML = REPO / "results" / "report.html"


def _discover_test_size_dirs(sigma_root: Path) -> list[Path]:
    """Find every `test_size_<ts>/` subdirectory of sigma_root, sorted by
    numeric test_size value."""
    if not sigma_root.exists():
        return []
    out: list[tuple[float, Path]] = []
    for p in sigma_root.iterdir():
        if not p.is_dir() or not p.name.startswith("test_size_"):
            continue
        try:
            ts = float(p.name[len("test_size_"):])
        except ValueError:
            continue
        out.append((ts, p))
    return [p for _, p in sorted(out, key=lambda t: t[0])]


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

    sigma_root = Path(args.results_root)
    out_html = Path(args.out)
    if args.jitter_sigma is not None:
        sigma_root = sigma_root / f"sigma_{sigma_tag(args.jitter_sigma)}"
        if Path(args.out) == _DEFAULT_OUT_HTML:
            out_html = sigma_root / "report.html"
        logging.warning("jitter_sigma=%s -> reading %s, writing %s",
                        args.jitter_sigma, sigma_root, out_html)

    test_size_dirs = _discover_test_size_dirs(sigma_root)
    if test_size_dirs:
        logging.info(
            "discovered %d test_size subtrees: %s",
            len(test_size_dirs),
            [d.name for d in test_size_dirs],
        )

    for ds in datasets:
        # LOO row probe + column probe artefacts at the sigma-only level.
        sigma_ds = sigma_root / ds
        loo_jsonl = sigma_ds / "row" / "metrics.jsonl"
        sigma_viz = sigma_ds / "viz"

        if loo_jsonl.exists():
            sigma_viz.mkdir(parents=True, exist_ok=True)
            try:
                plot_row_curves(ds, loo_jsonl, sigma_viz)
            except Exception as e:  # noqa: BLE001
                logging.warning("row curve (LOO) failed for %s: %s", ds, e)

        column_dir = sigma_ds / "column"
        if (column_dir / "mlr.npz").exists():
            sigma_viz.mkdir(parents=True, exist_ok=True)
            try:
                plot_column_heatmaps(ds, column_dir, sigma_viz)
            except Exception as e:  # noqa: BLE001
                logging.warning("heatmaps failed for %s: %s", ds, e)
        elif not loo_jsonl.exists():
            # Only warn when the dataset has no LOO and no column (i.e.,
            # we'd expect PROP-only).
            logging.info("no LOO/column artefacts for %s under %s", ds, sigma_ds)

        # Per-test_size proportional row probe.
        for ts_dir in test_size_dirs:
            prop_jsonl = ts_dir / ds / "row" / "metrics.jsonl"
            prop_viz = ts_dir / ds / "viz"
            if prop_jsonl.exists():
                prop_viz.mkdir(parents=True, exist_ok=True)
                try:
                    plot_row_curves(ds, prop_jsonl, prop_viz)
                except Exception as e:  # noqa: BLE001
                    logging.warning(
                        "row curve (PROP %s) failed for %s: %s",
                        ts_dir.name, ds, e,
                    )

    build_report(datasets, sigma_root, test_size_dirs, out_html)
    logging.warning("Report written to %s", out_html)


if __name__ == "__main__":
    main()
