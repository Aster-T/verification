"""CLI entry point for column probing. Delegates to src/probing/column_probe.py."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third-party" / "tabpfn" / "src"))

from src.configs import add_openml_cli_args, resolve_openml_args  # noqa: E402
from src.probing.column_probe import run_column_probe  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Registered dataset name, repeatable. e.g. --dataset diabetes --dataset cali_housing",
    )
    add_openml_cli_args(p)
    p.add_argument("--out", type=Path, default=REPO / "results" / "column")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    datasets: list[str] = list(args.dataset) + resolve_openml_args(args)
    if not datasets:
        p.error("provide at least one of: --dataset / --openml-id / --openml-preset / --openml-all")

    for ds in datasets:
        run_column_probe(ds, args.out, args.seed)


if __name__ == "__main__":
    main()
