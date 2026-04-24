"""CLI entry point for row probing. Delegates to src/probing/row_probe.py."""

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
from src.probing.row_probe import VALID_SPLIT_MODES, run_row_probe  # noqa: E402


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _csv_strs(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", default=[])
    add_openml_cli_args(p)
    p.add_argument(
        "--k-list", type=_csv_ints, default=[2, 3, 5, 10],
        help="Context duplication factors, comma-separated. Default: 2,3,5,10 "
             "(the 2x-10x exploration range). Pass e.g. '1,2,3,5,10' to include "
             "the no-duplication baseline.",
    )
    p.add_argument("--modes", type=_csv_strs, default=["exact", "jitter"])
    p.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    p.add_argument(
        "--split-mode", choices=list(VALID_SPLIT_MODES), default="proportional",
        help="'proportional' (default): train_test_split per seed, predict on "
             "fixed test set. 'loo': leave-one-out over the full dataset, "
             "aggregated metrics per (k, mode, seed).",
    )
    p.add_argument("--out", type=Path, default=REPO / "results",
                   help="Results root. Row artefacts go to <out>/<dataset>/row/.")
    p.add_argument("--fresh", action="store_true",
                   help="Clear <out>/<dataset>/row/ before running.")
    p.add_argument("--no-tabpfn", action="store_true",
                   help="Skip TabPFN branch (MLR only).")
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
        row_dir = args.out / ds / "row"
        if args.fresh and row_dir.exists():
            for p_ in row_dir.glob("*"):
                if p_.is_file():
                    p_.unlink()
        run_row_probe(
            ds,
            row_dir,
            args.k_list,
            args.modes,
            args.seeds,
            include_tabpfn=not args.no_tabpfn,
            split_mode=args.split_mode,
        )


if __name__ == "__main__":
    main()
