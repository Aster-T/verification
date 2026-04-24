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
from src.probing.row_probe import run_row_probe  # noqa: E402


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _csv_strs(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", default=[])
    add_openml_cli_args(p)
    p.add_argument("--k-list", type=_csv_ints, default=[1, 2, 3, 5, 10])
    p.add_argument("--modes", type=_csv_strs, default=["exact", "jitter"])
    p.add_argument("--seeds", type=_csv_ints, default=[0, 1, 2])
    p.add_argument("--out", type=Path, default=REPO / "results" / "row")
    p.add_argument("--fresh", action="store_true")
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

    args.out.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        jsonl_path = args.out / f"{ds}.jsonl"
        if args.fresh and jsonl_path.exists():
            jsonl_path.unlink()
        run_row_probe(
            ds,
            jsonl_path,
            args.k_list,
            args.modes,
            args.seeds,
            include_tabpfn=not args.no_tabpfn,
        )


if __name__ == "__main__":
    main()
