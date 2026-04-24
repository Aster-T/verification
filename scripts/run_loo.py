"""CLI entry point for leave-one-out probing.

Examples:
  # full LOO on diabetes with MLR
  python scripts/run_loo.py --dataset diabetes --model mlr

  # predict only row 7 on synth_linear with TabPFN
  python scripts/run_loo.py --dataset synth_linear --model tabpfn --idx 7

  # full LOO on both models, written to results/loo/<dataset>/
  python scripts/run_loo.py --dataset diabetes --model mlr --model tabpfn
"""

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
from src.probing.loo import VALID_MODELS, run_loo  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", default=[],
                   help="Registered dataset name, repeatable.")
    add_openml_cli_args(p)
    p.add_argument("--model", action="append", default=[],
                   choices=list(VALID_MODELS),
                   help="Model to run, repeatable. Default: mlr.")
    p.add_argument("--idx", type=int, default=None,
                   help="If given, predict only this single index (else full LOO).")
    p.add_argument("--out", type=Path, default=REPO / "results" / "loo")
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

    models = list(args.model) or ["mlr"]

    for ds in datasets:
        for model in models:
            res = run_loo(ds, args.out, model=model, seed=args.seed, idx=args.idx)
            logging.warning(
                "[%s / %s] n=%d mse=%s rmse=%s mae=%s r2=%s -> %s",
                ds, model, res["n"], res["mse"], res["rmse"],
                res["mae"], res["r2"], res["csv"],
            )


if __name__ == "__main__":
    main()
