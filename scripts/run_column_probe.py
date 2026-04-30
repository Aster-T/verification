"""CLI entry point for column probing. Delegates to src/probing/column_probe.py."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Silence tabpfn-common-utils' PostHog telemetry (see run_row_probe.py).
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third-party" / "tabpfn" / "src"))

from src.configs import (  # noqa: E402
    CONFIG, VALID_TABPFN_WEIGHTS, add_openml_cli_args, resolve_openml_args,
    tabpfn_weights_tag,
)
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
    p.add_argument("--out", type=Path, default=REPO / "results",
                   help="Results root. Column artefacts go to <out>/<dataset>/column/.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--tabpfn-weights",
        choices=list(VALID_TABPFN_WEIGHTS),
        default=str(CONFIG["tabpfn"].get("weights", "v2_6")),
        help="Which TabPFN checkpoint to load. 'v2_6' (default, legacy): "
             "TabPFNRegressor's built-in 'auto' (= latest, v2.6). "
             "'v2': pin the original v2 weights "
             "(`tabpfn-v2-regressor.ckpt`); first use auto-downloads. "
             "Non-default values get their own subtree "
             "(<out>/weights_<tag>/<dataset>/column/) so v2 and v2.6 "
             "results coexist as independent ablation runs.",
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

    # Mirror the row-probe path partition: weights_<tag>/ subtree only when
    # not the legacy "v2_6" default. Column probe has no σ / test_size /
    # jitter_scale partitions, so the tag layer goes directly under <out>.
    out_root = args.out
    weights_subdir = tabpfn_weights_tag(args.tabpfn_weights)
    if weights_subdir is not None:
        out_root = out_root / weights_subdir
        logging.warning("tabpfn_weights=%s -> column results under %s",
                        args.tabpfn_weights, out_root)

    for ds in datasets:
        run_column_probe(
            ds, out_root / ds / "column", args.seed,
            tabpfn_weights=args.tabpfn_weights,
        )


if __name__ == "__main__":
    main()
