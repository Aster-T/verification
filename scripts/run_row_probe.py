"""CLI entry point for row probing. Delegates to src/probing/row_probe.py."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Silence tabpfn-common-utils' PostHog telemetry. Each TabPFN fit/predict
# emits one event; LOO mode fires ~150k calls per dataset and the async
# queue overflows ("analytics-python queue is full"). setdefault leaves any
# externally-set value alone. Set BEFORE importing tabpfn so the service
# reads the env var at init time.
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third-party" / "tabpfn" / "src"))

from src.configs import (  # noqa: E402
    CONFIG, VALID_JITTER_SCALES, VALID_TABPFN_WEIGHTS, add_openml_cli_args,
    jitter_scale_tag, resolve_openml_args, sigma_tag, tabpfn_weights_tag,
    test_size_tag,
)
from src.probing.row_probe import VALID_SPLIT_MODES, run_row_probe  # noqa: E402


def _csv_ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def _csv_strs(s: str) -> list[str]:
    return [x for x in s.split(",") if x]


def main() -> None:
    # Pull defaults from CONFIG so editing src/configs.py actually flows
    # through to the CLI. Copy into plain lists so argparse doesn't share
    # mutable state with CONFIG.
    rp = CONFIG["row_probe"]
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", default=[])
    add_openml_cli_args(p)
    p.add_argument(
        "--k-list",
        type=_csv_ints,
        default=list(rp["k_list"]),
        help="Context duplication factors, comma-separated. "
        "Default from CONFIG['row_probe']['k_list'] "
        f"(currently {list(rp['k_list'])}). "
        "Pass e.g. '1,2,3,5,10' to include the no-duplication baseline.",
    )
    p.add_argument(
        "--modes",
        type=_csv_strs,
        default=list(rp["modes"]),
        help="Modes to run. Default from CONFIG['row_probe']['modes'] "
        f"(currently {list(rp['modes'])}).",
    )
    p.add_argument(
        "--seeds",
        type=_csv_ints,
        default=list(rp["seeds"]),
        help="Seeds to iterate. Default from CONFIG['row_probe']['seeds'] "
        f"(currently {list(rp['seeds'])}).",
    )
    p.add_argument(
        "--split-mode",
        choices=list(VALID_SPLIT_MODES),
        default="proportional",
        help="'proportional' (default): train_test_split per seed, predict on "
        "fixed test set. 'loo': leave-one-out over the full dataset, "
        "aggregated metrics per (k, mode, seed).",
    )
    p.add_argument(
        "--test-size",
        type=float,
        default=0.5,
        help="Fraction of rows held out as the test set in proportional mode "
        "(must be in (0, 1)). Default: per-dataset value from CONFIG "
        "(usually 0.2). Ignored when --split-mode=loo.",
    )
    p.add_argument(
        "--jitter-sigma",
        type=float,
        default=None,
        help="Gaussian std added to each X cell in mode=jitter "
        "(must be >= 0). Default: CONFIG['row_probe']['jitter_sigma'] "
        "(currently 1e-6). 0 makes jitter numerically identical to exact; "
        "used only when 'jitter' is in --modes.",
    )
    p.add_argument(
        "--jitter-scale",
        choices=list(VALID_JITTER_SCALES),
        default=str(rp.get("jitter_scale", "absolute")),
        help="How σ relates to per-column scale in mode=jitter. "
        "'absolute' (default, legacy): noise = N(0, σ²) for every numeric "
        "cell. 'per_col_std': noise per column scaled by that column's std "
        "on the untiled X — σ becomes a relative perturbation strength. "
        "Non-default values get their own subtree "
        "(<out>/sigma_<σ>/jitter_<scale>/...) so absolute and per_col_std "
        "results coexist as independent ablation runs.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=REPO / "results",
        help="Results root. Row artefacts go to <out>/<dataset>/row/.",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Clear <out>/<dataset>/row/ before running.",
    )
    p.add_argument(
        "--no-tabpfn", action="store_true", help="Skip TabPFN branch (MLR only)."
    )
    p.add_argument(
        "--tabpfn-numeric", action="store_true",
        help="Pass numeric (per-column factorized) X to TabPFN instead of "
             "raw strings. Default: TabPFN receives the original text "
             "columns and uses its internal categorical handler. MLR is "
             "unaffected (it always factorizes internally).",
    )
    p.add_argument(
        "--tabpfn-weights",
        choices=list(VALID_TABPFN_WEIGHTS),
        default=str(CONFIG["tabpfn"].get("weights", "v2_6")),
        help="Which TabPFN checkpoint to load. 'v2_6' (default, legacy): "
             "let TabPFNRegressor resolve to its built-in 'auto' (latest "
             "bundled = v2.6). 'v2': pin the original v2 weights "
             "(`tabpfn-v2-regressor.ckpt`); auto-downloads on first use. "
             "Non-default values get their own subtree "
             "(<out>/sigma_<σ>/weights_<tag>/...) so v2 and v2.6 results "
             "coexist as independent ablation runs.",
    )
    p.add_argument(
        "--parallel-k",
        type=int,
        default=None,
        help="Number of TabPFN fit/predict slots to run concurrently on "
             "the GPU. Default: CONFIG['row_probe']['parallel_k'] "
             f"(currently {int(rp.get('parallel_k', 1))}). 1 disables "
             "threading and recovers the legacy serial path. Drop this "
             "if you start hitting CUDA OOM (each slot holds one X_ctx "
             "+ ensemble activations on-device).",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    datasets: list[str] = list(args.dataset) + resolve_openml_args(args)
    if not datasets:
        p.error(
            "provide at least one of: --dataset / --openml-id / --openml-preset / --openml-all"
        )

    # When --jitter-sigma is given, partition the results tree by sigma so
    # sweeping multiple sigmas doesn't overwrite each other:
    #   <out>/sigma_<σ>/<dataset>/row/...
    # When --tabpfn-weights != "v2_6", insert a weights_<tag>/ layer so v2
    # and v2.6 ablations live in disjoint subtrees. Placed *above* the
    # jitter_<scale>/ layer so a v2 run can still A/B against the absolute
    # vs per_col_std jitter ablations independently:
    #   <out>/sigma_<σ>/weights_<tag>/<dataset>/row/...
    # When --jitter-scale != "absolute", insert a jitter_<scale>/ layer so
    # absolute and per_col_std runs (the two ablation siblings) live in
    # disjoint subtrees:
    #   <out>/sigma_<σ>/[weights_<tag>/]jitter_<scale>/<dataset>/row/...
    # All non-default tags drop out for the legacy default — pre-existing
    # trees stay bit-for-bit compatible.
    # In proportional mode we additionally partition by test_size so the
    # 9 test_size sweep produces 9 distinct result subtrees:
    #   <out>/sigma_<σ>/[weights_<tag>/][jitter_<scale>/]test_size_<ts>/<dataset>/row/...
    # LOO output paths are NOT partitioned by test_size (LOO ignores it).
    out_root = args.out
    if args.jitter_sigma is not None:
        out_root = out_root / f"sigma_{sigma_tag(args.jitter_sigma)}"
        logging.warning("jitter_sigma=%s -> results under %s",
                        args.jitter_sigma, out_root)
    weights_subdir = tabpfn_weights_tag(args.tabpfn_weights)
    if weights_subdir is not None:
        out_root = out_root / weights_subdir
        logging.warning("tabpfn_weights=%s -> results under %s",
                        args.tabpfn_weights, out_root)
    scale_subdir = jitter_scale_tag(args.jitter_scale)
    if scale_subdir is not None:
        out_root = out_root / scale_subdir
        logging.warning("jitter_scale=%s -> results under %s",
                        args.jitter_scale, out_root)
    if args.split_mode == "proportional":
        out_root = out_root / f"test_size_{test_size_tag(args.test_size)}"
        logging.warning("test_size=%s -> proportional results under %s",
                        args.test_size, out_root)

    for ds in datasets:
        row_dir = out_root / ds / "row"
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
            test_size=args.test_size,
            jitter_sigma=args.jitter_sigma,
            jitter_scale=args.jitter_scale,
            tabpfn_numeric=args.tabpfn_numeric,
            tabpfn_weights=args.tabpfn_weights,
            parallel_k=args.parallel_k,
        )


if __name__ == "__main__":
    main()
