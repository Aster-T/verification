"""Export registered datasets to `datasets/<name>/data.csv` + `meta.json`.

See datasets/README.md for the target layout. This script is a one-way
snapshot tool: it calls src.data.loaders.load_dataset_full(name, seed), then
writes the resulting (X, y) and metadata out as CSV + JSON.

Examples:
  python scripts/export_datasets.py --dataset diabetes --dataset synth_linear
  python scripts/export_datasets.py --dataset cali_housing

  # OpenML, auto-named 'openml_560':
  python scripts/export_datasets.py --openml-id 560 --openml-subsample 1500

  # OpenML with a custom name 'bodyfat' (folder becomes datasets/bodyfat/):
  python scripts/export_datasets.py --openml-id 560:bodyfat
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third-party" / "tabpfn" / "src"))

from src.configs import (  # noqa: E402
    CONFIG,
    add_openml_cli_args,
    get_dataset_cfg,
    resolve_openml_args,
)
from src.data.loaders import load_dataset_full  # noqa: E402
from src.utils.io import dump_json, ensure_dir  # noqa: E402


_SOURCE_BY_LOADER = {
    "sklearn_builtin": lambda cfg: f"sklearn.datasets.{_SKLEARN_FN[cfg['sklearn_name']]}",
    "make_regression": lambda cfg: "sklearn.datasets.make_regression",
    "openml_id":       lambda cfg: f"openml.data_id={cfg['openml_id']}",
}
_SKLEARN_FN = {
    "diabetes": "load_diabetes",
    "california_housing": "fetch_california_housing",
}


def _describe_source(cfg: dict) -> str:
    fn = _SOURCE_BY_LOADER.get(cfg["loader"])
    return fn(cfg) if fn else f"unknown:{cfg['loader']}"


def _target_col_name(cfg: dict) -> str:
    if cfg["loader"] == "make_regression":
        return "y"
    return "target"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def export_one(name: str, out_root: Path, seed: int) -> Path:
    cfg = get_dataset_cfg(name)
    X, y, feature_names, is_nominal = load_dataset_full(name, seed)

    target_col = _target_col_name(cfg)
    # Guarantee target column name doesn't clash with a feature name.
    if target_col in feature_names:
        target_col = f"{target_col}_"

    df = pd.DataFrame(X, columns=feature_names)
    df[target_col] = np.asarray(y, dtype=np.float64)

    ds_dir = ensure_dir(Path(out_root) / name)
    csv_path = ds_dir / "data.csv"
    df.to_csv(csv_path, index=False)

    generator_params = None
    if cfg["loader"] == "make_regression":
        generator_params = {
            k: cfg[k] for k in ("n_samples", "n_features", "n_informative", "noise")
            if k in cfg
        }
    elif cfg["loader"] == "openml_id":
        generator_params = {
            "openml_id": int(cfg["openml_id"]),
            "subsample": cfg.get("subsample"),
        }
    elif cfg["loader"] == "sklearn_builtin":
        sub = cfg.get("subsample")
        if sub is not None:
            generator_params = {"subsample": int(sub)}

    meta = {
        "name": name,
        "source": _describe_source(cfg),
        "generator_params": generator_params,
        "target_col": target_col,
        "feature_names": list(feature_names),
        "is_nominal": list(is_nominal),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "seed": int(seed),
        "exported_at": dt.date.today().isoformat(),
        "sha256": _sha256_file(csv_path),
    }
    dump_json(ds_dir / "meta.json", meta)
    return ds_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", default=[])
    add_openml_cli_args(p)
    p.add_argument("--out", type=Path, default=REPO / "datasets")
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

    for name in datasets:
        if name not in CONFIG["datasets"]:
            raise KeyError(
                f"Dataset {name!r} not registered in CONFIG['datasets']. "
                f"Available: {sorted(CONFIG['datasets'].keys())}"
            )
        ds_dir = export_one(name, args.out, args.seed)
        logging.warning("exported %s -> %s", name, ds_dir)


if __name__ == "__main__":
    main()
