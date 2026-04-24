"""Infer `meta.json` from a `data.csv`.

Use this when you drop a user-supplied CSV into `datasets/<name>/data.csv`
and want the sibling meta.json generated automatically.

Non-destructive by design: **data.csv is never modified**. Text columns
stay as text on disk; downstream loaders (src/data/loaders.py) factorize
them into integer codes at load time, consulting
`meta.json['categorical_mappings']` for a stable category→code mapping.

Conventions (see datasets/README.md):
  - last column is the regression target (override with --target-col)
  - target must be numeric (this repo is regression-only)
  - any column with string/object/bool/category dtype is marked nominal
  - integer columns that are actually categorical can be forced via
    --nominal <name1,name2>

Examples:
  python scripts/infer_meta.py datasets/my_ds/data.csv
  python scripts/infer_meta.py datasets/my_ds/data.csv --target-col price
  python scripts/infer_meta.py datasets/my_ds/data.csv --source "Kaggle ABC" \\
      --nominal day_of_week,region
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

from src.utils.io import dump_json  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_text_dtype(s: pd.Series) -> bool:
    dtype = s.dtype
    return (
        str(dtype) == "category"
        or dtype == object  # noqa: E721  (pandas idiom)
        or pd.api.types.is_bool_dtype(dtype)
        or pd.api.types.is_string_dtype(dtype)
    )


def _categorical_mapping(s: pd.Series) -> list[str]:
    """Compute the stable category list for a nominal column.

    Returns `categories` such that `categories[i]` is the string form of the
    value that should map to integer code i at load time (pd.factorize with
    sort=True ordering). NaN cells are excluded from the list and get code
    -1 (represented as NaN downstream).
    """
    _codes, uniques = pd.factorize(s, sort=True)
    return [str(u) for u in uniques.tolist()]


def infer_meta(
    csv_path: Path,
    target_col: str | None = None,
    source: str = "user_supplied",
    extra_nominal: set[str] | None = None,
    seed: int | None = None,
) -> dict:
    """Read `csv_path` (read-only) and return a meta dict describing it.

    Never modifies the CSV. For nominal columns the category→code mapping
    is recorded under `meta['categorical_mappings']` so that downstream
    loaders can apply a stable factorization at load time without requiring
    the CSV itself to be numeric.

    A column is treated as nominal if its pandas dtype is non-numeric
    (string/object/bool/category) OR if its name is in `extra_nominal`
    (useful for integer-coded categoricals like "day_of_week").
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError(
            f"{csv_path} needs at least 2 columns (1+ features + 1 target); "
            f"got shape {df.shape}"
        )

    cols: list[str] = list(df.columns)
    if target_col is None:
        target_col = cols[-1]
    elif target_col not in cols:
        raise KeyError(
            f"--target-col={target_col!r} not in CSV columns {cols}"
        )

    y_series = df[target_col]
    if not pd.api.types.is_numeric_dtype(y_series):
        raise ValueError(
            f"target column {target_col!r} has dtype={y_series.dtype}; "
            f"this repo is regression-only — numeric target required"
        )

    feature_cols = [c for c in cols if c != target_col]
    extra_nominal = extra_nominal or set()
    for bad in extra_nominal - set(feature_cols):
        raise KeyError(
            f"--nominal column {bad!r} not in feature columns {feature_cols}"
        )

    is_nominal: list[bool] = []
    categorical_mappings: dict[str, list[str]] = {}
    for c in feature_cols:
        text_like = _is_text_dtype(df[c])
        forced = c in extra_nominal
        nominal = text_like or forced
        is_nominal.append(bool(nominal))
        # Record the mapping for any nominal column so loaders can encode
        # consistently across runs. Text columns always record; forced
        # integer-nominal columns also record (same pd.factorize semantics).
        if nominal:
            cats = _categorical_mapping(df[c])
            categorical_mappings[c] = cats
            if text_like:
                logging.info(
                    "text column %r: recorded %d categories (CSV unchanged)",
                    c, len(cats),
                )

    meta = {
        "name": csv_path.parent.name,
        "source": source,
        "generator_params": None,
        "target_col": str(target_col),
        "feature_names": [str(c) for c in feature_cols],
        "is_nominal": is_nominal,
        "n_rows": int(df.shape[0]),
        "n_features": int(len(feature_cols)),
        "seed": int(seed) if seed is not None else None,
        "exported_at": dt.date.today().isoformat(),
        "sha256": _sha256_file(csv_path),
        # categories[i] is the string form of the value that maps to code i
        # at load time (see src/data/loaders.py::_load_local_csv).
        # NaN cells stay NaN (code -1 is re-mapped to NaN).
        "categorical_mappings": categorical_mappings,
    }
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path, help="path to data.csv")
    p.add_argument("--target-col", default=None,
                   help="target column name (default: last column)")
    p.add_argument("--source", default="user_supplied",
                   help="free-form provenance string (default: 'user_supplied')")
    p.add_argument("--nominal", default="",
                   help="comma-separated feature names to force-mark as nominal "
                        "(numeric-but-categorical columns)")
    p.add_argument("--seed", type=int, default=None,
                   help="seed used to derive this CSV, if any (default: null)")
    p.add_argument("--overwrite", action="store_true",
                   help="overwrite existing meta.json (default: refuse)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    csv_path: Path = args.csv.resolve()
    if not csv_path.exists():
        p.error(f"CSV not found: {csv_path}")
    meta_path = csv_path.parent / "meta.json"
    if meta_path.exists() and not args.overwrite:
        p.error(f"{meta_path} already exists; pass --overwrite to replace it")

    extra_nominal = {s for s in args.nominal.split(",") if s}

    meta = infer_meta(
        csv_path,
        target_col=args.target_col,
        source=args.source,
        extra_nominal=extra_nominal,
        seed=args.seed,
    )
    dump_json(meta_path, meta)
    logging.warning(
        "wrote %s (n_rows=%d n_features=%d target=%r is_nominal=%d)",
        meta_path, meta["n_rows"], meta["n_features"],
        meta["target_col"], sum(meta["is_nominal"]),
    )


if __name__ == "__main__":
    main()
