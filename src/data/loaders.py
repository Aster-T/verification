"""
Dataset loaders for the probing experiments.

PUBLIC API:
  load_dataset(name, seed) -> (X_tr, y_tr, X_te, y_te, feature_names, is_nominal)

All arrays are numpy; `feature_names` and `is_nominal` are returned separately
rather than via a pandas DataFrame because downstream code (MLR / TabPFN
wrappers) works on plain numpy arrays. Per 00_conventions.md:
  - shapes are (n, f) for X and (n,) for y
  - NaN is preserved (not imputed here)
  - split is random but deterministic per seed
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.datasets import (
    fetch_california_housing,
    fetch_openml,
    load_diabetes,
    make_regression,
)
from sklearn.model_selection import train_test_split

from src.configs import get_dataset_cfg

logger = logging.getLogger(__name__)


def load_dataset_full(
    name: str, seed: int
) -> tuple[np.ndarray, np.ndarray, list[str], list[bool]]:
    """
    Load a dataset WITHOUT performing a train/test split.

    Used by LOO and any caller that wants to decide the splitting scheme
    itself (e.g. leave-one-out, k-fold CV). `seed` still controls synthetic
    data generation and subsampling so that `load_dataset_full(name, seed)`
    and `load_dataset(name, seed)` operate on the same underlying rows.

    RETURNS:
      (X, y, feature_names, is_nominal)
        X:             (n, f) float64
        y:             (n,)   float64
        feature_names: list[str] of length f
        is_nominal:    list[bool] of length f
    """
    cfg = get_dataset_cfg(name)
    loader = cfg["loader"]

    if loader == "sklearn_builtin":
        sk_name = cfg["sklearn_name"]
        if sk_name == "diabetes":
            data = load_diabetes()
            X = np.asarray(data.data, dtype=np.float64)
            y = np.asarray(data.target, dtype=np.float64)
            feature_names = list(data.feature_names)
        elif sk_name == "california_housing":
            data = fetch_california_housing()
            X = np.asarray(data.data, dtype=np.float64)
            y = np.asarray(data.target, dtype=np.float64)
            feature_names = list(data.feature_names)
        else:
            raise ValueError(f"Unsupported sklearn_name: {sk_name!r}")

        sub = cfg.get("subsample")
        if sub is not None and sub < X.shape[0]:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=int(sub), replace=False)
            X = X[idx]
            y = y[idx]
        is_nominal = [False] * X.shape[1]

    elif loader == "make_regression":
        X, y = make_regression(
            n_samples=cfg["n_samples"],
            n_features=cfg["n_features"],
            n_informative=cfg["n_informative"],
            noise=cfg["noise"],
            random_state=seed,
        )
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        feature_names = [f"x{i}" for i in range(X.shape[1])]
        is_nominal = [False] * X.shape[1]

    elif loader == "openml_id":
        X, y, feature_names, is_nominal = _load_openml_by_id(cfg, seed)

    else:
        raise ValueError(f"Unknown loader: {loader!r}")

    return X, y, feature_names, is_nominal


def load_dataset(
    name: str, seed: int
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[bool],
]:
    """
    Load and split one of the configured datasets.

    ARGS:
      name:  key in CONFIG["datasets"]. One of {"diabetes", "cali_housing",
             "synth_linear"}.
      seed:  used for train/test split (and data generation for synth_linear).

    RETURNS:
      (X_tr, y_tr, X_te, y_te, feature_names, is_nominal)
        X shapes: (n_tr, f) / (n_te, f); y: (n_tr,) / (n_te,)
        feature_names: list[str] of length f
        is_nominal:    list[bool] of length f  (all False for these 3 datasets)
    """
    cfg = get_dataset_cfg(name)
    test_size = cfg.get("test_size", 0.2)
    X, y, feature_names, is_nominal = load_dataset_full(name, seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return X_tr, y_tr, X_te, y_te, feature_names, is_nominal


def _load_openml_by_id(cfg: dict, seed: int):
    """
    Fetch one OpenML dataset by data_id, convert to a regression-ready
    (X, y, feature_names, is_nominal) tuple.

    CATEGORICAL / STRING / BOOL COLUMNS:
      Encoded via pd.factorize (NaN → -1 → NaN again so TabPFN can see it).
      Marked True in `is_nominal` so MLR skips z-score on them.

    TARGET VALIDATION:
      If the target column is non-numeric (classification), raises ValueError.
      The wrappers in this repo are regression-only; pick a regression task
      or wrap TabPFNClassifier separately.
    """
    openml_id = int(cfg["openml_id"])
    bunch = fetch_openml(data_id=openml_id, as_frame=True, parser="auto")

    target_names = list(bunch.target_names or [])
    if not target_names:
        raise ValueError(
            f"OpenML id {openml_id} has no declared target column."
        )
    target_col = target_names[0]
    df = bunch.frame
    y_series = df[target_col]
    if not pd.api.types.is_numeric_dtype(y_series):
        raise ValueError(
            f"OpenML id {openml_id} target {target_col!r} is non-numeric "
            f"(dtype={y_series.dtype}); this repo's wrappers are regression-only."
        )
    y = y_series.to_numpy(dtype=np.float64)

    X_df = df.drop(columns=target_names)
    is_nominal: list[bool] = []
    for col in list(X_df.columns):
        dtype = X_df[col].dtype
        is_categorical = (
            str(dtype) == "category"
            or dtype == object  # noqa: E721  (pandas idiom)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
        )
        if is_categorical:
            codes, _ = pd.factorize(X_df[col], sort=True)
            col_float = codes.astype(np.float64)
            col_float[codes < 0] = np.nan  # preserve NaN
            X_df[col] = col_float
            is_nominal.append(True)
        else:
            is_nominal.append(False)
    X = X_df.to_numpy(dtype=np.float64)
    feature_names = [str(c) for c in X_df.columns]

    sub = cfg.get("subsample")
    if sub is not None and sub < X.shape[0]:
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=int(sub), replace=False)
        X = X[idx]
        y = y[idx]
        logger.info(
            "openml id=%d subsampled to %d rows (original=%d)",
            openml_id, int(sub), len(y_series),
        )

    return X, y, feature_names, is_nominal
