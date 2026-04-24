"""
Leave-One-Out probe for small regression datasets.

For each index i in 0..n-1 (or only the specified `idx`):
  train on every other row, predict row i, record (y_true, y_pred, residual).

OUTPUT LAYOUT:
  <out_root>/<dataset>/predictions_<model>.csv     y_true, y_pred, residual
  <out_root>/<dataset>/metrics_<model>.json        mse, rmse, mae, r2, n,
                                                   y_mean, y_std, dataset,
                                                   model, seed, feature_names

For single-point runs (idx is not None) filenames become
predictions_<model>_idx{i}.csv / metrics_<model>_idx{i}.json so successive
single-point calls don't overwrite each other.

R² is NaN (written as null in JSON) when fewer than 2 predictions exist.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.configs import get_device
from src.data.loaders import load_dataset_full
from src.models.mlr_wrapper import MLRWithW
from src.utils.io import dump_json, ensure_dir
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

VALID_MODELS = ("mlr", "tabpfn")


def _predict_one(
    model: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    is_nominal: list[bool],
    seed: int,
) -> float:
    if model == "mlr":
        m = MLRWithW(standardize=True).fit(X_tr, y_tr, is_nominal=is_nominal)
        return float(m.predict(X_te)[0])
    if model == "tabpfn":
        from src.models.tabpfn_wrapper import TabPFNWithColAttn  # noqa: PLC0415

        m = TabPFNWithColAttn(device=get_device(), seed=seed).fit(X_tr, y_tr)
        return float(m.predict(X_te)[0])
    raise ValueError(f"Unknown model {model!r}; expected one of {VALID_MODELS}")


def _write_predictions_csv(
    path: Path,
    indices: list[int],
    y_true: list[float],
    y_pred: list[float],
) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "y_true", "y_pred", "residual"])
        for i, yt, yp in zip(indices, y_true, y_pred):
            w.writerow([i, yt, yp, yt - yp])


def _metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    n = int(len(y_true))
    mse = float(mean_squared_error(y_true, y_pred)) if n >= 1 else float("nan")
    rmse = float(np.sqrt(mse)) if n >= 1 else float("nan")
    mae = float(mean_absolute_error(y_true, y_pred)) if n >= 1 else float("nan")
    # r2_score requires >=2 samples AND non-constant y_true; guard both.
    r2: float | None
    if n >= 2 and float(np.var(y_true)) > 0.0:
        r2 = float(r2_score(y_true, y_pred))
    else:
        r2 = None
    return {"n": n, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def run_loo(
    dataset: str,
    out_root: Path,
    model: str = "mlr",
    seed: int = 0,
    idx: int | None = None,
) -> dict:
    """
    Run leave-one-out on `dataset` using `model`.

    ARGS:
      dataset:  key in CONFIG["datasets"].
      out_root: parent dir, e.g. `results/loo`. A subdir `<dataset>/` is created.
      model:    "mlr" or "tabpfn".
      seed:     forwarded to data loader and model.
      idx:      None -> full LOO over all n rows.
                int  -> predict only this single index.

    RETURNS:
      dict with metrics + paths of the two files written.
    """
    if model not in VALID_MODELS:
        raise ValueError(f"model must be one of {VALID_MODELS}, got {model!r}")

    set_seed(seed)
    X, y, feature_names, is_nominal = load_dataset_full(dataset, seed)
    n = X.shape[0]

    if idx is None:
        indices = list(range(n))
    else:
        if not 0 <= idx < n:
            raise IndexError(f"idx={idx} out of range for dataset of size {n}")
        indices = [int(idx)]

    out_dir = ensure_dir(Path(out_root) / dataset)
    suffix = f"_idx{idx}" if idx is not None else ""
    csv_path = out_dir / f"predictions_{model}{suffix}.csv"
    json_path = out_dir / f"metrics_{model}{suffix}.json"

    y_true_list: list[float] = []
    y_pred_list: list[float] = []

    all_idx = np.arange(n)
    for k, i in enumerate(indices):
        mask = all_idx != i
        X_tr, y_tr = X[mask], y[mask]
        X_te = X[i : i + 1]
        y_hat = _predict_one(model, X_tr, y_tr, X_te, is_nominal, seed)
        y_true_list.append(float(y[i]))
        y_pred_list.append(float(y_hat))
        if (k + 1) % 50 == 0 or (k + 1) == len(indices):
            logger.info("LOO %s/%s: %d/%d done", dataset, model, k + 1, len(indices))

    _write_predictions_csv(csv_path, indices, y_true_list, y_pred_list)

    y_true_arr = np.asarray(y_true_list, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred_list, dtype=np.float64)
    metrics = _metrics(y_true_arr, y_pred_arr)
    payload = {
        "dataset": dataset,
        "model": model,
        "seed": int(seed),
        "idx": int(idx) if idx is not None else None,
        "n_total": int(n),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
        "y_mean": float(np.mean(y_true_arr)) if len(y_true_arr) else None,
        "y_std": float(np.std(y_true_arr)) if len(y_true_arr) else None,
        **metrics,
    }
    dump_json(json_path, payload)
    logger.info("wrote %s and %s", csv_path, json_path)

    return {"csv": csv_path, "json": json_path, **metrics}
