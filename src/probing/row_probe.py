"""
Row probe: measure how nRMSE / R² / RMSE / MAE react as the context is
duplicated (2x–10x or user-specified).

SPLIT MODES:
  proportional  — one train/test split per seed via load_dataset(name, seed).
                  Duplicate (X_tr, y_tr) k× → context. Predict fixed X_te.
  loo           — no split. For each held-out index i: duplicate all other
                  n-1 rows k× → context, predict row i. Aggregate over all
                  N left-out points into one record. Intended for small
                  datasets where a proportional split is wasteful.

For MLR in "exact" mode the R² (hence nRMSE) is mathematically independent
of k — OLS on a uniformly replicated dataset returns the same coefficients.

PRIMARY METRIC: nRMSE = RMSE / std(y_query)
  - Dimensionless, scale-invariant -> comparable across datasets.
  - 0 = perfect; 1 ≈ predicting the query mean; >1 = worse than that baseline.
  - Equals sqrt(1 - R²) exactly on the same query set.
  - Returns null when std(y_query) == 0 (degenerate constant target).

OUTPUT LAYOUT (for a given dataset, under `row_dir`):
  metrics.jsonl
     One record per (model, split_mode, k, mode, seed). Schema:
       dataset, model, split_mode, k, mode, seed,
       n_ctx, n_query, n_folds, n_features, y_query_std,
       nrmse, r2, rmse, mae, fit_sec, predict_sec
     n_ctx / n_query describe a SINGLE model invocation:
       proportional:  n_ctx = k×n_tr,  n_query = n_te,  n_folds = 1
       loo:           n_ctx = k×(N-1), n_query = 1,     n_folds = N
     Skipped records (only written when fit/predict itself raises):
       {"skipped": true, "reason": "<ExceptionType>: <msg>", <base fields>}

  predictions_<model>_<split>_<mode>_k<k>_s<seed>.csv
     Per-combo detail. Columns: id, y_true, y_pred, residual.
     For proportional: id = 0..n_te-1 (index in the held-out test set).
     For loo:          id = 0..N-1   (index in the full dataset).
     Skipped combos do NOT produce a CSV.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.configs import CONFIG, get_dataset_cfg, get_device
from src.data.loaders import load_dataset_full
from src.models.mlr_wrapper import MLRWithW
from src.utils.io import ensure_dir, write_jsonl
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

VALID_SPLIT_MODES = ("proportional", "loo")


def duplicate_context(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    mode: str,
    rng: np.random.Generator,
    jitter_sigma: float | None = None,
    is_nominal: list[bool] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tile (X, y) k-fold. In "jitter" mode, add N(0, sigma) noise to the
    duplicated X cells — **except**:
      - the first tile (rows 0..n-1) is preserved as the original anchor and
        receives no noise; only tiles 1..k-1 are perturbed.
      - columns flagged nominal via `is_nominal` pass through untouched —
        integer category codes would be corrupted by any float noise.
    With anchor preservation, k=1 + jitter is numerically identical to exact
    (no copies to perturb). For k>=2 the context is one pristine copy of X
    plus (k-1) jittered copies.

    ARGS:
      jitter_sigma: if None, read CONFIG["row_probe"]["jitter_sigma"]
        (usually 1e-6). Explicit values override the config; 0.0 makes
        "jitter" numerically identical to "exact".
      is_nominal: optional list[bool] of length X.shape[1]. When provided,
        noise is zeroed on columns where is_nominal[j] is True. When None,
        all columns receive noise (legacy behaviour).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if mode not in ("exact", "jitter"):
        raise ValueError(f"mode must be 'exact' or 'jitter', got {mode!r}")
    n = X.shape[0]
    X_rep = np.tile(X, (k, 1))
    y_rep = np.tile(y, k)
    if mode == "jitter":
        sigma = (CONFIG["row_probe"]["jitter_sigma"]
                 if jitter_sigma is None else jitter_sigma)
        if sigma < 0:
            raise ValueError(f"jitter_sigma must be >= 0, got {sigma}")
        if is_nominal is not None:
            nominal_mask = np.asarray(is_nominal, dtype=bool)
            if nominal_mask.shape[0] != X_rep.shape[1]:
                raise ValueError(
                    f"is_nominal length {nominal_mask.shape[0]} != "
                    f"X.shape[1] {X_rep.shape[1]}"
                )
        else:
            nominal_mask = None

        if sigma > 0 and k >= 2:
            if X_rep.dtype == object:
                # Per-column path: leave nominal/text columns exactly as-is,
                # add N(0, sigma) noise to numeric columns only.
                for j in range(X_rep.shape[1]):
                    if nominal_mask is not None and nominal_mask[j]:
                        continue
                    col = X_rep[:, j]
                    # Skip columns that aren't numeric (safety: an undeclared
                    # text column would break the float cast).
                    if any(isinstance(v, str) for v in col):
                        continue
                    float_col = np.array(col, dtype=np.float64)
                    noise_col = rng.standard_normal(float_col.shape[0]) * sigma
                    # Anchor: rows 0..n-1 are tile 0 — keep pristine.
                    noise_col[:n] = 0.0
                    float_col += noise_col
                    X_rep[:, j] = float_col
            else:
                noise = rng.standard_normal(X_rep.shape) * sigma
                # Anchor: rows 0..n-1 are tile 0 — keep pristine.
                noise[:n, :] = 0.0
                if nominal_mask is not None:
                    # Suppress noise on nominal columns so integer codes stay
                    # integer-valued (preserves the category semantics).
                    noise[:, nominal_mask] = 0.0
                X_rep = X_rep + noise
    y_out = y_rep.astype(np.float64, copy=False)
    # Keep object dtype when present (caller decides how to use it); cast
    # numeric case to float64 for downstream math.
    if X_rep.dtype == object:
        return X_rep, y_out
    return X_rep.astype(np.float64, copy=False), y_out


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Regression metrics, with nRMSE = RMSE/std(y_true). nrmse is None when
    std=0; r² is NaN when n<2 or std=0."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    y_std = float(np.std(y_true, ddof=0))
    nrmse: float | None = float(rmse / y_std) if y_std > 0 else None
    r2 = float(r2_score(y_true, y_pred)) if (y_true.size >= 2 and y_std > 0) else float("nan")
    return {
        "nrmse": nrmse,
        "r2": r2,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "y_query_std": y_std,
    }


def _fit_predict_mlr(X_ctx, y_ctx, X_te, is_nominal):
    t0 = time.perf_counter()
    m = MLRWithW(standardize=True).fit(X_ctx, y_ctx, is_nominal=is_nominal)
    fit_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = m.predict(X_te)
    predict_sec = time.perf_counter() - t0
    return y_pred, fit_sec, predict_sec


def _fit_predict_tabpfn(X_ctx, y_ctx, X_te, seed, *, accept_text: bool = True):
    from src.models.tabpfn_wrapper import TabPFNWithColAttn  # noqa: PLC0415

    t0 = time.perf_counter()
    # preprocess_y=True restores TabPFN's default (None, "safepower") y-preprocess
    # ensemble, which row probing needs for heavy-tailed targets. Column probing
    # keeps it disabled for attention alignment.
    m = TabPFNWithColAttn(
        device=get_device(), seed=seed,
        accept_text=accept_text, preprocess_y=True,
    ).fit(X_ctx, y_ctx)
    fit_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = m.predict(X_te)
    predict_sec = time.perf_counter() - t0
    return y_pred, fit_sec, predict_sec


def _combo_stem(model: str, split_mode: str, mode: str, k: int, seed: int) -> str:
    return f"predictions_{model}_{split_mode}_{mode}_k{k}_s{seed}"


def _write_predictions_csv(
    path: Path,
    ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "y_true", "y_pred", "residual"])
        for i, yt, yp in zip(ids, y_true, y_pred):
            w.writerow([int(i), float(yt), float(yp), float(yt) - float(yp)])


def _run_proportional(
    dataset: str,
    row_dir: Path,
    jsonl_path: Path,
    k_list: list[int],
    modes: list[str],
    seeds: list[int],
    models: list[str],
    test_size: float | None = None,
    jitter_sigma: float | None = None,
    tabpfn_numeric: bool = False,
) -> None:
    effective_test_size = (
        test_size if test_size is not None
        else float(get_dataset_cfg(dataset).get("test_size", 0.2))
    )
    for seed in seeds:
        set_seed(seed)
        X, y, _names, is_nominal = load_dataset_full(dataset, seed)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=effective_test_size, random_state=seed,
        )
        n_tr_original = X_tr.shape[0]
        n_query = X_te.shape[0]
        n_features = X_tr.shape[1]

        for k in k_list:
            n_ctx = k * n_tr_original
            for mode in modes:
                rng = np.random.default_rng(np.random.SeedSequence(
                    entropy=[seed, k, 0 if mode == "exact" else 1],
                ))
                X_ctx, y_ctx = duplicate_context(
                    X_tr, y_tr, k, mode, rng,
                    jitter_sigma=jitter_sigma,
                    is_nominal=is_nominal,
                )

                for model in models:
                    base_rec = {
                        "dataset": dataset,
                        "model": model,
                        "split_mode": "proportional",
                        "k": int(k),
                        "mode": mode,
                        "seed": int(seed),
                        "n_ctx": int(n_ctx),
                        "n_query": int(n_query),
                        "n_folds": 1,
                        "n_features": int(n_features),
                    }

                    try:
                        if model == "mlr":
                            y_pred, fit_sec, predict_sec = _fit_predict_mlr(
                                X_ctx, y_ctx, X_te, is_nominal,
                            )
                        else:
                            y_pred, fit_sec, predict_sec = _fit_predict_tabpfn(
                                X_ctx, y_ctx, X_te, seed,
                                accept_text=not tabpfn_numeric,
                            )
                    except Exception as e:  # noqa: BLE001
                        write_jsonl(jsonl_path, [{
                            "skipped": True,
                            "reason": f"{type(e).__name__}: {e}",
                            **base_rec,
                        }], append=True)
                        logger.warning("failed %s k=%d mode=%s seed=%d: %s",
                                       model, k, mode, seed, e)
                        continue

                    y_pred_arr = np.asarray(y_pred, dtype=np.float64).ravel()
                    if not np.all(np.isfinite(y_pred_arr)):
                        n_bad = int(np.sum(~np.isfinite(y_pred_arr)))
                        write_jsonl(jsonl_path, [{
                            "skipped": True,
                            "reason": f"non-finite y_pred ({n_bad}/{y_pred_arr.size})",
                            **base_rec,
                        }], append=True)
                        logger.warning(
                            "failed %s k=%d mode=%s seed=%d: non-finite y_pred (%d/%d)",
                            model, k, mode, seed, n_bad, y_pred_arr.size,
                        )
                        continue

                    metrics = _metric_row(y_te, y_pred_arr)
                    rec = {
                        **base_rec,
                        "y_query_std": metrics["y_query_std"],
                        "nrmse": metrics["nrmse"], "r2": metrics["r2"],
                        "rmse": metrics["rmse"], "mae": metrics["mae"],
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(predict_sec),
                    }
                    write_jsonl(jsonl_path, [rec], append=True)

                    _write_predictions_csv(
                        row_dir / f"{_combo_stem(model, 'proportional', mode, k, seed)}.csv",
                        np.arange(n_query),
                        np.asarray(y_te, dtype=np.float64),
                        y_pred_arr,
                    )
                    logger.info(
                        "%s dataset=%s k=%d mode=%s seed=%d nrmse=%s fit=%.2fs predict=%.2fs",
                        model, dataset, k, mode, seed,
                        f"{metrics['nrmse']:.4f}" if metrics["nrmse"] is not None else "N/A",
                        fit_sec, predict_sec,
                    )


def _run_loo(
    dataset: str,
    row_dir: Path,
    jsonl_path: Path,
    k_list: list[int],
    modes: list[str],
    seeds: list[int],
    models: list[str],
    jitter_sigma: float | None = None,
    tabpfn_numeric: bool = False,
) -> None:
    for seed in seeds:
        set_seed(seed)
        X, y, _names, is_nominal = load_dataset_full(dataset, seed)
        n = X.shape[0]
        n_features = X.shape[1]
        all_idx = np.arange(n)

        for k in k_list:
            n_ctx = k * (n - 1)
            for mode in modes:
                for model in models:
                    base_rec = {
                        "dataset": dataset, "model": model,
                        "split_mode": "loo",
                        "k": int(k), "mode": mode, "seed": int(seed),
                        "n_ctx": int(n_ctx),
                        "n_query": 1, "n_folds": int(n),
                        "n_features": int(n_features),
                    }

                    y_true_all = np.empty(n, dtype=np.float64)
                    y_pred_all = np.empty(n, dtype=np.float64)
                    fit_sec_total = 0.0
                    predict_sec_total = 0.0
                    failed: Exception | None = None
                    nonfinite_fold: int | None = None
                    for i in range(n):
                        mask = all_idx != i
                        rng = np.random.default_rng(np.random.SeedSequence(
                            entropy=[seed, k, 0 if mode == "exact" else 1, i],
                        ))
                        X_ctx, y_ctx = duplicate_context(
                            X[mask], y[mask], k, mode, rng,
                            jitter_sigma=jitter_sigma,
                            is_nominal=is_nominal,
                        )
                        try:
                            if model == "mlr":
                                y_pred, fs, ps = _fit_predict_mlr(
                                    X_ctx, y_ctx, X[i : i + 1], is_nominal,
                                )
                            else:
                                y_pred, fs, ps = _fit_predict_tabpfn(
                                    X_ctx, y_ctx, X[i : i + 1], seed,
                                    accept_text=not tabpfn_numeric,
                                )
                        except Exception as e:  # noqa: BLE001
                            failed = e
                            break
                        yp = float(np.asarray(y_pred, dtype=np.float64).ravel()[0])
                        if not np.isfinite(yp):
                            nonfinite_fold = i
                            break
                        y_true_all[i] = float(y[i])
                        y_pred_all[i] = yp
                        fit_sec_total += fs
                        predict_sec_total += ps

                    if failed is not None:
                        write_jsonl(jsonl_path, [{
                            "skipped": True,
                            "reason": f"{type(failed).__name__}: {failed}",
                            **base_rec,
                        }], append=True)
                        logger.warning("failed %s (LOO) k=%d mode=%s seed=%d: %s",
                                       model, k, mode, seed, failed)
                        continue

                    if nonfinite_fold is not None:
                        write_jsonl(jsonl_path, [{
                            "skipped": True,
                            "reason": f"non-finite y_pred at fold {nonfinite_fold}",
                            **base_rec,
                        }], append=True)
                        logger.warning(
                            "failed %s (LOO) k=%d mode=%s seed=%d: non-finite y_pred at fold %d",
                            model, k, mode, seed, nonfinite_fold,
                        )
                        continue

                    metrics = _metric_row(y_true_all, y_pred_all)
                    rec = {
                        **base_rec,
                        "y_query_std": metrics["y_query_std"],
                        "nrmse": metrics["nrmse"], "r2": metrics["r2"],
                        "rmse": metrics["rmse"], "mae": metrics["mae"],
                        "fit_sec": float(fit_sec_total),
                        "predict_sec": float(predict_sec_total),
                    }
                    write_jsonl(jsonl_path, [rec], append=True)

                    _write_predictions_csv(
                        row_dir / f"{_combo_stem(model, 'loo', mode, k, seed)}.csv",
                        np.arange(n),
                        y_true_all,
                        y_pred_all,
                    )
                    logger.info(
                        "%s (LOO) dataset=%s k=%d mode=%s seed=%d nrmse=%s "
                        "fit=%.1fs predict=%.1fs",
                        model, dataset, k, mode, seed,
                        f"{metrics['nrmse']:.4f}" if metrics["nrmse"] is not None else "N/A",
                        fit_sec_total, predict_sec_total,
                    )


def run_row_probe(
    dataset: str,
    row_dir: Path,
    k_list: list[int],
    modes: list[str],
    seeds: list[int],
    include_tabpfn: bool = True,
    split_mode: str = "proportional",
    test_size: float | None = None,
    jitter_sigma: float | None = None,
    tabpfn_numeric: bool = False,
) -> None:
    """
    Run all (model, k, mode, seed) combos. Writes `metrics.jsonl` + one
    `predictions_<combo>.csv` per non-skipped combo into `row_dir`.

    ARGS:
      test_size: fraction of rows held out as the query set in
        `split_mode="proportional"`. None → use the per-dataset default from
        CONFIG (usually 0.2). Ignored when `split_mode="loo"`.
      jitter_sigma: Gaussian std used by `mode="jitter"`. None → use
        CONFIG["row_probe"]["jitter_sigma"] (usually 1e-6). 0 makes jitter
        numerically identical to exact.

    Existing files are APPENDED/overwritten in place; the caller (CLI) is
    responsible for clearing the dir ahead of time to enforce --fresh.
    """
    if split_mode not in VALID_SPLIT_MODES:
        raise ValueError(
            f"split_mode must be one of {VALID_SPLIT_MODES}, got {split_mode!r}"
        )
    if test_size is not None and not (0.0 < test_size < 1.0):
        raise ValueError(
            f"test_size must be in (0, 1), got {test_size!r}"
        )
    if jitter_sigma is not None and jitter_sigma < 0:
        raise ValueError(
            f"jitter_sigma must be >= 0, got {jitter_sigma!r}"
        )
    row_dir = ensure_dir(row_dir)
    jsonl_path = row_dir / "metrics.jsonl"

    models = ["mlr"] + (["tabpfn"] if include_tabpfn else [])

    if split_mode == "proportional":
        _run_proportional(
            dataset, row_dir, jsonl_path, k_list, modes, seeds, models,
            test_size=test_size, jitter_sigma=jitter_sigma,
            tabpfn_numeric=tabpfn_numeric,
        )
    else:
        if test_size is not None:
            logger.info(
                "test_size=%s ignored in split_mode='loo'", test_size,
            )
        _run_loo(
            dataset, row_dir, jsonl_path, k_list, modes, seeds, models,
            jitter_sigma=jitter_sigma,
            tabpfn_numeric=tabpfn_numeric,
        )
