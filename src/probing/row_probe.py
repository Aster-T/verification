"""
Row probe: measure how R²/RMSE/MAE react as the TabPFN context is duplicated.

We split (X, y) once per (dataset, seed), then for each k ∈ k_list and each
mode ∈ {"exact", "jitter"} build a context by duplicating (X_tr, y_tr) k times.
TabPFN and MLR are both evaluated on the SAME fixed test set.

For MLR in "exact" mode the R² is mathematically independent of k (OLS on a
uniformly replicated dataset returns the same coefficients). This is the
framework-level invariant verified in tests/test_row_probe.py.

JSONL SCHEMA (fields in this fixed order):
  dataset, model, k, mode, seed, n_ctx, n_te,
  r2, rmse, mae, fit_sec, predict_sec

A skipped combination writes:
  {"skipped": true, "reason": "...", dataset, model, k, mode, seed, n_ctx, n_te}
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.configs import CONFIG, get_device
from src.data.loaders import load_dataset
from src.models.mlr_wrapper import MLRWithW
from src.utils.io import write_jsonl
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def duplicate_context(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    mode: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tile (X, y) k-fold. In "jitter" mode, add N(0, sigma) noise to every
    duplicated X cell (original rows are unchanged in "exact" and still
    perturbed in "jitter" -- the sigma is tiny by design, so both start
    from the same distribution).

    ARGS:
      X:    (n, f) feature matrix.
      y:    (n,)   target vector.
      k:    positive integer tile count.
      mode: "exact" | "jitter".
      rng:  numpy Generator (required -- this function never touches globals).
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if mode not in ("exact", "jitter"):
        raise ValueError(f"mode must be 'exact' or 'jitter', got {mode!r}")
    X_rep = np.tile(X, (k, 1))
    y_rep = np.tile(y, k)
    if mode == "jitter":
        sigma = CONFIG["row_probe"]["jitter_sigma"]
        X_rep = X_rep + rng.standard_normal(X_rep.shape) * sigma
    return X_rep.astype(np.float64, copy=False), y_rep.astype(np.float64, copy=False)


def _metric_row(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _fit_predict_mlr(X_ctx, y_ctx, X_te, is_nominal):
    t0 = time.perf_counter()
    m = MLRWithW(standardize=True).fit(X_ctx, y_ctx, is_nominal=is_nominal)
    fit_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = m.predict(X_te)
    predict_sec = time.perf_counter() - t0
    return y_pred, fit_sec, predict_sec


def _fit_predict_tabpfn(X_ctx, y_ctx, X_te, seed):
    from src.models.tabpfn_wrapper import TabPFNWithColAttn  # noqa: PLC0415

    t0 = time.perf_counter()
    m = TabPFNWithColAttn(device=get_device(), seed=seed).fit(X_ctx, y_ctx)
    fit_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = m.predict(X_te)
    predict_sec = time.perf_counter() - t0
    return y_pred, fit_sec, predict_sec


def run_row_probe(
    dataset: str,
    out_path: Path,
    k_list: list[int],
    modes: list[str],
    seeds: list[int],
    include_tabpfn: bool = True,
) -> None:
    """
    Run all (model, k, mode, seed) combos and append to `out_path` as jsonl.

    Caller is responsible for deleting the output file ahead of time if
    `--fresh` semantics are wanted. Existing records are NOT resumed here --
    keep it simple; resume semantics can be added later if needed.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_ctx_limit = CONFIG["tabpfn"]["context_limit_n"]

    for seed in seeds:
        set_seed(seed)
        X_tr, y_tr, X_te, y_te, _names, is_nominal = load_dataset(dataset, seed)
        n_tr_original = X_tr.shape[0]
        n_te = X_te.shape[0]
        models = ["mlr"] + (["tabpfn"] if include_tabpfn else [])

        for k in k_list:
            n_ctx = k * n_tr_original
            for mode in modes:
                # rng for jitter: deterministic in (dataset, seed, k, mode).
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        entropy=[seed, k, 0 if mode == "exact" else 1]
                    )
                )
                X_ctx, y_ctx = duplicate_context(X_tr, y_tr, k, mode, rng)

                for model in models:
                    base_rec = {
                        "dataset": dataset,
                        "model": model,
                        "k": int(k),
                        "mode": mode,
                        "seed": int(seed),
                        "n_ctx": int(n_ctx),
                        "n_te": int(n_te),
                    }

                    if model == "tabpfn" and n_ctx > n_ctx_limit:
                        rec = {
                            "skipped": True,
                            "reason": f"n_ctx > {n_ctx_limit}",
                            **base_rec,
                        }
                        write_jsonl(out_path, [rec], append=True)
                        logger.info(
                            "skip tabpfn k=%d mode=%s seed=%d (n_ctx=%d)",
                            k, mode, seed, n_ctx,
                        )
                        continue

                    try:
                        if model == "mlr":
                            y_pred, fit_sec, predict_sec = _fit_predict_mlr(
                                X_ctx, y_ctx, X_te, is_nominal
                            )
                        else:
                            y_pred, fit_sec, predict_sec = _fit_predict_tabpfn(
                                X_ctx, y_ctx, X_te, seed
                            )
                    except Exception as e:  # noqa: BLE001
                        rec = {
                            "skipped": True,
                            "reason": f"{type(e).__name__}: {e}",
                            **base_rec,
                        }
                        write_jsonl(out_path, [rec], append=True)
                        logger.warning(
                            "failed %s k=%d mode=%s seed=%d: %s",
                            model, k, mode, seed, e,
                        )
                        continue

                    metrics = _metric_row(y_te, y_pred)
                    rec = {
                        **base_rec,
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(predict_sec),
                    }
                    write_jsonl(out_path, [rec], append=True)
                    logger.info(
                        "%s dataset=%s k=%d mode=%s seed=%d r2=%.4f fit=%.2fs predict=%.2fs",
                        model, dataset, k, mode, seed, metrics["r2"],
                        fit_sec, predict_sec,
                    )
