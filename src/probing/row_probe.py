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

PRIMARY METRIC: nRMSE = RMSE / std(y_full)
  - Dimensionless, scale-invariant -> comparable across datasets.
  - 0 = perfect; ~1 = predicting the dataset-mean baseline; >1 = worse.
  - Denominator is std over the FULL loaded target — not the held-out
    query — so the metric stays stable across seeds in proportional
    mode (small test sets otherwise make std(y_query) jitter and so
    nRMSE jitters with it). For LOO the query equals the full set, so
    the two coincide.
  - Returns null when std(y_full) == 0 (degenerate constant target).
  - R² intentionally still uses std(y_query) — it's the variance-
    explained ratio on the query set itself, an intrinsic statistic.

OUTPUT LAYOUT (for a given dataset, under `row_dir`):
  metrics.jsonl
     One record per (model, split_mode, k, mode, seed). Schema:
       dataset, model, split_mode, k, mode, seed,
       n_ctx, n_query, n_folds, n_features,
       y_query_std,        # std over the actual query rows (diagnostic)
       y_denom_std,        # std actually used as the nRMSE denominator —
                           # set to std(y_full) so nRMSE is stable across
                           # seeds (it doesn't fluctuate with whichever
                           # rows happen to land in the test split).
       nrmse, r2, rmse, mae, mape, mape_n, fit_sec, predict_sec
     mape  = mean(|y_true - y_pred| / |y_true|), restricted to
             rows where y_true != 0. null when every y_true is 0.
     mape_n = number of query rows that contributed to mape (i.e.
              y_true != 0). For datasets with zero targets (e.g.
              forest-fires), mape_n < n_query.
     mape_tanker / mape_tanker_n: same definition but restricted to
              the 9 tanker vessels (船号 ∈ _TANKER_IDS). Present only
              for `ship-all` and `ship-selected` records.
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.configs import CONFIG, get_dataset_cfg, get_device
from src.data.loaders import load_dataset_full
from src.models.mlr_wrapper import MLRWithW
from src.utils.io import ensure_dir, iter_jsonl, write_jsonl
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)

VALID_SPLIT_MODES = ("proportional", "loo")

# Datasets where we additionally compute MAPE restricted to tanker rows.
# `ship-tanker` is excluded because it already contains only tankers
# (mape_tanker would just equal mape).
_TANKER_DATASETS = frozenset({"ship-all", "ship-selected"})
_SHIP_ID_COL = "船号"
_TANKER_IDS = frozenset({
    "G1099", "G1107", "G1113", "G1132", "G1171",
    "G1186", "G1194", "G1198", "GTEST",
})

# Fields the current writer emits on every NON-skipped record. Used by the
# schema-aware resume path: records missing any of these are considered
# stale, dropped from the jsonl, and re-fit on the next run. Bumping this
# tuple is what tells old runs "your row needs to be redone".
_REQUIRED_FIELDS_BASE = (
    "dataset", "model", "split_mode", "k", "mode", "seed",
    "n_ctx", "n_query", "n_folds", "n_features",
    "y_query_std", "y_denom_std",
    "nrmse", "r2", "rmse", "mae",
    "mape", "mape_n",
    "fit_sec", "predict_sec",
)
# Extra fields required for ship-all / ship-selected (per-tanker MAPE).
_REQUIRED_FIELDS_TANKER = ("mape_tanker", "mape_tanker_n")


def _record_is_current(rec: dict, dataset: str) -> bool:
    """Schema check used by the resume path. A skipped record is always
    'current' — its semantic value (we tried this combo and it raised)
    doesn't get invalidated by adding new metric fields elsewhere."""
    if rec.get("skipped"):
        return True
    for f in _REQUIRED_FIELDS_BASE:
        if f not in rec:
            return False
    if dataset in _TANKER_DATASETS:
        for f in _REQUIRED_FIELDS_TANKER:
            if f not in rec:
                return False
    return True


def _resume_keys(jsonl_path: Path, dataset: str) -> set[tuple]:
    """Schema-aware resume: scan an existing metrics.jsonl, drop records
    that are missing fields the current writer would emit (so they get
    re-fit on this run), and return the (model, k, mode, seed) tuples
    that should be skipped because their record IS up-to-date.

    Side effect: if any stale records were found, the file is rewritten
    in place without them. Skipped records are always preserved (the
    'we tried, it failed' note is still useful)."""
    if not jsonl_path.exists():
        return set()
    keep: list[dict] = []
    done: set[tuple] = set()
    n_total = 0
    n_dropped = 0
    for rec in iter_jsonl(jsonl_path):
        n_total += 1
        if _record_is_current(rec, dataset):
            keep.append(rec)
            try:
                done.add((rec["model"], rec["k"], rec["mode"], rec["seed"]))
            except KeyError:
                # Record without the combo key isn't usable for resume; drop.
                n_dropped += 1
                keep.pop()
        else:
            n_dropped += 1
    if n_dropped > 0:
        logger.info(
            "schema-aware resume: dropping %d/%d stale record(s) from %s "
            "(missing fields the current writer emits); rewriting file",
            n_dropped, n_total, jsonl_path.name,
        )
        write_jsonl(jsonl_path, keep, append=False)
    return done


def _tanker_mask(
    dataset: str,
    feature_names: list[str],
    X_query: np.ndarray,
) -> np.ndarray | None:
    """Boolean mask over `X_query` rows belonging to the 9 tanker ships.
    Returns None when the dataset is not in `_TANKER_DATASETS` or the ship
    column is not in `feature_names` — caller treats None as 'skip the
    tanker metric'."""
    if dataset not in _TANKER_DATASETS:
        return None
    if _SHIP_ID_COL not in feature_names:
        return None
    j = feature_names.index(_SHIP_ID_COL)
    col = X_query[:, j]
    return np.asarray([str(v) in _TANKER_IDS for v in col], dtype=bool)


def duplicate_context(
    X: np.ndarray,
    y: np.ndarray,
    k: int,
    mode: str,
    rng: np.random.Generator,
    jitter_sigma: float | None = None,
    is_nominal: list[bool] | None = None,
    jitter_scale: str = "absolute",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Tile (X, y) k-fold. In "jitter" mode, add gaussian noise to the
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
      jitter_scale: how σ relates to per-column scale.
        "absolute"    -> noise[j] ~ N(0, σ²) for every numeric column j
                         (legacy behavior; the same σ regardless of how big
                         the column's values are).
        "per_col_std" -> noise[j] ~ N(0, (σ · std_j)²) where std_j is the
                         std of column j on the **untiled** X. Makes σ a
                         dimensionless relative perturbation strength so
                         the same σ behaves comparably across columns and
                         datasets of vastly different magnitudes. Constant
                         columns (std_j == 0) still receive zero noise.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if mode not in ("exact", "jitter"):
        raise ValueError(f"mode must be 'exact' or 'jitter', got {mode!r}")
    if jitter_scale not in ("absolute", "per_col_std"):
        raise ValueError(
            f"jitter_scale must be 'absolute' or 'per_col_std', "
            f"got {jitter_scale!r}"
        )
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
                # add gaussian noise to numeric columns only.
                for j in range(X_rep.shape[1]):
                    if nominal_mask is not None and nominal_mask[j]:
                        continue
                    col = X_rep[:, j]
                    # Skip columns that aren't numeric (safety: an undeclared
                    # text column would break the float cast).
                    if any(isinstance(v, str) for v in col):
                        continue
                    float_col = np.array(col, dtype=np.float64)
                    if jitter_scale == "per_col_std":
                        # std on the original (untiled) rows, not X_rep.
                        col_std = float(np.std(float_col[:n], ddof=0))
                        eff_sigma = sigma * col_std  # 0 -> column stays put
                    else:
                        eff_sigma = sigma
                    noise_col = (
                        rng.standard_normal(float_col.shape[0]) * eff_sigma
                    )
                    # Anchor: rows 0..n-1 are tile 0 — keep pristine.
                    noise_col[:n] = 0.0
                    float_col += noise_col
                    X_rep[:, j] = float_col
            else:
                X_rep_f = X_rep.astype(np.float64, copy=False)
                noise = rng.standard_normal(X_rep_f.shape) * sigma
                if jitter_scale == "per_col_std":
                    # std per column on the original (untiled) rows, not the
                    # k-fold tile (the tile has the same std as X anyway,
                    # but anchoring on X is unambiguous).
                    col_std = np.std(X_rep_f[:n], axis=0, ddof=0)
                    noise = noise * col_std[np.newaxis, :]
                # Anchor: rows 0..n-1 are tile 0 — keep pristine.
                noise[:n, :] = 0.0
                if nominal_mask is not None:
                    # Suppress noise on nominal columns so integer codes stay
                    # integer-valued (preserves the category semantics).
                    noise[:, nominal_mask] = 0.0
                X_rep = X_rep_f + noise
    y_out = y_rep.astype(np.float64, copy=False)
    # Keep object dtype when present (caller decides how to use it); cast
    # numeric case to float64 for downstream math.
    if X_rep.dtype == object:
        return X_rep, y_out
    return X_rep.astype(np.float64, copy=False), y_out


def _metric_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    denom_std: float | None = None,
    tanker_mask: np.ndarray | None = None,
) -> dict:
    """Regression metrics. nrmse = RMSE / `denom_std` when provided, else
    RMSE / std(y_true). When `denom_std == 0` or None and std(y_true) == 0,
    nrmse is None; r² is NaN when n<2 or std(y_true) == 0.

    Why `denom_std` exists: in proportional split mode std(y_query) is
    computed over a small held-out test set and fluctuates seed-to-seed,
    making nRMSE less stable as a cross-(seed, k) summary. Caller passes
    std(y_full) — the standard deviation of the loaded dataset's full
    target — so nRMSE is normalised by a stable per-(dataset, seed)
    constant. R² intentionally still uses std(y_true): it's the variance-
    explained ratio on the query set itself, not a normalisation choice.
    For LOO the query IS the full dataset, so passing std(y_full) is a
    no-op equality.

    mape = mean(|y_true - y_pred| / |y_true|) over rows where y_true != 0.
    Rows with y_true == 0 are excluded (their per-point relative error is
    undefined). mape is None when every y_true is 0; mape_n reports how
    many rows contributed.

    When `tanker_mask` is provided (boolean ndarray, same length as y_true),
    additionally compute the same MAPE restricted to rows where the mask is
    True AND y_true != 0, returned as `mape_tanker` / `mape_tanker_n`. Used
    by ship-* datasets to track relative error on the 9 tanker vessels
    separately from the rest of the fleet."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    y_query_std = float(np.std(y_true, ddof=0))
    denom = denom_std if denom_std is not None else y_query_std
    nrmse: float | None = float(rmse / denom) if denom > 0 else None
    r2 = (
        float(r2_score(y_true, y_pred))
        if (y_true.size >= 2 and y_query_std > 0)
        else float("nan")
    )
    nz = y_true != 0
    mape_n = int(nz.sum())
    mape: float | None = (
        float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])))
        if mape_n > 0 else None
    )
    out = {
        "nrmse": nrmse,
        "r2": r2,
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mape": mape,
        "mape_n": mape_n,
        "y_query_std": y_query_std,
        # Actual denominator used for nRMSE. Equals y_query_std when caller
        # didn't pass denom_std (legacy path) and equals std(y_full) for the
        # row probe today.
        "y_denom_std": float(denom),
    }
    if tanker_mask is not None:
        m = np.asarray(tanker_mask, dtype=bool).ravel()
        if m.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"tanker_mask length {m.shape[0]} != y_true length {y_true.shape[0]}"
            )
        nz_t = nz & m
        mape_tanker_n = int(nz_t.sum())
        out["mape_tanker"] = (
            float(np.mean(np.abs((y_true[nz_t] - y_pred[nz_t]) / y_true[nz_t])))
            if mape_tanker_n > 0 else None
        )
        out["mape_tanker_n"] = mape_tanker_n
    return out


def _fit_predict_mlr(X_ctx, y_ctx, X_te, is_nominal):
    t0 = time.perf_counter()
    m = MLRWithW(standardize=True).fit(X_ctx, y_ctx, is_nominal=is_nominal)
    fit_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = m.predict(X_te)
    predict_sec = time.perf_counter() - t0
    return y_pred, fit_sec, predict_sec


def _prep_tabpfn_X(X, *, accept_text: bool):
    """Hand TabPFN an input that lets it see each column under its true
    dtype.

    - Numeric ndarrays pass through.
    - Object ndarrays with `accept_text=True` are wrapped into a pandas
      DataFrame, with text columns kept as object and numeric columns
      coerced to float; TabPFN's own categorical handling kicks in for
      the text columns.
    - With `accept_text=False` (i.e. CLI --tabpfn-numeric) we cast the
      whole object array to float so non-numeric values surface as NaN
      and TabPFN treats every column as numeric.
    """
    X_arr = np.asarray(X)
    if X_arr.dtype != object:
        return X_arr
    if not accept_text:
        return X_arr.astype(np.float64)
    import pandas as pd  # noqa: PLC0415  (lazy)

    df = pd.DataFrame(X_arr).copy()
    for j in range(df.shape[1]):
        col = df.iloc[:, j]
        if not any(isinstance(v, str) for v in col):
            df.iloc[:, j] = pd.to_numeric(col, errors="coerce")
    return df


def _fit_predict_tabpfn(X_ctx, y_ctx, X_te, seed, *, accept_text: bool = True):
    """Row-probe path: use TabPFNRegressor straight from the box with its
    default inference config (default n_estimators ensemble, default
    PREPROCESS_TRANSFORMS / FEATURE_SHIFT / POLYNOMIAL_FEATURES / etc).

    Why not the column-probe wrapper TabPFNWithColAttn:
      - That wrapper hardcodes n_estimators=1 and disables every TabPFN
        feature that would shuffle / augment columns, because column
        probing needs the captured attention matrix to stay axis-aligned
        with the input column order.
      - Row probing doesn't consume attention; it only consumes
        predictions. Forcing n_estimators=1 + zero-augmentation actively
        hurts predictive accuracy with no upside, so this path uses
        TabPFN's stock ensemble instead.
    """
    from tabpfn import TabPFNRegressor  # noqa: PLC0415

    X_ctx_in = _prep_tabpfn_X(X_ctx, accept_text=accept_text)
    X_te_in = _prep_tabpfn_X(X_te, accept_text=accept_text)
    y_ctx_in = np.asarray(y_ctx, dtype=np.float64).ravel()

    t0 = time.perf_counter()
    m = TabPFNRegressor(
        device=get_device(),
        random_state=int(seed),
        ignore_pretraining_limits=True,
    )
    m.fit(X_ctx_in, y_ctx_in)
    fit_sec = time.perf_counter() - t0
    t0 = time.perf_counter()
    y_pred = m.predict(X_te_in)
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
    jitter_scale: str = "absolute",
    tabpfn_numeric: bool = False,
    parallel_k: int = 1,
) -> None:
    effective_test_size = (
        test_size if test_size is not None
        else float(get_dataset_cfg(dataset).get("test_size", 0.2))
    )
    # Schema-aware resume: combos with a current-schema record are skipped;
    # combos whose existing record is missing fields the current writer
    # emits (e.g. an old jsonl predating MAPE) get dropped from the file
    # and re-fit on this run. Caller controls "wipe everything" via
    # run_row_probe.py --fresh, which clears row_dir before we get here.
    done_keys = _resume_keys(jsonl_path, dataset)
    if done_keys:
        logger.info(
            "resume dataset=%s split_mode=proportional: %d combos already in %s, will skip them",
            dataset, len(done_keys), jsonl_path.name,
        )

    for seed in seeds:
        set_seed(seed)
        X, y, feature_names, is_nominal = load_dataset_full(dataset, seed)
        # std over the full loaded target — stable per (dataset, seed) and
        # used as the nRMSE denominator so the metric doesn't fluctuate with
        # which rows happen to land in the test split.
        y_full_std = float(np.std(np.asarray(y, dtype=np.float64), ddof=0))
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=effective_test_size, random_state=seed,
        )
        n_tr_original = X_tr.shape[0]
        n_query = X_te.shape[0]
        n_features = X_tr.shape[1]
        tanker_mask = _tanker_mask(dataset, feature_names, X_te)
        logger.info(
            "split dataset=%s seed=%d test_size=%g n_train=%d n_test=%d "
            "n_features=%d n_nominal=%d n_tanker_in_query=%s y_full_std=%.4g",
            dataset, seed, effective_test_size,
            n_tr_original, n_query, n_features,
            sum(is_nominal) if is_nominal is not None else 0,
            int(tanker_mask.sum()) if tanker_mask is not None else "n/a",
            y_full_std,
        )

        # Build the (k, mode, model) task list for this seed, skipping any
        # combos that resume already covered. Each task is an independent
        # fit+predict — duplicate_context runs *inside* the worker so the
        # CPU jitter prep can overlap with another worker's GPU compute.
        # Note: the legacy code shared X_ctx between MLR and TabPFN within
        # a (k, mode); we lose that micro-optimization, but jitter prep is
        # cheap relative to fit, so the parallelism win dwarfs it.
        tasks = [
            (int(k), mode, model)
            for k in k_list for mode in modes for model in models
            if (model, int(k), mode, int(seed)) not in done_keys
        ]
        if not tasks:
            continue

        def _compute(k: int, mode: str, model: str):
            rng = np.random.default_rng(np.random.SeedSequence(
                entropy=[seed, k, 0 if mode == "exact" else 1],
            ))
            X_ctx, y_ctx = duplicate_context(
                X_tr, y_tr, k, mode, rng,
                jitter_sigma=jitter_sigma,
                is_nominal=is_nominal,
                jitter_scale=jitter_scale,
            )
            if model == "mlr":
                return _fit_predict_mlr(X_ctx, y_ctx, X_te, is_nominal)
            return _fit_predict_tabpfn(
                X_ctx, y_ctx, X_te, seed,
                accept_text=not tabpfn_numeric,
            )

        def _process(k: int, mode: str, model: str, exc, result) -> None:
            base_rec = {
                "dataset": dataset,
                "model": model,
                "split_mode": "proportional",
                "k": int(k),
                "mode": mode,
                "seed": int(seed),
                "n_ctx": int(k * n_tr_original),
                "n_query": int(n_query),
                "n_folds": 1,
                "n_features": int(n_features),
                "jitter_scale": jitter_scale,
            }
            if exc is not None:
                write_jsonl(jsonl_path, [{
                    "skipped": True,
                    "reason": f"{type(exc).__name__}: {exc}",
                    **base_rec,
                }], append=True)
                logger.warning("failed %s k=%d mode=%s seed=%d: %s",
                               model, k, mode, seed, exc)
                return
            y_pred, fit_sec, predict_sec = result
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
                return
            metrics = _metric_row(
                y_te, y_pred_arr,
                denom_std=y_full_std,
                tanker_mask=tanker_mask,
            )
            rec = {
                **base_rec,
                "y_query_std": metrics["y_query_std"],
                "y_denom_std": metrics["y_denom_std"],
                "nrmse": metrics["nrmse"], "r2": metrics["r2"],
                "rmse": metrics["rmse"], "mae": metrics["mae"],
                "mape": metrics["mape"], "mape_n": metrics["mape_n"],
                "fit_sec": float(fit_sec),
                "predict_sec": float(predict_sec),
            }
            if "mape_tanker" in metrics:
                rec["mape_tanker"] = metrics["mape_tanker"]
                rec["mape_tanker_n"] = metrics["mape_tanker_n"]
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

        if parallel_k <= 1:
            # Legacy serial path. Identical writes/log lines as before.
            for k, mode, model in tasks:
                try:
                    res = _compute(k, mode, model)
                    _process(k, mode, model, None, res)
                except Exception as e:  # noqa: BLE001
                    _process(k, mode, model, e, None)
        else:
            # All writes happen on the main thread (inside _process, called
            # from the as_completed iterator), so jsonl/CSV stay single-
            # writer without explicit locking. Workers are pure compute.
            with ThreadPoolExecutor(max_workers=parallel_k) as pool:
                futs = {
                    pool.submit(_compute, k, mode, model): (k, mode, model)
                    for k, mode, model in tasks
                }
                for fut in as_completed(futs):
                    k, mode, model = futs[fut]
                    try:
                        res = fut.result()
                    except Exception as e:  # noqa: BLE001
                        _process(k, mode, model, e, None)
                        continue
                    _process(k, mode, model, None, res)


def _run_loo(
    dataset: str,
    row_dir: Path,
    jsonl_path: Path,
    k_list: list[int],
    modes: list[str],
    seeds: list[int],
    models: list[str],
    jitter_sigma: float | None = None,
    jitter_scale: str = "absolute",
    tabpfn_numeric: bool = False,
    parallel_k: int = 1,
) -> None:
    # Schema-aware resume — see _run_proportional for the rationale.
    done_keys = _resume_keys(jsonl_path, dataset)
    if done_keys:
        logger.info(
            "resume dataset=%s split_mode=loo: %d combos already in %s, will skip them",
            dataset, len(done_keys), jsonl_path.name,
        )

    for seed in seeds:
        set_seed(seed)
        X, y, feature_names, is_nominal = load_dataset_full(dataset, seed)
        n = X.shape[0]
        n_features = X.shape[1]
        all_idx = np.arange(n)
        # std of the full target — for LOO this equals std(y_query) by
        # construction (every row is queried once), but we still pass it
        # explicitly so the jsonl carries the same y_denom_std field as
        # proportional records.
        y_full_std = float(np.std(np.asarray(y, dtype=np.float64), ddof=0))
        tanker_mask = _tanker_mask(dataset, feature_names, X)
        logger.info(
            "split dataset=%s seed=%d split_mode=loo n_folds=%d "
            "n_features=%d n_nominal=%d n_tanker_in_query=%s y_full_std=%.4g",
            dataset, seed, n, n_features,
            sum(is_nominal) if is_nominal is not None else 0,
            int(tanker_mask.sum()) if tanker_mask is not None else "n/a",
            y_full_std,
        )

        for k in k_list:
            n_ctx = k * (n - 1)
            for mode in modes:
                for model in models:
                    if (model, int(k), mode, int(seed)) in done_keys:
                        logger.info(
                            "skip %s (LOO) dataset=%s k=%d mode=%s seed=%d: already done",
                            model, dataset, k, mode, seed,
                        )
                        continue
                    base_rec = {
                        "dataset": dataset, "model": model,
                        "split_mode": "loo",
                        "k": int(k), "mode": mode, "seed": int(seed),
                        "n_ctx": int(n_ctx),
                        "n_query": 1, "n_folds": int(n),
                        "n_features": int(n_features),
                        "jitter_scale": jitter_scale,
                    }

                    y_true_all = np.empty(n, dtype=np.float64)
                    y_pred_all = np.empty(n, dtype=np.float64)
                    fit_sec_total = 0.0
                    predict_sec_total = 0.0
                    failed: Exception | None = None
                    nonfinite_fold: int | None = None

                    def _fold(i: int):
                        """Pure-compute LOO fold. Returns (y_pred_scalar,
                        fit_sec, predict_sec). Raises on fit/predict error;
                        caller handles non-finite checks."""
                        mask = all_idx != i
                        rng = np.random.default_rng(np.random.SeedSequence(
                            entropy=[seed, k, 0 if mode == "exact" else 1, i],
                        ))
                        X_ctx, y_ctx = duplicate_context(
                            X[mask], y[mask], k, mode, rng,
                            jitter_sigma=jitter_sigma,
                            is_nominal=is_nominal,
                            jitter_scale=jitter_scale,
                        )
                        if model == "mlr":
                            y_pred, fs, ps = _fit_predict_mlr(
                                X_ctx, y_ctx, X[i : i + 1], is_nominal,
                            )
                        else:
                            y_pred, fs, ps = _fit_predict_tabpfn(
                                X_ctx, y_ctx, X[i : i + 1], seed,
                                accept_text=not tabpfn_numeric,
                            )
                        return y_pred, fs, ps

                    def _consume(i: int, y_pred, fs: float, ps: float) -> bool:
                        """Stash one fold's result. Returns False when the
                        prediction is non-finite (caller must record the
                        offending fold and stop). Updates closure-captured
                        nonfinite_fold and the y_*_all / *_sec_total
                        accumulators in place."""
                        nonlocal nonfinite_fold, fit_sec_total, predict_sec_total
                        yp = float(np.asarray(y_pred, dtype=np.float64).ravel()[0])
                        if not np.isfinite(yp):
                            nonfinite_fold = i
                            return False
                        y_true_all[i] = float(y[i])
                        y_pred_all[i] = yp
                        fit_sec_total += fs
                        predict_sec_total += ps
                        return True

                    # MLR is CPU-only; threading just adds GIL contention
                    # over BLAS calls. Only parallelize TabPFN.
                    use_parallel = parallel_k > 1 and model == "tabpfn"
                    if not use_parallel:
                        for i in range(n):
                            try:
                                y_pred, fs, ps = _fold(i)
                            except Exception as e:  # noqa: BLE001
                                failed = e
                                break
                            if not _consume(i, y_pred, fs, ps):
                                break
                    else:
                        with ThreadPoolExecutor(max_workers=parallel_k) as pool:
                            futs = {pool.submit(_fold, i): i for i in range(n)}
                            try:
                                for fut in as_completed(futs):
                                    i = futs[fut]
                                    try:
                                        y_pred, fs, ps = fut.result()
                                    except Exception as e:  # noqa: BLE001
                                        failed = e
                                        break
                                    if not _consume(i, y_pred, fs, ps):
                                        break
                            finally:
                                # Cancel pending tasks on early exit so the
                                # pool tears down quickly; running futures
                                # finish naturally (cancel is a no-op on
                                # already-started work).
                                for f in futs:
                                    f.cancel()

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

                    metrics = _metric_row(
                        y_true_all, y_pred_all,
                        denom_std=y_full_std,
                        tanker_mask=tanker_mask,
                    )
                    rec = {
                        **base_rec,
                        "y_query_std": metrics["y_query_std"],
                        "y_denom_std": metrics["y_denom_std"],
                        "nrmse": metrics["nrmse"], "r2": metrics["r2"],
                        "rmse": metrics["rmse"], "mae": metrics["mae"],
                        "mape": metrics["mape"], "mape_n": metrics["mape_n"],
                        "fit_sec": float(fit_sec_total),
                        "predict_sec": float(predict_sec_total),
                    }
                    if "mape_tanker" in metrics:
                        rec["mape_tanker"] = metrics["mape_tanker"]
                        rec["mape_tanker_n"] = metrics["mape_tanker_n"]
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
    jitter_scale: str = "absolute",
    tabpfn_numeric: bool = False,
    parallel_k: int | None = None,
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
      jitter_scale: "absolute" (default, legacy) or "per_col_std". Controls
        whether σ is the noise std directly or a multiplier on each column's
        std. The two are intended as ablation siblings — caller is expected
        to partition the output path on this value (see scripts/run_row_probe
        .py) so records from the two variants don't collide.
      parallel_k: number of TabPFN fit/predict slots to run concurrently
        on the GPU. None → CONFIG["row_probe"]["parallel_k"] (default 5,
        sized for a 48 GB card). 1 disables threading and recovers the
        legacy serial path. In proportional mode the pool spans (k, mode,
        model) tasks per seed; in LOO mode it spans the N folds within
        each (model='tabpfn', k, mode, seed). MLR folds always run serial
        — they're CPU-bound and threading just hits BLAS contention.

    RESUME / FRESH semantics:
      The runners skip any (model, k, mode, seed) combo whose record already
      lives in `metrics.jsonl` (read once at startup). To force a full re-run
      the caller must clear `row_dir` first — that's exactly what `--fresh`
      does on the CLI side. Without --fresh, this function resumes from where
      a previous invocation left off (or was interrupted), and skipped combos
      keep their existing CSVs.
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
    if jitter_scale not in ("absolute", "per_col_std"):
        raise ValueError(
            f"jitter_scale must be 'absolute' or 'per_col_std', "
            f"got {jitter_scale!r}"
        )
    effective_parallel_k = (
        int(parallel_k) if parallel_k is not None
        else int(CONFIG["row_probe"].get("parallel_k", 1))
    )
    if effective_parallel_k < 1:
        raise ValueError(
            f"parallel_k must be >= 1, got {effective_parallel_k!r}"
        )
    row_dir = ensure_dir(row_dir)
    jsonl_path = row_dir / "metrics.jsonl"

    models = ["mlr"] + (["tabpfn"] if include_tabpfn else [])

    effective_sigma = (
        jitter_sigma if jitter_sigma is not None
        else float(CONFIG["row_probe"]["jitter_sigma"])
    )
    sigma_str = f"{effective_sigma:g}" if "jitter" in modes else "n/a"
    scale_str = jitter_scale if "jitter" in modes else "n/a"
    test_size_str: str
    if split_mode == "proportional":
        if test_size is not None:
            test_size_str = f"{test_size:g}"
        else:
            test_size_str = (
                f"{float(get_dataset_cfg(dataset).get('test_size', 0.2)):g}"
                f" (from CONFIG)"
            )
    else:
        test_size_str = "n/a (loo)"
    logger.info(
        "row_probe start dataset=%s split_mode=%s models=%s "
        "k_list=%s modes=%s seeds=%s test_size=%s jitter_sigma=%s "
        "jitter_scale=%s tabpfn_numeric=%s parallel_k=%d out=%s",
        dataset, split_mode, models, k_list, modes, seeds,
        test_size_str, sigma_str, scale_str, tabpfn_numeric,
        effective_parallel_k, row_dir,
    )

    if split_mode == "proportional":
        _run_proportional(
            dataset, row_dir, jsonl_path, k_list, modes, seeds, models,
            test_size=test_size, jitter_sigma=jitter_sigma,
            jitter_scale=jitter_scale,
            tabpfn_numeric=tabpfn_numeric,
            parallel_k=effective_parallel_k,
        )
    else:
        if test_size is not None:
            logger.info(
                "test_size=%s ignored in split_mode='loo'", test_size,
            )
        _run_loo(
            dataset, row_dir, jsonl_path, k_list, modes, seeds, models,
            jitter_sigma=jitter_sigma,
            jitter_scale=jitter_scale,
            tabpfn_numeric=tabpfn_numeric,
            parallel_k=effective_parallel_k,
        )
