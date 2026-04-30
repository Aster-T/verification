"""
Column probe: fit MLR + TabPFN on a dataset, export W and column attention.

No plotting is done here -- see src/viz/heatmap.py.

OUTPUT LAYOUT:
  <column_dir>/mlr.npz         keys: w_vec, w_outer, feature_names
  <column_dir>/tabpfn.npz      keys: col_attn, col_attn_per_layer, feature_names
                               (omitted if TabPFN run failed/skipped)
  <column_dir>/meta.json       seed, n_tr, n_te, n_features,
                               sklearn/tabpfn versions, tabpfn_status.

`column_dir` is normally `results/<dataset>/column/`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from src.configs import CONFIG, get_device
from src.data.loaders import load_dataset
from src.models.mlr_wrapper import MLRWithW
from src.utils.io import dump_json, ensure_dir, save_npz
from src.utils.seed import set_seed

logger = logging.getLogger(__name__)


def run_column_probe(
    dataset: str,
    column_dir: Path,
    seed: int,
    *,
    tabpfn_weights: str = "v2_6",
) -> None:
    """
    Run MLR + TabPFN column probing for one dataset and persist numerical
    artefacts. See module docstring for the output layout.

    ARGS:
      dataset:    key in CONFIG["datasets"].
      column_dir: target directory (e.g. `results/<dataset>/column/`). Will be
                  created if missing. Files are written directly into it.
      seed:       forwarded to data loader, seed utility, and model seeds.
      tabpfn_weights: "v2_6" (default) → TabPFNRegressor's built-in "auto"
                  (latest bundled checkpoint). "v2" → pin the original v2
                  weights via TabPFNWithColAttn(use_v2_weights=True). The
                  CLI partitions the output path so v2 vs v2_6 ablations
                  don't overwrite each other.
    """
    if tabpfn_weights not in ("v2_6", "v2"):
        raise ValueError(
            f"tabpfn_weights must be 'v2_6' or 'v2', got {tabpfn_weights!r}"
        )
    import sklearn  # noqa: PLC0415

    target_dir = ensure_dir(column_dir)

    set_seed(seed)
    X_tr, y_tr, X_te, y_te, feature_names, is_nominal = load_dataset(dataset, seed)

    logger.info(
        "column_probe: dataset=%s n_tr=%d n_te=%d n_features=%d",
        dataset, X_tr.shape[0], X_te.shape[0], X_tr.shape[1],
    )

    # ---- MLR branch ----
    mlr = MLRWithW(standardize=True).fit(
        X_tr, y_tr, is_nominal=is_nominal, feature_names=feature_names,
    )
    W = mlr.get_W()
    save_npz(
        target_dir / "mlr.npz",
        w_vec=W["w_vec"],
        w_outer=W["w_outer"],
        feature_names=feature_names,
    )
    logger.info("saved %s", target_dir / "mlr.npz")

    # ---- TabPFN branch (optional) ----
    tabpfn_status = "ok"
    tabpfn_version = "unknown"
    try:
        from src.models.tabpfn_wrapper import TabPFNWithColAttn  # noqa: PLC0415
        import tabpfn  # noqa: PLC0415

        tabpfn_version = getattr(tabpfn, "__version__", "unknown")
        y_tr_f64 = np.asarray(y_tr, dtype=np.float64)
        tp = TabPFNWithColAttn(
            device=get_device(),
            seed=seed,
            use_v2_weights=(tabpfn_weights == "v2"),
        ).fit(X_tr, y_tr_f64)
        _ = tp.predict(X_te)
        col_attn = tp.get_col_attn(reduce="mean")
        col_attn_per_layer = tp.get_col_attn(reduce="per_layer")
        save_npz(
            target_dir / "tabpfn.npz",
            col_attn=col_attn,
            col_attn_per_layer=col_attn_per_layer,
            feature_names=feature_names,
        )
        logger.info("saved %s", target_dir / "tabpfn.npz")
    except Exception as e:  # noqa: BLE001
        tabpfn_status = f"skipped: {type(e).__name__}: {e}"
        logger.warning("TabPFN branch skipped for %s: %s", dataset, tabpfn_status)

    # ---- Meta ----
    meta = {
        "dataset": dataset,
        "seed": seed,
        "n_tr": int(X_tr.shape[0]),
        "n_te": int(X_te.shape[0]),
        "n_features": int(X_tr.shape[1]),
        "feature_names": feature_names,
        "is_nominal": list(is_nominal),
        "sklearn_version": sklearn.__version__,
        "tabpfn_version": tabpfn_version,
        "tabpfn_status": tabpfn_status,
        "tabpfn_weights": tabpfn_weights,
        "config_snapshot": {
            "attn_reduce": CONFIG["column_probe"]["attn_reduce"],
        },
    }
    dump_json(target_dir / "meta.json", meta)

    # ---- Alignment assert ----
    mlr_path = target_dir / "mlr.npz"
    tp_path = target_dir / "tabpfn.npz"
    if mlr_path.exists() and tp_path.exists():
        with np.load(mlr_path, allow_pickle=True) as a:
            a_names = [str(x) for x in a["feature_names"].tolist()]
        with np.load(tp_path, allow_pickle=True) as b:
            b_names = [str(x) for x in b["feature_names"].tolist()]
        assert a_names == b_names, (
            f"feature_names misaligned for {dataset}: MLR={a_names} vs TabPFN={b_names}"
        )

