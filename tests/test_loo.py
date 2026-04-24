"""Tests for src/probing/loo.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.loaders import load_dataset_full
from src.probing.loo import run_loo


@pytest.fixture
def tmp_path():
    """Repo-local tmp dir (avoid %LOCALAPPDATA%\\Temp PermissionError on Windows)."""
    repo = Path(__file__).resolve().parent.parent
    base = repo / ".pytest_cache" / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"loo-{np.random.default_rng().integers(0, 10**9)}"
    d.mkdir(parents=True, exist_ok=True)
    yield d


def test_load_dataset_full_matches_split_concat():
    """load_dataset_full and load_dataset should cover the same rows."""
    from src.data.loaders import load_dataset

    X_full, y_full, names_full, nom_full = load_dataset_full("synth_linear", seed=0)
    X_tr, y_tr, X_te, y_te, names, nom = load_dataset("synth_linear", seed=0)
    assert X_full.shape[0] == X_tr.shape[0] + X_te.shape[0]
    assert X_full.shape[1] == X_tr.shape[1]
    assert names_full == names
    assert nom_full == nom


def test_loo_single_idx_writes_one_row(tmp_path):
    res = run_loo("synth_linear", tmp_path, model="mlr", seed=0, idx=3)
    csv_path = tmp_path / "synth_linear" / "predictions_mlr_idx3.csv"
    json_path = tmp_path / "synth_linear" / "metrics_mlr_idx3.json"
    assert csv_path.exists() and json_path.exists()

    with csv_path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["idx", "y_true", "y_pred", "residual"]
    assert len(rows) == 2  # header + 1 row
    assert int(rows[1][0]) == 3
    # residual == y_true - y_pred
    yt, yp, resid = float(rows[1][1]), float(rows[1][2]), float(rows[1][3])
    assert np.isclose(resid, yt - yp)

    meta = json.loads(json_path.read_text())
    assert meta["n"] == 1
    assert meta["idx"] == 3
    assert meta["r2"] is None  # R² undefined for n=1
    assert res["n"] == 1


def test_loo_full_run_has_n_rows_and_finite_r2(tmp_path):
    """Full LOO on a small synthetic dataset. Use a tiny override to stay fast."""
    # Build a tiny synthetic problem in-memory via a registered dataset is overkill;
    # just run LOO on synth_linear (800 rows is still slow for tabpfn but MLR is fine).
    # Keep this test MLR-only.
    # To stay fast, we carve out a subset by registering a one-off dataset:
    from src.configs import CONFIG

    CONFIG["datasets"]["loo_tiny"] = {
        "loader": "make_regression",
        "n_samples": 30,
        "n_features": 4,
        "n_informative": 3,
        "noise": 1.0,
        "test_size": 0.2,
    }
    try:
        res = run_loo("loo_tiny", tmp_path, model="mlr", seed=0, idx=None)
    finally:
        del CONFIG["datasets"]["loo_tiny"]

    csv_path = tmp_path / "loo_tiny" / "predictions_mlr.csv"
    json_path = tmp_path / "loo_tiny" / "metrics_mlr.json"
    with csv_path.open() as f:
        rows = list(csv.reader(f))
    assert len(rows) == 1 + 30  # header + n predictions

    meta = json.loads(json_path.read_text())
    assert meta["n"] == 30
    assert meta["idx"] is None
    assert meta["r2"] is not None
    assert np.isfinite(meta["mse"]) and np.isfinite(meta["rmse"]) and np.isfinite(meta["mae"])
    assert res["n"] == 30


def test_loo_idx_out_of_range_raises(tmp_path):
    with pytest.raises(IndexError):
        run_loo("synth_linear", tmp_path, model="mlr", seed=0, idx=10**9)


def test_loo_unknown_model_raises(tmp_path):
    with pytest.raises(ValueError):
        run_loo("synth_linear", tmp_path, model="xgboost", seed=0, idx=0)
