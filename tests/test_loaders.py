"""Tests for src/data/loaders.py."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from src.data.loaders import load_dataset, load_dataset_full


@pytest.mark.parametrize(
    "name,expected_f",
    [("diabetes", 10), ("synth_linear", 12)],
)
def test_loader_shapes_and_split(name, expected_f):
    X_tr, y_tr, X_te, y_te, names, is_nominal = load_dataset(name, seed=0)
    assert X_tr.shape[1] == expected_f
    assert X_te.shape[1] == expected_f
    assert X_tr.shape[0] == y_tr.shape[0]
    assert X_te.shape[0] == y_te.shape[0]
    assert len(names) == expected_f
    assert len(is_nominal) == expected_f
    # Disjoint train/test (sanity via simple row-hash check).
    tr_rows = {tuple(r) for r in X_tr}
    te_rows = {tuple(r) for r in X_te}
    assert tr_rows.isdisjoint(te_rows)


def test_cali_housing_subsample():
    """California housing is subsampled to 2000 rows per configs.CONFIG."""
    X_tr, y_tr, X_te, y_te, names, is_nominal = load_dataset("cali_housing", seed=0)
    assert X_tr.shape[0] + X_te.shape[0] == 2000
    assert X_tr.shape[1] == 8
    assert len(names) == 8
    assert all(not b for b in is_nominal)


def test_seed_determinism():
    a = load_dataset("synth_linear", seed=42)
    b = load_dataset("synth_linear", seed=42)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[2], b[2])


def test_load_dataset_full_matches_split_concat():
    """load_dataset_full covers exactly the rows that load_dataset splits."""
    X_full, y_full, names_full, nom_full = load_dataset_full("synth_linear", seed=0)
    X_tr, y_tr, X_te, y_te, names, nom = load_dataset("synth_linear", seed=0)
    assert X_full.shape[0] == X_tr.shape[0] + X_te.shape[0]
    assert X_full.shape[1] == X_tr.shape[1]
    assert names_full == names
    assert nom_full == nom


def test_local_csv_auto_discovery():
    """Drop data.csv + meta.json under datasets/<name>/, then call
    load_dataset_full(name) — configs.get_dataset_cfg should register the
    local_csv loader on the fly."""
    from src.configs import CONFIG

    repo = Path(__file__).resolve().parent.parent
    name = "_test_local_csv_auto"
    ds_dir = repo / "datasets" / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ds_dir / "data.csv"
    meta_path = ds_dir / "meta.json"

    # Tiny linear dataset y = 2*x1 + 3*x2 + noise
    rng = np.random.default_rng(0)
    N = 15
    X1 = rng.standard_normal(N)
    X2 = rng.standard_normal(N)
    y_vals = 2 * X1 + 3 * X2 + rng.standard_normal(N) * 0.1
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x1", "x2", "y"])
        for a, b, c in zip(X1, X2, y_vals):
            w.writerow([a, b, c])

    meta_path.write_text(json.dumps({
        "name": name, "source": "user_supplied",
        "target_col": "y",
        "feature_names": ["x1", "x2"],
        "is_nominal": [False, False],
        "n_rows": N, "n_features": 2,
    }), encoding="utf-8")

    try:
        X, y_arr, feature_names, is_nominal = load_dataset_full(name, seed=0)
        assert X.shape == (N, 2)
        assert y_arr.shape == (N,)
        assert feature_names == ["x1", "x2"]
        assert is_nominal == [False, False]
    finally:
        CONFIG["datasets"].pop(name, None)
        for p in (csv_path, meta_path):
            if p.exists():
                p.unlink()
        if ds_dir.exists():
            ds_dir.rmdir()
