"""Tests for src/data/loaders.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.loaders import load_dataset


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
