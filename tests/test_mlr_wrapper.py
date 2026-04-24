"""Tests for src/models/mlr_wrapper.py."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import load_diabetes

from src.models.mlr_wrapper import MLRWithW


@pytest.fixture(scope="module")
def diabetes():
    data = load_diabetes()
    return data.data, data.target, list(data.feature_names)


def test_basic_shapes(diabetes):
    X, y, names = diabetes
    m = MLRWithW().fit(X, y, feature_names=names)
    W = m.get_W()
    assert W["w_vec"].shape == (10,)
    assert W["w_outer"].shape == (10, 10)
    assert W["feature_names"] == names
    # predict returns one value per row
    assert m.predict(X[:5]).shape == (5,)


def test_standardization_balances_coef_scales():
    """
    When one column dominates raw-variance, standardize=True should produce
    coefficients whose magnitudes are MUCH more uniform than standardize=False.
    """
    rng = np.random.default_rng(0)
    n, f = 300, 4
    X = rng.standard_normal((n, f))
    # Blow up column 0's scale; true y depends equally on all columns.
    X[:, 0] *= 1000.0
    beta_std_space = np.array([1.0, 1.0, 1.0, 1.0])
    # Generate y with equal *standardized-space* effects.
    X_std = X / X.std(axis=0, ddof=0)
    y = X_std @ beta_std_space + rng.standard_normal(n) * 0.1

    w_on = MLRWithW(standardize=True).fit(X, y).get_W()["w_vec"]
    w_off = MLRWithW(standardize=False).fit(X, y).get_W()["w_vec"]

    ratio_on = np.abs(w_on).max() / np.abs(w_on).min()
    ratio_off = np.abs(w_off).max() / np.abs(w_off).min()
    # standardized coefs should be MUCH closer in magnitude
    assert ratio_on < ratio_off / 10, (ratio_on, ratio_off)


def test_duplicate_sample_invariance(diabetes):
    """
    Framework-level sanity check: OLS on a uniformly tripled dataset must
    produce identical w_vec. This underpins the MLR/exact baseline in the
    row probe. If this fails, the row-probe "flat line" invariant is lost.
    """
    X, y, _ = diabetes
    m1 = MLRWithW().fit(X, y)
    X2, y2 = np.tile(X, (3, 1)), np.tile(y, 3)
    m2 = MLRWithW().fit(X2, y2)
    assert np.allclose(
        m1.get_W()["w_vec"], m2.get_W()["w_vec"], atol=1e-8
    )


def test_nominal_skips_standardization():
    rng = np.random.default_rng(1)
    n = 200
    cat = rng.integers(0, 3, size=n).astype(float)  # {0, 1, 2}
    cont = rng.standard_normal(n) * 10.0 + 5.0
    X = np.column_stack([cat, cont])
    y = cat + 0.3 * cont + rng.standard_normal(n) * 0.1

    m = MLRWithW().fit(X, y, is_nominal=[True, False])
    assert m.mu_[0] == 0.0
    assert m.sd_[0] == 1.0
    # Continuous column was standardized.
    assert m.mu_[1] != 0.0
    assert m.sd_[1] != 1.0
