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


def test_fit_with_string_column_auto_factorizes():
    """MLR receives an object ndarray with a string column; it factorizes
    internally, marks that column nominal, and trains / predicts cleanly."""
    rng = np.random.default_rng(2)
    n = 150
    text_col = rng.choice(["red", "green", "blue"], size=n)
    num_col = rng.standard_normal(n) * 5.0 + 2.0
    # Encode hidden "true" effect of category to verify the factorization
    # preserves learnability.
    hidden = np.where(text_col == "red", 1.0,
             np.where(text_col == "green", 2.0, 3.0))
    y = 0.4 * num_col + hidden + rng.standard_normal(n) * 0.05

    X = np.empty((n, 2), dtype=object)
    X[:, 0] = text_col
    X[:, 1] = num_col

    m = MLRWithW().fit(X, y, feature_names=["color", "num"])
    # Column 0 was auto-detected as text → factorized → marked nominal.
    assert m.is_nominal_[0]
    assert m.is_nominal_[1] is np.False_ or m.is_nominal_[1] == False  # noqa: E712
    assert m._cat_mappings_ == {0: ["blue", "green", "red"]}
    # Predict on unseen rows (same categories) — should fit reasonably.
    y_pred = m.predict(X)
    assert y_pred.shape == (n,)
    # Since the relationship is clean, R² should be high.
    r2 = 1 - ((y - y_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    assert r2 > 0.95, r2


def test_predict_with_unseen_category_becomes_nan_then_imputed():
    """A test-time category not present at fit time becomes NaN via the
    stored mapping, then gets filled by the built-in SimpleImputer."""
    rng = np.random.default_rng(3)
    n = 100
    text_col = rng.choice(["A", "B"], size=n)
    num_col = rng.standard_normal(n)
    y = np.where(text_col == "A", 1.0, -1.0) + 0.1 * num_col

    X_train = np.empty((n, 2), dtype=object)
    X_train[:, 0] = text_col; X_train[:, 1] = num_col

    m = MLRWithW().fit(X_train, y)
    # Predict with a brand-new category "C" — should not raise.
    X_test = np.empty((3, 2), dtype=object)
    X_test[:, 0] = ["A", "B", "C"]  # "C" is unseen
    X_test[:, 1] = [0.0, 0.0, 0.0]
    y_pred = m.predict(X_test)
    assert y_pred.shape == (3,)
    assert np.all(np.isfinite(y_pred))
