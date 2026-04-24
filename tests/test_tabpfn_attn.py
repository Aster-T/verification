"""Tests for src/models/tabpfn_wrapper.py."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.datasets import load_diabetes

from src.models.tabpfn_wrapper import (
    TabPFNWithColAttn,
    capture_column_attention,
)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def diabetes_small():
    data = load_diabetes()
    X = data.data[:200, :8].astype(np.float64)
    y = data.target[:200].astype(np.float64)
    return X, y


@pytest.fixture(scope="module")
def fitted(diabetes_small):
    X, y = diabetes_small
    m = TabPFNWithColAttn(device=_device(), seed=0).fit(X, y)
    # Pre-compute one predict so get_col_attn has a _last_attn to reduce.
    _ = m.predict(X)
    return m


def test_col_attn_shape(fitted):
    attn = fitted.get_col_attn()
    assert attn.shape == (8, 8)
    assert attn.dtype == np.float64


def test_col_attn_rows_sum_to_one(fitted):
    """
    Softmax-based column attention: each query row over the F+1 (incl. target)
    feature axis sums to 1. After stripping the target column, the residual row
    sum can be < 1; so we verify sum-to-1 on the UNSTRIPPED per-layer tensor
    by re-running capture directly on the inner model over a small input.
    """
    # Capture raw (unstripped) attention so the softmax invariant is clean.
    # We do this by reaching into the model and running one predict under capture,
    # asserting on the raw tensor before any trimming.
    X, y = load_diabetes(return_X_y=True)
    X = X[:200, :8].astype(np.float64)
    y = y[:200].astype(np.float64)

    with capture_column_attention(fitted._inner_model) as captured:
        _ = fitted._regressor.predict(X)

    # The capture list here corresponds to THIS predict only.
    assert len(captured) > 0
    for _, t in captured:
        # t: (B, H, F+, F+) -- last dim sums to 1 under softmax.
        row_sum = t.sum(dim=-1)
        assert torch.allclose(
            row_sum, torch.ones_like(row_sum), atol=1e-4
        ), f"row sum deviates; max abs err {torch.max(torch.abs(row_sum - 1)).item():.2e}"


def test_context_manager_restores_forward(diabetes_small):
    """
    After the context manager exits, predicting again without it should still
    work and not leave any _captured state behind on the module.
    """
    X, y = diabetes_small
    m = TabPFNWithColAttn(device=_device(), seed=0).fit(X, y)
    # Record the forward attributes BEFORE entering the context.
    from src.models.tabpfn_wrapper import _find_feature_attention_modules

    attn_mods = [mod for _, mod in _find_feature_attention_modules(m._inner_model)]
    assert len(attn_mods) > 0
    before = [mod.forward for mod in attn_mods]

    with capture_column_attention(m._inner_model) as _captured:
        _ = m._regressor.predict(X)
    # After exit, forwards are restored.
    after = [mod.forward for mod in attn_mods]
    assert before == after

    # Predict again, should not raise.
    _ = m._regressor.predict(X)


def test_predict_is_deterministic(diabetes_small):
    """With n_estimators=1 and fixed seed, two predicts should agree exactly."""
    X, y = diabetes_small
    m = TabPFNWithColAttn(device=_device(), seed=0).fit(X, y)
    y1 = m.predict(X)
    y2 = m.predict(X)
    assert np.allclose(y1, y2, atol=1e-6)


def test_per_layer_reduce_shape(fitted):
    per_layer = fitted.get_col_attn(reduce="per_layer")
    assert per_layer.ndim == 3
    assert per_layer.shape[1:] == (8, 8)
    last = fitted.get_col_attn(reduce="last")
    assert last.shape == (8, 8)
    assert np.allclose(last, per_layer[-1])
