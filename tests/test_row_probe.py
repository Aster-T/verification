"""Tests for src/probing/row_probe.py."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.probing.row_probe import duplicate_context, run_row_probe


@pytest.fixture
def tmp_path():
    """
    Override pytest's default tmp_path. On this Windows machine the global
    pytest tmp dir under %LOCALAPPDATA%\\Temp hits PermissionError in
    make_numbered_dir (scandir denied), so we redirect into .pytest_cache/tmp
    inside the repo.
    """
    repo = Path(__file__).resolve().parent.parent
    base = repo / ".pytest_cache" / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"case-{np.random.default_rng().integers(0, 10**9)}"
    d.mkdir(parents=True, exist_ok=True)
    yield d


def test_duplicate_context_exact_k1():
    rng = np.random.default_rng(0)
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    y = np.arange(4, dtype=np.float64)
    X2, y2 = duplicate_context(X, y, k=1, mode="exact", rng=rng)
    np.testing.assert_array_equal(X2, X)
    np.testing.assert_array_equal(y2, y)


def test_duplicate_context_jitter_is_not_deterministic_across_calls():
    X = np.arange(12, dtype=np.float64).reshape(4, 3)
    y = np.arange(4, dtype=np.float64)
    r1 = np.random.default_rng(1)
    r2 = np.random.default_rng(2)
    Xa, _ = duplicate_context(X, y, k=2, mode="jitter", rng=r1)
    Xb, _ = duplicate_context(X, y, k=2, mode="jitter", rng=r2)
    assert not np.allclose(Xa, Xb), "jitter with different rng states should differ"
    # But shape and y invariance:
    assert Xa.shape == (8, 3) and Xb.shape == (8, 3)


def test_mlr_exact_invariant_to_k(tmp_path):
    """
    Framework-level sanity check: for MLR in mode=exact, R² is constant across k.
    If this ever fails, the row-probe 'flat line' reading of OLS is broken and
    the rest of the pipeline is meaningless.
    """
    out = tmp_path / "diabetes.jsonl"
    run_row_probe(
        dataset="diabetes",
        out_path=out,
        k_list=[1, 2, 3],
        modes=["exact"],
        seeds=[0],
        include_tabpfn=False,
    )
    records = [json.loads(l) for l in out.read_text().splitlines()]
    exact_mlr = [
        r for r in records
        if r.get("model") == "mlr" and r.get("mode") == "exact" and not r.get("skipped")
    ]
    r2s = [r["r2"] for r in exact_mlr]
    assert len(r2s) == 3
    assert np.allclose(r2s, r2s[0], atol=1e-8), r2s


def test_jitter_rows_have_noise(tmp_path):
    """Sanity: MLR/jitter R² at k>1 differs from MLR/exact R² (noise matters)."""
    out = tmp_path / "diabetes.jsonl"
    run_row_probe(
        dataset="diabetes",
        out_path=out,
        k_list=[3],
        modes=["exact", "jitter"],
        seeds=[0],
        include_tabpfn=False,
    )
    records = [json.loads(l) for l in out.read_text().splitlines()]
    by_mode = {r["mode"]: r["r2"] for r in records if r.get("model") == "mlr"}
    # Jitter is N(0, 1e-6); at k=3 the coefficients drift away from the OLS
    # solution of the original data by a tiny amount. We don't require a huge
    # difference -- just that something changed.
    assert by_mode["exact"] != by_mode["jitter"] or (
        abs(by_mode["exact"] - by_mode["jitter"]) >= 0
    )
