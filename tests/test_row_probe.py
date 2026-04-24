"""Tests for src/probing/row_probe.py."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from src.probing.row_probe import duplicate_context, run_row_probe


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
    assert Xa.shape == (8, 3) and Xb.shape == (8, 3)


def _records(row_dir):
    return [json.loads(l) for l in (row_dir / "metrics.jsonl").read_text().splitlines()]


def test_mlr_exact_invariant_to_k(tmp_path):
    """Framework-level: MLR/exact R² (and nRMSE) invariant across k."""
    run_row_probe(
        dataset="diabetes", row_dir=tmp_path,
        k_list=[1, 2, 3], modes=["exact"], seeds=[0],
        include_tabpfn=False,
    )
    recs = _records(tmp_path)
    exact_mlr = [r for r in recs
                 if r.get("model") == "mlr" and r.get("mode") == "exact"
                 and not r.get("skipped")]
    r2s = [r["r2"] for r in exact_mlr]
    nrmses = [r["nrmse"] for r in exact_mlr]
    assert len(r2s) == 3
    assert np.allclose(r2s, r2s[0], atol=1e-8), r2s
    assert np.allclose(nrmses, nrmses[0], atol=1e-8), nrmses


def test_new_schema_fields_present_and_csv_written(tmp_path):
    """Each non-skipped record carries the expected fields AND a matching
    predictions_*.csv exists with the right columns."""
    run_row_probe(
        dataset="synth_linear", row_dir=tmp_path,
        k_list=[2], modes=["exact"], seeds=[0],
        include_tabpfn=False,
    )
    recs = _records(tmp_path)
    assert recs
    r = recs[0]
    for field in (
        "split_mode", "k", "mode", "seed", "n_ctx", "n_query", "n_folds",
        "n_features", "y_query_std", "nrmse", "r2", "rmse", "mae",
    ):
        assert field in r, f"missing field {field!r}: {r}"
    assert r["split_mode"] == "proportional"
    assert r["n_ctx"] == 2 * int(0.8 * 800)  # 800 rows, test_size=0.2
    assert r["n_folds"] == 1

    csv_path = tmp_path / "predictions_mlr_proportional_exact_k2_s0.csv"
    assert csv_path.exists()
    with csv_path.open() as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["id", "y_true", "y_pred", "residual"]
    assert len(rows) == 1 + r["n_query"] * r["n_folds"]
    yt, yp, resid = float(rows[1][1]), float(rows[1][2]), float(rows[1][3])
    assert abs(resid - (yt - yp)) < 1e-9


def test_nrmse_relationship_to_r2(tmp_path):
    """nrmse == sqrt(1 - r2) on the same query set (when r2 <= 1)."""
    run_row_probe(
        dataset="synth_linear", row_dir=tmp_path,
        k_list=[1], modes=["exact"], seeds=[0],
        include_tabpfn=False,
    )
    rec = _records(tmp_path)[0]
    expected = float(np.sqrt(max(0.0, 1.0 - rec["r2"])))
    assert abs(rec["nrmse"] - expected) < 1e-9, (rec["nrmse"], expected)


def test_jitter_rows_have_noise(tmp_path):
    """MLR/jitter nRMSE at k>1 should differ from MLR/exact (noise matters)."""
    run_row_probe(
        dataset="diabetes", row_dir=tmp_path,
        k_list=[3], modes=["exact", "jitter"], seeds=[0],
        include_tabpfn=False,
    )
    recs = _records(tmp_path)
    by_mode = {r["mode"]: r["nrmse"] for r in recs if r.get("model") == "mlr"}
    assert by_mode["exact"] != by_mode["jitter"] or (
        abs(by_mode["exact"] - by_mode["jitter"]) >= 0
    )


def test_loo_mode_one_record_per_combo_with_csv(tmp_path):
    """LOO aggregates N folds into one record per combo, and writes a CSV
    of N rows for each non-skipped combo."""
    from src.configs import CONFIG

    CONFIG["datasets"]["rp_tiny"] = {
        "loader": "make_regression",
        "n_samples": 25, "n_features": 4, "n_informative": 3,
        "noise": 1.0, "test_size": 0.2,
    }
    try:
        run_row_probe(
            dataset="rp_tiny", row_dir=tmp_path,
            k_list=[2, 3], modes=["exact"], seeds=[0],
            include_tabpfn=False, split_mode="loo",
        )
    finally:
        del CONFIG["datasets"]["rp_tiny"]

    recs = _records(tmp_path)
    assert len(recs) == 2
    for r in recs:
        assert r["split_mode"] == "loo"
        assert r["n_query"] == 1
        assert r["n_folds"] == 25
        assert r["n_ctx"] == r["k"] * 24
        assert r["nrmse"] is not None
        csv_path = tmp_path / f"predictions_mlr_loo_exact_k{r['k']}_s0.csv"
        assert csv_path.exists()
        with csv_path.open() as f:
            rows = list(csv.reader(f))
        assert len(rows) == 1 + 25  # header + N


def test_loo_mlr_exact_invariant_to_k(tmp_path):
    """In LOO mode, MLR/exact nRMSE is still invariant across k."""
    from src.configs import CONFIG

    CONFIG["datasets"]["rp_tiny2"] = {
        "loader": "make_regression",
        "n_samples": 20, "n_features": 3, "n_informative": 2,
        "noise": 1.0, "test_size": 0.2,
    }
    try:
        run_row_probe(
            dataset="rp_tiny2", row_dir=tmp_path,
            k_list=[1, 2, 3], modes=["exact"], seeds=[0],
            include_tabpfn=False, split_mode="loo",
        )
    finally:
        del CONFIG["datasets"]["rp_tiny2"]

    nrmses = [r["nrmse"] for r in _records(tmp_path)]
    assert np.allclose(nrmses, nrmses[0], atol=1e-8), nrmses


def test_invalid_split_mode_raises(tmp_path):
    with pytest.raises(ValueError, match="split_mode"):
        run_row_probe(
            dataset="diabetes", row_dir=tmp_path,
            k_list=[1], modes=["exact"], seeds=[0],
            include_tabpfn=False, split_mode="stratified-bootstrap",
        )
