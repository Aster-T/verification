"""Tests for scripts/infer_meta.py (import as module)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "_infer_meta", REPO / "scripts" / "infer_meta.py"
)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
infer_meta = _mod.infer_meta


def _write_csv(path: Path, rows: list[list], header: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_basic_numeric_csv(tmp_path):
    ds = tmp_path / "ds"
    ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(
        csv_path,
        [[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]],
        ["x1", "x2", "y"],
    )
    meta = infer_meta(csv_path)
    assert meta["name"] == "ds"
    assert meta["target_col"] == "y"
    assert meta["feature_names"] == ["x1", "x2"]
    assert meta["is_nominal"] == [False, False]
    assert meta["n_rows"] == 3
    assert meta["n_features"] == 2
    assert meta["source"] == "user_supplied"
    assert len(meta["sha256"]) == 64


def test_explicit_target_col(tmp_path):
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(
        csv_path,
        [[100.0, 1.0, 2.0], [200.0, 3.0, 4.0]],
        ["price", "x1", "x2"],  # target is NOT last
    )
    meta = infer_meta(csv_path, target_col="price")
    assert meta["target_col"] == "price"
    assert meta["feature_names"] == ["x1", "x2"]


def test_factorize_text_column(tmp_path):
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(
        csv_path,
        [["A", 1.0, 10.0], ["B", 2.0, 20.0], ["A", 3.0, 30.0]],
        ["region", "x1", "y"],
    )
    meta = infer_meta(csv_path, factorize_text=True)
    assert meta["is_nominal"] == [True, False]
    df = pd.read_csv(csv_path)
    # region must now be numeric
    assert pd.api.types.is_numeric_dtype(df["region"])
    # A/B -> 0/1 via pd.factorize(sort=True)
    assert set(df["region"].unique()) == {0.0, 1.0}


def test_no_factorize_errors_on_text(tmp_path):
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(
        csv_path,
        [["A", 1.0, 10.0], ["B", 2.0, 20.0]],
        ["region", "x1", "y"],
    )
    with pytest.raises(ValueError, match="non-numeric"):
        infer_meta(csv_path, factorize_text=False)


def test_target_must_be_numeric(tmp_path):
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(
        csv_path,
        [[1.0, 2.0, "yes"], [3.0, 4.0, "no"]],
        ["x1", "x2", "label"],
    )
    with pytest.raises(ValueError, match="regression-only"):
        infer_meta(csv_path)


def test_force_nominal_for_integer_column(tmp_path):
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(
        csv_path,
        [[0, 1.0, 10.0], [1, 2.0, 20.0], [2, 3.0, 30.0]],
        ["day", "x1", "y"],
    )
    meta = infer_meta(csv_path, extra_nominal={"day"})
    assert meta["is_nominal"] == [True, False]


def test_nan_preserved_when_factorizing(tmp_path):
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(
        csv_path,
        [["A", 1.0, 10.0], ["", 2.0, 20.0], ["A", 3.0, 30.0]],
        ["region", "x1", "y"],
    )
    infer_meta(csv_path, factorize_text=True)
    df = pd.read_csv(csv_path)
    assert df["region"].isna().sum() == 1


def test_unknown_target_col_raises(tmp_path):
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(csv_path, [[1.0, 2.0]], ["x1", "y"])
    with pytest.raises(KeyError, match="not in CSV columns"):
        infer_meta(csv_path, target_col="nope")


def test_unknown_nominal_col_raises(tmp_path):
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(csv_path, [[1.0, 2.0]], ["x1", "y"])
    with pytest.raises(KeyError, match="not in feature columns"):
        infer_meta(csv_path, extra_nominal={"nope"})
