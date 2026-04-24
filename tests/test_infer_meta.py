"""Tests for scripts/infer_meta.py (import as module).

New contract (non-destructive):
  - data.csv is NEVER modified by infer_meta
  - text columns stay as text on disk; categorical_mappings records
    category→code for downstream loaders to apply consistently
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np  # noqa: F401  (available for test authoring)
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
    ds = tmp_path / "ds"; ds.mkdir()
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
    # Numeric-only CSV: no mappings recorded.
    assert meta["categorical_mappings"] == {}


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


def test_text_column_is_nominal_and_csv_untouched(tmp_path):
    """Text column → is_nominal=True, categorical_mappings recorded, and
    the original CSV bytes are NOT modified."""
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(
        csv_path,
        [["A", 1.0, 10.0], ["B", 2.0, 20.0], ["A", 3.0, 30.0]],
        ["region", "x1", "y"],
    )
    original_bytes = csv_path.read_bytes()
    meta = infer_meta(csv_path)
    assert meta["is_nominal"] == [True, False]
    # CSV is NOT rewritten — the strings survive on disk.
    assert csv_path.read_bytes() == original_bytes
    df = pd.read_csv(csv_path)
    assert list(df["region"]) == ["A", "B", "A"]
    # Mapping is recorded (sort=True → 'A'=0, 'B'=1).
    assert meta["categorical_mappings"] == {"region": ["A", "B"]}


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
    # Mapping is recorded even for integer-encoded nominals.
    assert "day" in meta["categorical_mappings"]


def test_categorical_mappings_sorted(tmp_path):
    ds = tmp_path / "ds"; ds.mkdir()
    csv_path = ds / "data.csv"
    _write_csv(
        csv_path,
        [["beta", 1.0, 10.0], ["alpha", 2.0, 20.0], ["beta", 3.0, 30.0]],
        ["tag", "x1", "y"],
    )
    meta = infer_meta(csv_path)
    # sort=True → 'alpha' is code 0, 'beta' is code 1
    assert meta["categorical_mappings"] == {"tag": ["alpha", "beta"]}


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
