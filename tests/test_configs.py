"""Tests for small helpers in src/configs.py."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from src.configs import (
    CONFIG,
    add_openml_cli_args,
    load_openml_config,
    parse_openml_spec,
    register_openml_dataset,
    resolve_openml_args,
)


def _write_presets(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_parse_bare_id():
    assert parse_openml_spec("560") == (560, None)


def test_parse_id_with_name():
    assert parse_openml_spec("560:bodyfat") == (560, "bodyfat")


def test_parse_id_with_blank_name_is_none():
    # '560:' means "user typed the colon but left name empty" — treat as None.
    assert parse_openml_spec("560:") == (560, None)


def test_parse_handles_whitespace():
    assert parse_openml_spec(" 42 : foo ") == (42, "foo")


def test_parse_rejects_non_integer_id():
    with pytest.raises(ValueError):
        parse_openml_spec("abc")
    with pytest.raises(ValueError):
        parse_openml_spec("abc:name")


def test_register_openml_with_custom_name():
    name = register_openml_dataset(99999901, name="test_custom_ds")
    try:
        assert name == "test_custom_ds"
        cfg = CONFIG["datasets"]["test_custom_ds"]
        assert cfg["loader"] == "openml_id"
        assert cfg["openml_id"] == 99999901
    finally:
        CONFIG["datasets"].pop("test_custom_ds", None)


def test_register_openml_default_name():
    name = register_openml_dataset(99999902)
    try:
        assert name == "openml_99999902"
        assert "openml_99999902" in CONFIG["datasets"]
    finally:
        CONFIG["datasets"].pop("openml_99999902", None)


def test_register_is_idempotent():
    n1 = register_openml_dataset(99999903, name="idem")
    n2 = register_openml_dataset(99999903, name="idem")
    try:
        assert n1 == n2 == "idem"
        assert CONFIG["datasets"]["idem"]["openml_id"] == 99999903
    finally:
        CONFIG["datasets"].pop("idem", None)


# --------------------------------------------------------------------- presets

def test_load_openml_config_skips_underscore_keys(tmp_path):
    path = _write_presets(tmp_path / "p.json", {
        "_comment": "a note",
        "foo": {"id": 1, "subsample": 100},
        "bar": {"id": 2},
    })
    out = load_openml_config(path)
    assert set(out.keys()) == {"foo", "bar"}
    assert out["foo"]["id"] == 1


def test_load_openml_config_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_openml_config(tmp_path / "nope.json")


def test_load_openml_config_bad_entry_errors(tmp_path):
    path = _write_presets(tmp_path / "p.json", {"bad": {"name": "missing id"}})
    with pytest.raises(ValueError, match="'id'"):
        load_openml_config(path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", action="append", default=[])
    add_openml_cli_args(p)
    return p


def test_resolve_ad_hoc_openml_id_only():
    parser = _build_parser()
    args = parser.parse_args(["--openml-id", "99999801:testds_a"])
    try:
        names = resolve_openml_args(args)
        assert names == ["testds_a"]
        assert CONFIG["datasets"]["testds_a"]["openml_id"] == 99999801
    finally:
        CONFIG["datasets"].pop("testds_a", None)


def test_resolve_preset(tmp_path):
    presets = _write_presets(tmp_path / "p.json", {
        "testds_b": {"id": 99999802, "subsample": 50},
    })
    parser = _build_parser()
    args = parser.parse_args([
        "--openml-config", str(presets),
        "--openml-preset", "testds_b",
    ])
    try:
        names = resolve_openml_args(args)
        assert names == ["testds_b"]
        assert CONFIG["datasets"]["testds_b"]["subsample"] == 50
    finally:
        CONFIG["datasets"].pop("testds_b", None)


def test_resolve_preset_respects_cli_subsample_when_preset_omits_it(tmp_path):
    presets = _write_presets(tmp_path / "p.json", {
        "testds_c": {"id": 99999803},  # no subsample
    })
    parser = _build_parser()
    args = parser.parse_args([
        "--openml-config", str(presets),
        "--openml-preset", "testds_c",
        "--openml-subsample", "123",
    ])
    try:
        resolve_openml_args(args)
        assert CONFIG["datasets"]["testds_c"]["subsample"] == 123
    finally:
        CONFIG["datasets"].pop("testds_c", None)


def test_resolve_openml_all(tmp_path):
    presets = _write_presets(tmp_path / "p.json", {
        "testds_d": {"id": 99999804},
        "testds_e": {"id": 99999805},
    })
    parser = _build_parser()
    args = parser.parse_args(["--openml-config", str(presets), "--openml-all"])
    try:
        names = resolve_openml_args(args)
        assert set(names) == {"testds_d", "testds_e"}
    finally:
        CONFIG["datasets"].pop("testds_d", None)
        CONFIG["datasets"].pop("testds_e", None)


def test_resolve_unknown_preset_errors(tmp_path):
    presets = _write_presets(tmp_path / "p.json", {"known": {"id": 99999806}})
    parser = _build_parser()
    args = parser.parse_args([
        "--openml-config", str(presets),
        "--openml-preset", "unknown",
    ])
    try:
        with pytest.raises(KeyError, match="unknown"):
            resolve_openml_args(args)
    finally:
        CONFIG["datasets"].pop("known", None)


def test_resolve_no_openml_args_returns_empty():
    parser = _build_parser()
    args = parser.parse_args([])  # nothing
    assert resolve_openml_args(args) == []


def test_cli_default_subsample_is_unlimited():
    parser = _build_parser()
    args = parser.parse_args(["--openml-id", "99999890:testds_h"])
    try:
        resolve_openml_args(args)
        assert CONFIG["datasets"]["testds_h"]["subsample"] is None
    finally:
        CONFIG["datasets"].pop("testds_h", None)


def test_cli_subsample_zero_means_unlimited():
    parser = _build_parser()
    args = parser.parse_args([
        "--openml-id", "99999891:testds_i", "--openml-subsample", "0",
    ])
    try:
        resolve_openml_args(args)
        assert CONFIG["datasets"]["testds_i"]["subsample"] is None
    finally:
        CONFIG["datasets"].pop("testds_i", None)


def test_cli_subsample_negative_means_unlimited():
    parser = _build_parser()
    args = parser.parse_args([
        "--openml-id", "99999892:testds_j", "--openml-subsample", "-1",
    ])
    try:
        resolve_openml_args(args)
        assert CONFIG["datasets"]["testds_j"]["subsample"] is None
    finally:
        CONFIG["datasets"].pop("testds_j", None)


def test_cli_subsample_positive_applies():
    parser = _build_parser()
    args = parser.parse_args([
        "--openml-id", "99999893:testds_k", "--openml-subsample", "500",
    ])
    try:
        resolve_openml_args(args)
        assert CONFIG["datasets"]["testds_k"]["subsample"] == 500
    finally:
        CONFIG["datasets"].pop("testds_k", None)


def test_resolve_combines_ad_hoc_and_preset(tmp_path):
    presets = _write_presets(tmp_path / "p.json", {
        "testds_f": {"id": 99999807},
    })
    parser = _build_parser()
    args = parser.parse_args([
        "--openml-id", "99999808:testds_g",
        "--openml-config", str(presets),
        "--openml-preset", "testds_f",
    ])
    try:
        names = resolve_openml_args(args)
        assert names == ["testds_g", "testds_f"]  # ad-hoc first, then presets
    finally:
        CONFIG["datasets"].pop("testds_f", None)
        CONFIG["datasets"].pop("testds_g", None)
