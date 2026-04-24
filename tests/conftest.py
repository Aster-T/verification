"""Test-scoped fixtures shared across the suite.

1. Overrides pytest's default tmp_path to write under the repo's
   .pytest_cache, because this Windows host hits PermissionError in pytest's
   own make_numbered_dir() under %LOCALAPPDATA%\\Temp.
2. Auto-registers the built-in test datasets (diabetes / synth_linear /
   cali_housing) into CONFIG for the duration of each test. CONFIG["datasets"]
   is intentionally empty at module load (runtime data comes from presets /
   local_csv auto-discovery), but several tests still reference these names.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_path():
    base = _REPO_ROOT / ".pytest_cache" / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"case-{np.random.default_rng().integers(0, 10**9)}"
    d.mkdir(parents=True, exist_ok=True)
    yield d


_BUILTIN_TEST_DATASETS = {
    "diabetes": {
        "loader": "sklearn_builtin",
        "sklearn_name": "diabetes",
        "test_size": 0.2,
        "subsample": None,
    },
    "synth_linear": {
        "loader": "make_regression",
        "n_samples": 800,
        "n_features": 12,
        "n_informative": 6,
        "noise": 5.0,
        "test_size": 0.2,
    },
    "cali_housing": {
        "loader": "sklearn_builtin",
        "sklearn_name": "california_housing",
        "test_size": 0.2,
        "subsample": 2000,
    },
}


@pytest.fixture(autouse=True)
def _register_builtin_test_datasets():
    """Ensure hard-coded dataset names referenced in tests resolve, without
    committing them to the runtime CONFIG. Clean up afterwards."""
    from src.configs import CONFIG

    added: list[str] = []
    for name, cfg in _BUILTIN_TEST_DATASETS.items():
        if name not in CONFIG["datasets"]:
            CONFIG["datasets"][name] = dict(cfg)
            added.append(name)
    yield
    for name in added:
        CONFIG["datasets"].pop(name, None)
