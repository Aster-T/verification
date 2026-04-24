"""Test-scoped fixtures shared across the suite.

Overrides pytest's default tmp_path to write under the repo's .pytest_cache,
because this Windows host hits PermissionError in pytest's own
make_numbered_dir() under %LOCALAPPDATA%\\Temp.
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
