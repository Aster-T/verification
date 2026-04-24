"""
Seed helpers.

Call `set_seed(seed)` once at the start of any entry point
(scripts/run_*.py, top-level of run_column_probe / run_row_probe) to pin
python / numpy / torch / cuda RNGs together.

For local, isolated control prefer `make_rng(seed)` returning a numpy
Generator and pass it explicitly to the function that needs it
(see src/probing/row_probe.py::duplicate_context). Falling back to global
state is acceptable only for libraries that do not expose a random_state
argument -- in that case wrap the call in `temp_seed(...)`.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager

import numpy as np


def set_seed(seed: int, deterministic: bool = False) -> None:
    """
    Seed python / numpy / torch / cuda in one call.

    ARGS:
      seed: integer seed.
      deterministic: if True, force cuDNN into deterministic mode (slower
        but bit-reproducible on GPU). Default False for speed.

    NOTES:
      torch is imported lazily, so this module stays importable on
      CPU-only machines without torch installed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_rng(seed: int) -> np.random.Generator:
    """
    Return a fresh numpy Generator.

    Prefer this over np.random.* globals whenever the function you are
    calling accepts an explicit rng argument -- it avoids hidden coupling
    between unrelated code paths sharing the global RNG.
    """
    return np.random.default_rng(seed)


@contextmanager
def temp_seed(seed: int):
    """
    Temporarily seed python / numpy / torch, restoring previous RNG state
    on exit.

    USE WHEN:
      A library reads from global RNG state and does not accept an
      explicit random_state argument (some TabPFN internals, older
      sklearn code paths).

    DOES NOT:
      save / restore the cuDNN deterministic flag. Set that at program
      start via set_seed(..., deterministic=True) instead.
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    try:
        import torch

        have_torch = True
        torch_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
    except ImportError:
        have_torch = False
        torch_state = None
        cuda_states = None

    set_seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        if have_torch:
            import torch

            torch.random.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)
