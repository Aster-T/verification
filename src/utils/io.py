"""
Lightweight I/O helpers. No project knowledge leaks here.

Scope:
  - jsonl read / write / iter / existing-keys (for resume)
  - npz save / load (with string-list support for feature_names)
  - json read / write (for meta files)
  - ensure_dir
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np


# -----------------------------------------------------------------------------
# Path
# -----------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Create the directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# -----------------------------------------------------------------------------
# JSON encoder
# -----------------------------------------------------------------------------
def _json_default(obj: Any):
    """
    Fallback for json.dumps.

    Handles: numpy scalars/arrays, pathlib.Path, NaN/Inf (-> None).
    Raises TypeError for anything else so unknown types surface loudly.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        # json stdlib writes NaN/Inf as bare tokens, which is not valid JSON
        # and breaks pandas.read_json. Emit null instead.
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Cannot serialize object of type {type(obj).__name__}")


# -----------------------------------------------------------------------------
# JSONL
# -----------------------------------------------------------------------------
def write_jsonl(
    path: str | Path,
    records: Iterable[dict],
    append: bool = True,
) -> None:
    """
    Write dicts one-per-line to `path`.

    ARGS:
      path: target file (parents auto-created).
      records: iterable of dicts. Consumed lazily.
      append: True -> 'a' mode; False -> 'w' (truncate).

    NOTES:
      All records pass through _json_default so numpy scalars / arrays / Path
      work out of the box. NaN/Inf become null.
    """
    p = Path(path)
    ensure_dir(p.parent)
    mode = "a" if append else "w"
    with p.open(mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, default=_json_default, ensure_ascii=False))
            f.write("\n")


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    """Yield each JSON object from a jsonl file. Skips blank lines."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict]:
    """Load all records from a jsonl file into a list."""
    return list(iter_jsonl(path))


def existing_keys_jsonl(
    path: str | Path,
    key_fn: Callable[[dict], tuple],
) -> set[tuple]:
    """
    Scan an existing jsonl and return the set of keys already present.

    Used by row_probe for checkpoint/resume:

        done = existing_keys_jsonl(
            jsonl_path,
            lambda r: (r["model"], r["k"], r["mode"], r["seed"]),
        )
        for combo in all_combos:
            if combo in done:
                continue
            ...

    Returns empty set if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        return set()
    return {key_fn(rec) for rec in iter_jsonl(p)}


# -----------------------------------------------------------------------------
# NPZ
# -----------------------------------------------------------------------------
def save_npz(path: str | Path, **arrays: Any) -> Path:
    """
    Save named arrays to a .npz file. Parent dir is auto-created.

    Special-cases a list of strings (e.g. feature_names) by storing it as
    dtype=object, because np.savez otherwise chokes on str lists of uneven
    length.
    """
    p = Path(path)
    ensure_dir(p.parent)
    payload: dict[str, np.ndarray] = {}
    for key, val in arrays.items():
        if isinstance(val, (list, tuple)) and all(isinstance(x, str) for x in val):
            payload[key] = np.asarray(val, dtype=object)
        else:
            payload[key] = np.asarray(val)
    np.savez(p, **payload)
    return p


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load a .npz file into a plain dict so callers don't have to manage the
    NpzFile context manager.
    """
    with np.load(Path(path), allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


# -----------------------------------------------------------------------------
# JSON (for meta files)
# -----------------------------------------------------------------------------
def dump_json(path: str | Path, obj: dict, indent: int = 2) -> Path:
    """Write a dict to a JSON file (pretty-printed by default)."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=_json_default, ensure_ascii=False)
    return p


def load_json(path: str | Path) -> dict:
    """Read a JSON file into a dict."""
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
