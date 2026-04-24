"""
Project-wide configuration.

SINGLE SOURCE OF TRUTH for:
  - filesystem paths (derived from __file__, no hardcoded absolutes)
  - dataset descriptors
  - model hyper-parameters shared across probes
  - per-probe knobs (column_probe, row_probe)
  - visualization defaults

Other modules MUST read from CONFIG rather than re-declaring defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
# src/configs.py lives in src/; repo root is one level up.
REPO_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "repo": REPO_ROOT,
    "src": REPO_ROOT / "src",
    "scripts": REPO_ROOT / "scripts",
    "prompts": REPO_ROOT / "prompts",
    "third_party": REPO_ROOT / "third-party",
    "tabpfn_src": REPO_ROOT / "third-party" / "tabpfn" / "src",
    "results": REPO_ROOT / "results",
    "results_column": REPO_ROOT / "results" / "column",
    "results_row": REPO_ROOT / "results" / "row",
    "results_report": REPO_ROOT / "results" / "report.html",
}


# -----------------------------------------------------------------------------
# Main config dict
# -----------------------------------------------------------------------------
CONFIG = {
    "paths": PATHS,
    # Default seed for anything that does not explicitly take a seed argument.
    # Per-experiment seeds (row_probe, etc.) override this.
    "seed_default": 0,
    # -------------------------------------------------------------------------
    # Datasets
    #
    # Each entry describes HOW to load one dataset; actual loading happens in
    # src/data/loaders.py by dispatching on the "loader" field. Four loader
    # types are supported, each with its own required/optional fields.
    #
    # Shared fields (all loaders):
    #   "loader"     str,  required. One of
    #                  "sklearn_builtin" | "make_regression"
    #                  | "openml_id"     | "local_csv"
    #   "test_size"  float, optional (default 0.2). Fraction of rows held out
    #                by load_dataset()'s train_test_split. Ignored by
    #                load_dataset_full() and by row_probe with split_mode=loo.
    #
    # Per-loader schema:
    #
    #   loader == "sklearn_builtin":
    #     "sklearn_name"  str,  required. "diabetes" | "california_housing".
    #     "subsample"     int|None, optional. Cap rows via seed-deterministic
    #                     random choice (None = keep all).
    #
    #   loader == "make_regression":
    #     "n_samples"     int,  required.
    #     "n_features"    int,  required.
    #     "n_informative" int,  required.
    #     "noise"         float, required. Std of gaussian noise added to y.
    #
    #   loader == "openml_id":
    #     "openml_id"     int,  required. OpenML data_id (regression tasks
    #                     only — classification targets are rejected).
    #     "subsample"     int|None, optional. Row cap (None = keep all).
    #     Normally registered via register_openml_dataset() or the
    #     --openml-preset / --openml-id CLI paths, not hand-written here.
    #
    #   loader == "local_csv":
    #     "path"          str,  required. Absolute path to datasets/<name>/
    #                     (must contain data.csv + meta.json).
    #     Normally registered automatically by _register_local_csv_if_present
    #     on first use of --dataset <name>; hand-written entries are optional.
    #
    # Example entries (for reference — leave the dict empty to let local_csv /
    # OpenML auto-registration populate it at runtime):
    #
    #   "diabetes": {
    #       "loader": "sklearn_builtin",
    #       "sklearn_name": "diabetes",
    #       "test_size": 0.2,
    #       "subsample": None,
    #   },
    #   "synth_linear": {
    #       "loader": "make_regression",
    #       "n_samples": 800, "n_features": 12,
    #       "n_informative": 6, "noise": 5.0,
    #       "test_size": 0.2,
    #   },
    # -------------------------------------------------------------------------
    "datasets": {},
    # -------------------------------------------------------------------------
    # TabPFN settings shared across probes
    #
    # n_estimators MUST stay at 1 for both column_probe (feature-order
    # alignment with MLR) and row_probe (avoid ensemble smoothing the
    # context-size effect we want to measure).
    # -------------------------------------------------------------------------
    "tabpfn": {
        "device": "cuda",
        "n_estimators": 1,
        "ignore_pretraining_limits": True,
    },
    # -------------------------------------------------------------------------
    # Column probe (03)
    # -------------------------------------------------------------------------
    "column_probe": {
        "attn_reduce": "mean",  # one of: "mean", "last", "per_layer"
        "also_save_per_layer": True,
    },
    # -------------------------------------------------------------------------
    # Row probe (04)
    # -------------------------------------------------------------------------
    "row_probe": {
        "k_list": [i for i in range(1, 11)],
        "modes": ["exact", "jitter"],
        "seeds": [42],
        "jitter_sigma": 1e-6,
        "metrics": ["r2", "rmse", "mae"],
    },
    # -------------------------------------------------------------------------
    # Visualization (05)
    # -------------------------------------------------------------------------
    "viz": {
        "heatmap_cmap": "RdBu_r",
        "figsize_heatmap_single": (6, 5),
        "figsize_heatmap_triptych": (15, 5),
        "figsize_curve": (14, 8),
        "dpi": 150,
    },
}


# -----------------------------------------------------------------------------
# Small helpers (keep them thin; add more only when you find yourself
# repeating the same lookup in 3+ places).
# -----------------------------------------------------------------------------
def _register_local_csv_if_present(name: str) -> bool:
    """
    If `datasets/<name>/meta.json` exists on disk, register it in CONFIG as a
    local_csv dataset and return True. Otherwise return False. Test_size
    defaults to 0.2; override by putting 'test_size' in meta.json.
    """
    ds_dir = REPO_ROOT / "datasets" / name
    meta_path = ds_dir / "meta.json"
    if not meta_path.exists():
        return False
    if name in CONFIG["datasets"]:
        return True
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    CONFIG["datasets"][name] = {
        "loader": "local_csv",
        "path": str(ds_dir),
        "test_size": float(meta.get("test_size", 0.2)),
    }
    return True


def get_dataset_cfg(name: str) -> dict:
    """
    Look up a dataset descriptor by name. On miss, attempt to auto-register a
    local_csv dataset from `datasets/<name>/`. Raises KeyError with the list
    of available names if that also fails.
    """
    if name not in CONFIG["datasets"]:
        _register_local_csv_if_present(name)
    if name not in CONFIG["datasets"]:
        raise KeyError(
            f"Unknown dataset '{name}'. Available: {sorted(CONFIG['datasets'].keys())}. "
            f"For a local dataset, put data.csv + meta.json under datasets/{name}/ "
            f"(generate meta.json via scripts/infer_meta.py)."
        )
    return CONFIG["datasets"][name]


def register_openml_dataset(
    openml_id: int,
    subsample: int | None = None,
    test_size: float = 0.2,
    name: str | None = None,
) -> str:
    """
    Register a fetch_openml-sourced dataset under a synthetic name and return
    that name.

    ARGS:
      openml_id: OpenML data_id.
      subsample: row cap; None (default) keeps all rows.
      test_size: split fraction for load_dataset().
      name:      custom key under CONFIG["datasets"]. Defaults to
                 f"openml_{openml_id}" when None. A custom name lets you
                 refer to the dataset in --dataset / results/<name>/ just
                 like a built-in.

    Idempotent: calling twice with the same (name) is a no-op and returns
    the existing name.
    """
    if name is None:
        name = f"openml_{int(openml_id)}"
    if name not in CONFIG["datasets"]:
        CONFIG["datasets"][name] = {
            "loader": "openml_id",
            "openml_id": int(openml_id),
            "subsample": subsample,
            "test_size": test_size,
        }
    return name


DEFAULT_OPENML_CONFIG = REPO_ROOT / "datasets" / "openml.json"


def load_openml_config(path: str | Path) -> dict[str, dict]:
    """
    Load OpenML preset file. Format: flat dict from registration-name to entry:

        {
          "bodyfat":      {"id": 560, "subsample": 2000, "description": "..."},
          "house_prices": {"id": 41021}
        }

    Keys starting with '_' (e.g. "_comment") are ignored so users can leave
    notes inline. Missing file raises FileNotFoundError with guidance.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"OpenML preset file not found: {p}. "
            f"Create it (see datasets/README.md) or pass --openml-config <path>."
        )
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{p} must be a JSON object mapping name -> entry dict")
    out: dict[str, dict] = {}
    for name, entry in raw.items():
        if name.startswith("_"):
            continue
        if not isinstance(entry, dict) or "id" not in entry:
            raise ValueError(
                f"{p}: preset {name!r} must be an object with at least an 'id' field"
            )
        out[name] = entry
    return out


def _register_from_preset(
    preset_name: str,
    entry: dict,
    cli_subsample: int,
) -> str:
    """Register one preset entry. Preset-level subsample overrides CLI."""
    oid = int(entry["id"])
    subsample = entry.get("subsample", cli_subsample)
    subsample = int(subsample) if subsample is not None else None
    return register_openml_dataset(oid, subsample=subsample, name=preset_name)


def _subsample_cli_type(s: str) -> int | None:
    """argparse type for --openml-subsample: non-positive int means 'no cap'."""
    v = int(s)
    return None if v <= 0 else v


def add_openml_cli_args(
    parser: argparse.ArgumentParser,
    default_subsample: int | None = None,
) -> None:
    """
    Install the standard OpenML CLI args on `parser`. Every CLI script that
    wants to consume --openml-id / --openml-preset / --openml-all should
    call this to stay consistent.
    """
    parser.add_argument(
        "--openml-id",
        action="append",
        default=[],
        metavar="ID[:NAME]",
        help="OpenML data_id, repeatable. '560' -> 'openml_560'; "
        "'560:bodyfat' uses custom name 'bodyfat'.",
    )
    parser.add_argument(
        "--openml-preset",
        action="append",
        default=[],
        metavar="NAME",
        help=f"Load a preset (by its key) from --openml-config (default: "
        f"{DEFAULT_OPENML_CONFIG.relative_to(REPO_ROOT)}). Repeatable.",
    )
    parser.add_argument(
        "--openml-all",
        action="store_true",
        help="Load every preset from --openml-config.",
    )
    parser.add_argument(
        "--openml-config",
        type=Path,
        default=DEFAULT_OPENML_CONFIG,
        help="Path to the OpenML preset JSON file.",
    )
    parser.add_argument(
        "--openml-subsample",
        type=_subsample_cli_type,
        default=default_subsample,
        help=(
            "Row cap per OpenML dataset when a preset does not set its own. "
            "Pass 0 or any non-positive int to mean 'no cap'. Default: no cap "
            "(may exceed TabPFN's 10k context — cap explicitly for TabPFN runs)."
        ),
    )


def resolve_openml_args(args: argparse.Namespace) -> list[str]:
    """
    Process --openml-id / --openml-preset / --openml-all from parsed args,
    register each dataset, and return the list of registered names in the
    order CLI specified them (--openml-id first, then presets).
    """
    names: list[str] = []
    for spec in getattr(args, "openml_id", []) or []:
        oid, custom_name = parse_openml_spec(spec)
        names.append(
            register_openml_dataset(
                oid,
                subsample=args.openml_subsample,
                name=custom_name,
            )
        )

    wants_presets = bool(getattr(args, "openml_all", False)) or bool(
        getattr(args, "openml_preset", []) or []
    )
    if not wants_presets:
        return names

    presets = load_openml_config(args.openml_config)
    if args.openml_all:
        preset_names = list(presets.keys())
    else:
        preset_names = list(args.openml_preset)
        missing = [n for n in preset_names if n not in presets]
        if missing:
            raise KeyError(
                f"unknown preset(s) {missing} in {args.openml_config}; "
                f"available: {sorted(presets.keys())}"
            )
    for pname in preset_names:
        names.append(
            _register_from_preset(pname, presets[pname], args.openml_subsample)
        )
    return names


def parse_openml_spec(spec: str) -> tuple[int, str | None]:
    """
    Parse a CLI --openml-id value into (id, optional_name).

      "560"          -> (560, None)     # auto-name as 'openml_560'
      "560:bodyfat"  -> (560, "bodyfat")

    Raises ValueError on unparseable input.
    """
    if ":" in spec:
        id_part, name_part = spec.split(":", 1)
        name = name_part.strip() or None
    else:
        id_part, name = spec, None
    id_part = id_part.strip()
    if not id_part.lstrip("-").isdigit():
        raise ValueError(
            f"--openml-id spec {spec!r}: expected integer or 'int:name', "
            f"got id part {id_part!r}"
        )
    return int(id_part), name


def sigma_tag(sigma: float) -> str:
    """
    Format a jitter_sigma float as a short, filesystem-safe tag in consistent
    scientific notation. Used by CLI scripts to build `sigma_<tag>/` result
    sub-directories so sweeping multiple sigmas doesn't cause collisions.

    Examples:
      sigma_tag(1e-2)   -> "1e-2"
      sigma_tag(1e-6)   -> "1e-6"
      sigma_tag(2.5e-4) -> "2.5e-4"
      sigma_tag(5e-2)   -> "5e-2"
      sigma_tag(0)      -> "0"
    """
    if sigma == 0:
        return "0"
    # Split via Python's %e formatter so mantissa/exp extraction is robust.
    mantissa_str, exp_str = f"{sigma:e}".split("e")
    m = float(mantissa_str)
    e = int(exp_str)
    if abs(m - round(m)) < 1e-9:
        m_out = str(int(round(m)))
    else:
        m_out = f"{m:g}".rstrip("0").rstrip(".")
    return f"{m_out}e{e}"


def get_device() -> str:
    """
    Return the tabpfn device from CONFIG, falling back to 'cpu' if cuda is
    not available. Import torch lazily so configs.py stays importable on
    CPU-only boxes without torch installed.
    """
    want = CONFIG["tabpfn"]["device"]
    if want == "cpu":
        return "cpu"
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
