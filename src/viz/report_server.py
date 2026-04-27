"""Discovery + table-aggregation helpers backing scripts/serve_report.py.

Produces a manifest describing every (sigma, test_size, dataset, chart)
tuple that has artefacts on disk under `results/`, and computes the same
per-(model, mode, k) aggregated table that the legacy single-file HTML
report used to show — but on demand, one (sigma, dataset, test_size)
slice at a time, returned as plain JSON for the frontend to render.

LAYOUT (recap):
  results/sigma_<σ>/
    <ds>/
      row/metrics.jsonl                 ← LOO row probe
      column/{mlr,tabpfn}.npz
      viz/row_curve*.png, side_by_side.png, tabpfn_per_layer.png
    test_size_<ts>/<ds>/
      row/metrics.jsonl                 ← proportional row probe
      viz/row_curve*.png

In the manifest, LOO records use the literal string `"loo"` in the
test_size dimension; proportional records use `"0.1"`, `"0.2"`, etc.
That gives the frontend a single uniform dimension to filter on.
"""
from __future__ import annotations

from pathlib import Path
from statistics import mean, pstdev
from typing import Any

from src.utils.io import read_jsonl


# Chart types and their on-disk file names. Order is the display order.
_CHART_FILES: list[tuple[str, str, str]] = [
    # (id, label, filename)
    ("row_curve", "Row curve · combined", "row_curve.png"),
    ("row_curve_mlr", "Row curve · MLR (exact + jitter)", "row_curve_mlr.png"),
    ("row_curve_tabpfn", "Row curve · TabPFN (exact + jitter)", "row_curve_tabpfn.png"),
    ("row_curve_mlr_exact", "Row curve · MLR/exact", "row_curve_mlr_exact.png"),
    ("row_curve_mlr_jitter", "Row curve · MLR/jitter", "row_curve_mlr_jitter.png"),
    ("row_curve_tabpfn_exact", "Row curve · TabPFN/exact", "row_curve_tabpfn_exact.png"),
    ("row_curve_tabpfn_jitter", "Row curve · TabPFN/jitter", "row_curve_tabpfn_jitter.png"),
    ("side_by_side", "Column probe · side by side", "side_by_side.png"),
    ("tabpfn_per_layer", "Column probe · TabPFN per-layer", "tabpfn_per_layer.png"),
]
_CHART_LABELS = {cid: label for (cid, label, _) in _CHART_FILES}
_CHART_FILENAMES = {cid: fname for (cid, _, fname) in _CHART_FILES}


def _parse_sigma_dir(name: str) -> str | None:
    return name[len("sigma_"):] if name.startswith("sigma_") else None


def _parse_ts_dir(name: str) -> str | None:
    return name[len("test_size_"):] if name.startswith("test_size_") else None


def _scan_viz(viz_dir: Path) -> list[str]:
    """Return chart-type ids (`_CHART_FILES` keys) whose PNG exists in
    `viz_dir`."""
    if not viz_dir.is_dir():
        return []
    return [cid for cid, _, fname in _CHART_FILES if (viz_dir / fname).exists()]


def build_manifest(results_root: Path) -> dict[str, Any]:
    """Walk `results_root` and emit the JSON-shaped manifest the
    frontend consumes.

    The returned dict has these keys:
      sigmas:      sorted list of σ tags (e.g. ["1e-2", "1e-3", ...]).
      test_sizes:  sorted list of test_size labels with "loo" first if
                   present, then "0.1", "0.2", ... in numeric order.
      datasets:    sorted list of dataset names that appear anywhere.
      chart_types: list of {id, label} for charts present in any item.
      images:      list of {sigma, test_size, dataset, chart, label, url}.
      tables:      list of {sigma, test_size, dataset, label, jsonl}.
                   `jsonl` is a relative-to-results_root path the server
                   passes to `aggregate_table` on demand.
    """
    results_root = Path(results_root)
    sigmas: set[str] = set()
    test_sizes: set[str] = set()
    datasets: set[str] = set()
    chart_ids: set[str] = set()
    images: list[dict[str, str]] = []
    tables: list[dict[str, str]] = []

    if not results_root.exists():
        return _empty_manifest()

    for sigma_dir in sorted(results_root.iterdir()):
        sigma = _parse_sigma_dir(sigma_dir.name) if sigma_dir.is_dir() else None
        if sigma is None:
            continue
        sigmas.add(sigma)

        # 1) sigma-level (LOO + column probe). Each direct subdir that
        #    isn't `test_size_*` is a dataset directory.
        for ds_dir in sorted(sigma_dir.iterdir()):
            if not ds_dir.is_dir() or ds_dir.name.startswith("test_size_"):
                continue
            ds = ds_dir.name
            viz_dir = ds_dir / "viz"
            row_jsonl = ds_dir / "row" / "metrics.jsonl"

            charts_here = _scan_viz(viz_dir)
            if charts_here or row_jsonl.exists():
                datasets.add(ds)
                test_sizes.add("loo")

            for cid in charts_here:
                rel = (viz_dir / _CHART_FILENAMES[cid]).relative_to(results_root)
                images.append({
                    "sigma": sigma,
                    "test_size": "loo",
                    "dataset": ds,
                    "chart": cid,
                    "label": _CHART_LABELS[cid],
                    "url": f"/results/{rel.as_posix()}",
                })
                chart_ids.add(cid)

            if row_jsonl.exists():
                rel = row_jsonl.relative_to(results_root)
                tables.append({
                    "sigma": sigma,
                    "test_size": "loo",
                    "dataset": ds,
                    "label": f"{ds} · σ={sigma} · LOO",
                    "jsonl": rel.as_posix(),
                })

        # 2) proportional: test_size_<ts>/<ds>/...
        for ts_dir in sorted(sigma_dir.iterdir()):
            ts = _parse_ts_dir(ts_dir.name) if ts_dir.is_dir() else None
            if ts is None:
                continue
            for ds_dir in sorted(ts_dir.iterdir()):
                if not ds_dir.is_dir():
                    continue
                ds = ds_dir.name
                viz_dir = ds_dir / "viz"
                row_jsonl = ds_dir / "row" / "metrics.jsonl"

                charts_here = _scan_viz(viz_dir)
                if charts_here or row_jsonl.exists():
                    datasets.add(ds)
                    test_sizes.add(ts)

                for cid in charts_here:
                    # Column-probe charts only live at the sigma level
                    # (test_size-independent), so silently ignore stale
                    # copies under test_size_*/.
                    if cid in ("side_by_side", "tabpfn_per_layer"):
                        continue
                    rel = (viz_dir / _CHART_FILENAMES[cid]).relative_to(results_root)
                    images.append({
                        "sigma": sigma,
                        "test_size": ts,
                        "dataset": ds,
                        "chart": cid,
                        "label": _CHART_LABELS[cid],
                        "url": f"/results/{rel.as_posix()}",
                    })
                    chart_ids.add(cid)

                if row_jsonl.exists():
                    rel = row_jsonl.relative_to(results_root)
                    tables.append({
                        "sigma": sigma,
                        "test_size": ts,
                        "dataset": ds,
                        "label": f"{ds} · σ={sigma} · test_size={ts}",
                        "jsonl": rel.as_posix(),
                    })

    return {
        "sigmas": sorted(sigmas, key=_sigma_sort_key),
        "test_sizes": _order_test_sizes(test_sizes),
        "datasets": sorted(datasets),
        "chart_types": [
            {"id": cid, "label": _CHART_LABELS[cid]}
            for cid, _, _ in _CHART_FILES if cid in chart_ids
        ],
        "images": images,
        "tables": tables,
    }


def _empty_manifest() -> dict[str, Any]:
    return {
        "sigmas": [], "test_sizes": [], "datasets": [],
        "chart_types": [], "images": [], "tables": [],
    }


def _sigma_sort_key(s: str) -> float:
    # "1e-2" < "1e-3" numerically: smaller exponents = larger value, but
    # we want the natural reading order (1e-2, 1e-3, ..., 1e-6). So sort
    # by descending magnitude of σ.
    try:
        return -float(s)
    except ValueError:
        return 0.0


def _order_test_sizes(values: set[str]) -> list[str]:
    """LOO first (when present), then numeric test_sizes in ascending
    order. Non-numeric oddities go last to avoid tripping the frontend."""
    out: list[str] = []
    if "loo" in values:
        out.append("loo")
    numeric: list[tuple[float, str]] = []
    weird: list[str] = []
    for v in values:
        if v == "loo":
            continue
        try:
            numeric.append((float(v), v))
        except ValueError:
            weird.append(v)
    out.extend(v for _, v in sorted(numeric))
    out.extend(sorted(weird))
    return out


# --------------------------------------------------------------------------
# Aggregate table API (computed on demand for /table)
# --------------------------------------------------------------------------

def aggregate_table(jsonl_path: Path) -> dict[str, Any]:
    """Aggregate a `metrics.jsonl` into per-(split_mode, model, mode, k)
    summary rows. Mirrors the legacy report's table; returns plain dicts
    so the frontend can render them however it wants.

    Each row has:
      split, model, mode, k, n_ctx, n_query, n_folds, n_features,
      nrmse, r2, rmse, mae, mape,
      mape_tanker (only when present in any record),
      skipped_count.
    Numeric metrics are objects {mean, std, n}. `n=0` means no records
    contributed (rendered as '—').
    """
    records = read_jsonl(jsonl_path) if jsonl_path.exists() else []
    if not records:
        return {"rows": [], "has_tanker": False}

    has_tanker = any("mape_tanker" in r for r in records)

    by_key: dict[tuple, dict[str, Any]] = {}
    for r in records:
        split = r.get("split_mode", "proportional")
        key = (split, r["model"], r["mode"], r["k"])
        bucket = by_key.setdefault(key, {
            "split": split, "model": r["model"], "mode": r["mode"],
            "k": r["k"],
            "n_ctx": r.get("n_ctx"),
            "n_query": r.get("n_query", r.get("n_te")),
            "n_folds": r.get("n_folds", 1),
            "n_features": r.get("n_features"),
            "nrmse": [], "r2": [], "rmse": [], "mae": [], "mape": [],
            "mape_tanker": [],
            "skipped_count": 0,
        })
        if r.get("skipped"):
            bucket["skipped_count"] += 1
            continue
        if r.get("nrmse") is not None:
            bucket["nrmse"].append(r["nrmse"])
        bucket["r2"].append(r["r2"])
        bucket["rmse"].append(r["rmse"])
        bucket["mae"].append(r["mae"])
        if r.get("mape") is not None:
            bucket["mape"].append(r["mape"])
        if r.get("mape_tanker") is not None:
            bucket["mape_tanker"].append(r["mape_tanker"])

    def _stat(xs: list[float]) -> dict[str, float | int | None]:
        if not xs:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": float(mean(xs)),
            "std": float(pstdev(xs)) if len(xs) > 1 else 0.0,
            "n": len(xs),
        }

    rows: list[dict[str, Any]] = []
    for key in sorted(by_key.keys(), key=lambda t: (t[0], t[1], t[2], t[3])):
        b = by_key[key]
        row = {
            "split": b["split"], "model": b["model"], "mode": b["mode"],
            "k": b["k"],
            "n_ctx": b["n_ctx"],
            "n_query": b["n_query"],
            "n_folds": b["n_folds"],
            "n_features": b["n_features"],
            "nrmse": _stat(b["nrmse"]),
            "r2": _stat(b["r2"]),
            "rmse": _stat(b["rmse"]),
            "mae": _stat(b["mae"]),
            "mape": _stat(b["mape"]),
            "skipped_count": b["skipped_count"],
        }
        if has_tanker:
            row["mape_tanker"] = _stat(b["mape_tanker"])
        rows.append(row)

    return {"rows": rows, "has_tanker": has_tanker}


# --------------------------------------------------------------------------
# Macro summary (computed on demand for /macro)
# --------------------------------------------------------------------------

def aggregate_macro(jsonl_paths: list[Path]) -> dict[str, Any]:
    """Pool every non-skipped record across all `jsonl_paths` into a
    per-model rollup. Used to answer 'how does the dataset behave overall
    under the current filter?' — e.g. average MAPE_tanker for ship-all
    across every (k, mode, seed, test_size) the user is currently looking
    at.

    Each record in a metrics.jsonl is itself an aggregate over seeds for
    one (model, k, mode) experiment, so we treat records as the unit of
    averaging here (no extra weighting).

    Returns:
      {
        "n_records":  total non-skipped records considered,
        "n_skipped":  total skipped records,
        "per_model":  {"mlr": {n, nrmse, r2, rmse, mae, mape, mape_tanker?}, ...},
        "has_tanker": bool — true if any record carried mape_tanker,
      }

    Each metric value is {"mean", "std", "n"}; n may differ from the
    model's overall n when some records had a null for that metric (e.g.
    mape is null whenever every y_true == 0 in that experiment).
    """
    per_model: dict[str, dict[str, list[float]]] = {}
    n_records = 0
    n_skipped = 0
    has_tanker = False
    seen_models: set[str] = set()

    for path in jsonl_paths:
        if not path.exists():
            continue
        for r in read_jsonl(path):
            if r.get("skipped"):
                n_skipped += 1
                continue
            n_records += 1
            model = str(r.get("model", "?"))
            seen_models.add(model)
            bucket = per_model.setdefault(model, {
                "nrmse": [], "r2": [], "rmse": [], "mae": [],
                "mape": [], "mape_tanker": [],
            })
            if r.get("nrmse") is not None:
                bucket["nrmse"].append(float(r["nrmse"]))
            if r.get("r2") is not None and r["r2"] == r["r2"]:  # NaN-safe
                bucket["r2"].append(float(r["r2"]))
            if r.get("rmse") is not None:
                bucket["rmse"].append(float(r["rmse"]))
            if r.get("mae") is not None:
                bucket["mae"].append(float(r["mae"]))
            if r.get("mape") is not None:
                bucket["mape"].append(float(r["mape"]))
            if "mape_tanker" in r:
                has_tanker = True
                if r.get("mape_tanker") is not None:
                    bucket["mape_tanker"].append(float(r["mape_tanker"]))

    def _stat(xs: list[float]) -> dict[str, float | int | None]:
        if not xs:
            return {"mean": None, "std": None, "n": 0}
        return {
            "mean": float(mean(xs)),
            "std": float(pstdev(xs)) if len(xs) > 1 else 0.0,
            "n": len(xs),
        }

    out_per_model: dict[str, dict[str, Any]] = {}
    for model in sorted(seen_models):
        b = per_model[model]
        # m_total = best estimate of how many records contributed to this
        # model. Different metrics can have different n if some records
        # had null values (e.g. mape is null when every y_true is zero),
        # so we report the max — every record contributes to at least one.
        m_total = max(
            len(b["nrmse"]), len(b["r2"]), len(b["rmse"]),
            len(b["mae"]), len(b["mape"]),
        )
        entry: dict[str, Any] = {
            "n": m_total,
            "nrmse": _stat(b["nrmse"]),
            "r2": _stat(b["r2"]),
            "rmse": _stat(b["rmse"]),
            "mae": _stat(b["mae"]),
            "mape": _stat(b["mape"]),
        }
        if has_tanker:
            entry["mape_tanker"] = _stat(b["mape_tanker"])
        out_per_model[model] = entry

    return {
        "n_records": n_records,
        "n_skipped": n_skipped,
        "per_model": out_per_model,
        "has_tanker": has_tanker,
    }
