"""
Row-probe curve for one dataset.

Reads a single jsonl (results/row/<dataset>.jsonl), groups by (model, mode),
aggregates across seeds, and plots one PNG:
  {dataset}_row_curve.png   (x = k, log scale; y = R², mean ± std over seeds)

Four lines total: MLR/exact, MLR/jitter, TabPFN/exact, TabPFN/jitter.
MLR/exact is annotated as the theoretical flat line.
Skipped records (e.g. TabPFN over-context) are drawn as hollow squares on the
corresponding k with no connecting line.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.configs import CONFIG  # noqa: E402
from src.utils.io import read_jsonl  # noqa: E402


_COLORS = {
    ("mlr", "exact"):    "#1f77b4",
    ("mlr", "jitter"):   "#aec7e8",
    ("tabpfn", "exact"): "#d62728",
    ("tabpfn", "jitter"):"#ff9896",
}
_STYLES = {
    ("mlr", "exact"):    "--",
    ("mlr", "jitter"):   "-",
    ("tabpfn", "exact"): "--",
    ("tabpfn", "jitter"):"-",
}


def plot_row_curves(
    dataset: str,
    jsonl_path: Path,
    out_dir: Path,
) -> Path:
    """Produce the row-probe R² curve. RETURNS the png path."""
    jsonl_path = Path(jsonl_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(jsonl_path)
    if not records:
        raise ValueError(f"no records in {jsonl_path}")

    by_combo: dict[tuple[str, str, int], list[dict]] = {}
    skipped: dict[tuple[str, str, int], list[dict]] = {}
    for rec in records:
        key = (rec["model"], rec["mode"], rec["k"])
        if rec.get("skipped"):
            skipped.setdefault(key, []).append(rec)
        else:
            by_combo.setdefault(key, []).append(rec)

    dpi = CONFIG["viz"]["dpi"]
    fig, ax = plt.subplots(figsize=CONFIG["viz"]["figsize_curve"], dpi=dpi)

    # Group by (model, mode) for plotting.
    lines = {}
    for (model, mode, k), recs in by_combo.items():
        lines.setdefault((model, mode), []).append((k, [r["r2"] for r in recs]))

    for (model, mode), kvs in lines.items():
        kvs.sort(key=lambda t: t[0])
        ks = np.array([t[0] for t in kvs])
        means = np.array([np.mean(t[1]) for t in kvs])
        stds = np.array([np.std(t[1]) if len(t[1]) > 1 else 0.0 for t in kvs])
        color = _COLORS.get((model, mode), "gray")
        style = _STYLES.get((model, mode), "-")
        label = f"{model.upper()}/{mode}"
        ax.plot(ks, means, style, color=color, label=label, linewidth=2)
        ax.fill_between(ks, means - stds, means + stds, color=color, alpha=0.15)

    # Skipped markers (no line).
    for (model, mode, k), recs in skipped.items():
        color = _COLORS.get((model, mode), "gray")
        ax.plot([k], [0], marker="s", mfc="none", mec=color, ms=10,
                linestyle="none", zorder=5)

    # Annotate MLR/exact flat-line expectation.
    if ("mlr", "exact") in lines:
        mlr_exact_means = np.array([np.mean(t[1]) for t in lines[("mlr", "exact")]])
        flat_val = float(mlr_exact_means[0])
        ax.axhline(flat_val, color=_COLORS[("mlr", "exact")], alpha=0.35, linestyle=":")
        ax.annotate(
            "OLS invariant to uniform duplication",
            xy=(lines[("mlr", "exact")][0][0], flat_val),
            xytext=(6, 6), textcoords="offset points", fontsize=8,
            color=_COLORS[("mlr", "exact")],
        )

    ax.set_xscale("log")
    ax.set_xlabel("k (context duplicated k× the original training set)")
    ax.set_ylabel("R² on fixed test set")
    ax.set_title(f"{dataset}: R² vs context size (mean ± std over seeds)")
    ax.legend(loc="best", fontsize=9, frameon=True)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"{dataset}_row_curve.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
