"""Per-dataset feature distribution box plots.

Loads every requested dataset via `load_dataset_full(name, seed)`, then
renders ONE PNG per dataset showing each feature's distribution as a
box plot. The target column gets its own subplot at the end. Numeric
nominal columns (integer category codes) are still drawn as boxes but
tagged so you can spot them at a glance. String columns are drawn as
horizontal bar charts of their top-N value counts.

Use this to scan column scales / outliers / heavy-tail targets before
deciding on a jitter strategy or normalization choice.

Examples:

  # one named dataset:
  python scripts/plot_feature_distributions.py --dataset diabetes

  # everything registered locally (datasets/<name>/meta.json):
  python scripts/plot_feature_distributions.py --local-all

  # OpenML presets:
  python scripts/plot_feature_distributions.py --openml-all

  # custom output folder:
  python scripts/plot_feature_distributions.py --local-all --openml-all \
      --out results/feature_distributions
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "third-party" / "tabpfn" / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.configs import CONFIG, add_openml_cli_args, resolve_openml_args  # noqa: E402
from src.data.loaders import load_dataset_full  # noqa: E402

logger = logging.getLogger(__name__)


def _is_numeric_column(col: np.ndarray) -> bool:
    """Decide whether a column from an object ndarray is numeric. Catches
    columns that the loader passed through as object dtype but are actually
    floats."""
    if col.dtype.kind in ("i", "u", "f"):
        return True
    # Object dtype: scan a few values.
    if col.dtype == object:
        for v in col[: min(len(col), 50)]:
            if isinstance(v, str):
                return False
        return True
    return False


def _column_to_float(col: np.ndarray) -> np.ndarray:
    """Coerce a numeric-looking column to float64, dropping NaNs/Infs at
    plot time (boxplot handles those, but we want the descriptive stats
    to be clean)."""
    arr = np.asarray(col, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _set_rc() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.7,
        "grid.color": "#cccccc",
        "grid.linestyle": ":",
        "grid.linewidth": 0.5,
        # CJK-capable fallbacks first so Chinese feature names (e.g. 船号)
        # don't render as missing-glyph boxes. matplotlib walks this list
        # and uses the first font that has the glyph; DejaVu Sans (the
        # default) stays at the end for ASCII coverage.
        "font.sans-serif": [
            "Microsoft YaHei", "PingFang SC",
            "Noto Sans CJK SC", "WenQuanYi Zen Hei",
            "SimHei", "Arial Unicode MS",
            "DejaVu Sans",
        ],
        # Prevent the unicode minus glyph from rendering as a box on fonts
        # that lack U+2212.
        "axes.unicode_minus": False,
    })


def _plot_numeric(ax, col: np.ndarray, title: str, *, color: str) -> None:
    """Single-column box plot with descriptive-stats footer."""
    vals = _column_to_float(col)
    if vals.size == 0:
        ax.text(0.5, 0.5, "no finite values", ha="center", va="center",
                transform=ax.transAxes, color="#888")
        ax.set_title(title, fontsize=9, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        return
    bp = ax.boxplot(
        vals,
        vert=True,
        widths=0.5,
        showmeans=True,
        meanline=False,
        meanprops=dict(marker="D", markerfacecolor=color,
                       markeredgecolor="white", markersize=5),
        medianprops=dict(color=color, linewidth=1.6),
        boxprops=dict(color="#333", linewidth=0.8),
        whiskerprops=dict(color="#666", linewidth=0.8),
        capprops=dict(color="#666", linewidth=0.8),
        flierprops=dict(marker="o", markersize=2.5,
                        markerfacecolor=color, markeredgecolor="none",
                        alpha=0.4),
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.18)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xticks([])
    ax.grid(True, axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # Footer: n / mean ± std / [min, max]
    footer = (
        f"n={vals.size}  "
        f"μ={vals.mean():.3g} ± {vals.std(ddof=0):.3g}\n"
        f"[{vals.min():.3g},  {vals.max():.3g}]"
    )
    ax.text(0.5, -0.06, footer, transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color="#555")


def _plot_categorical(
    ax, col: np.ndarray, title: str, *, top_n: int, color: str,
) -> None:
    """Horizontal bar chart of the top-N most frequent string values."""
    s = pd.Series([str(v) for v in col])
    counts = s.value_counts()
    n_unique = int(counts.size)
    top = counts.head(top_n)
    truncated = n_unique > top_n

    y_pos = np.arange(top.size)
    ax.barh(y_pos, top.values, color=color, alpha=0.55, edgecolor="#333",
            linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [(label[:18] + "…") if len(label) > 18 else label for label in top.index],
        fontsize=7,
    )
    ax.invert_yaxis()
    ax.set_title(title, fontsize=9, loc="left")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, axis="x", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    n_total = int(s.size)
    note = (
        f"n={n_total}  unique={n_unique}"
        + (f"  (showing top {top_n})" if truncated else "")
    )
    ax.text(0.5, -0.10, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=7, color="#555")


def plot_dataset(
    name: str,
    out_dir: Path,
    seed: int = 0,
    cols_per_row: int = 4,
    top_categorical: int = 10,
    include_target: bool = True,
) -> Path:
    """Render one PNG of feature box plots for `name` into `out_dir`.

    Returns the saved path.
    """
    X, y, feature_names, is_nominal = load_dataset_full(name, seed)
    n_features = X.shape[1]
    panels: list[tuple[str, np.ndarray, str, bool]] = []
    # (title, raw column array, kind, is_nominal_flag)
    for j, fname in enumerate(feature_names):
        col = X[:, j] if X.ndim == 2 else X
        nominal_flag = bool(is_nominal[j]) if is_nominal else False
        if _is_numeric_column(col):
            tag = " [nominal]" if nominal_flag else ""
            panels.append((f"{fname}{tag}", col, "numeric", nominal_flag))
        else:
            panels.append((f"{fname} [text]", col, "categorical", nominal_flag))
    if include_target:
        panels.append(("target (y)", np.asarray(y), "numeric", False))

    n = len(panels)
    cols = max(1, min(cols_per_row, n))
    rows = math.ceil(n / cols)
    # Slightly bigger per-panel canvas than usual; box plots benefit from
    # a tall aspect ratio.
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 3.0, rows * 3.0),
        dpi=CONFIG["viz"].get("dpi", 150),
        squeeze=False,
    )

    feat_color = "#2563eb"   # blue
    target_color = "#d97706"  # amber, to set the target apart

    for idx, (title, col, kind, _nom) in enumerate(panels):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        is_target = (idx == n - 1) and include_target
        color = target_color if is_target else feat_color
        if kind == "numeric":
            _plot_numeric(ax, col, title, color=color)
        else:
            _plot_categorical(ax, col, title,
                              top_n=top_categorical, color=color)

    # Hide unused cells in the last row.
    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle(
        f"{name}  —  feature distributions  (n_rows={X.shape[0]}, "
        f"n_features={n_features})",
        fontsize=12, y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _discover_local_datasets() -> list[str]:
    """Every datasets/<name>/meta.json on disk gets a registration."""
    root = REPO / "datasets"
    if not root.is_dir():
        return []
    names: list[str] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if (child / "meta.json").is_file():
            names.append(child.name)
    return names


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--dataset", action="append", default=[],
                   help="Registered dataset name. Repeatable.")
    add_openml_cli_args(p)
    p.add_argument("--local-all", action="store_true",
                   help="Auto-discover every datasets/<name>/meta.json and "
                        "render its box plots.")
    p.add_argument("--out", type=Path,
                   default=REPO / "results" / "feature_distributions",
                   help="Output directory. Default: "
                        "results/feature_distributions/")
    p.add_argument("--seed", type=int, default=0,
                   help="Seed for stochastic loaders / OpenML subsamples.")
    p.add_argument("--cols-per-row", type=int, default=4,
                   help="Subplot grid width. Default: 4.")
    p.add_argument("--top-categorical", type=int, default=10,
                   help="Top-N value-count bars for string columns. "
                        "Default: 10.")
    p.add_argument("--no-target", action="store_true",
                   help="Skip the target (y) panel at the end.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    datasets: list[str] = list(args.dataset) + resolve_openml_args(args)
    if args.local_all:
        for name in _discover_local_datasets():
            if name not in datasets:
                datasets.append(name)
    if not datasets:
        p.error(
            "provide at least one of: --dataset / --openml-id / "
            "--openml-preset / --openml-all / --local-all"
        )

    _set_rc()
    out_dir = Path(args.out)
    n_done, n_err = 0, 0
    for name in datasets:
        try:
            path = plot_dataset(
                name,
                out_dir,
                seed=args.seed,
                cols_per_row=args.cols_per_row,
                top_categorical=args.top_categorical,
                include_target=not args.no_target,
            )
            logging.warning("wrote %s", path)
            n_done += 1
        except Exception as e:  # noqa: BLE001
            logging.warning("failed %s: %s", name, e)
            n_err += 1
    print(f"DONE — {n_done} dataset(s) plotted, {n_err} failed. "
          f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
