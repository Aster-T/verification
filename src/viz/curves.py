"""
Row-probe curves for one dataset.

Reads `results/<dataset>/row/metrics.jsonl`, groups by (model, mode),
aggregates across seeds, and writes FIVE PNGs into `out_dir`:

  row_curve.png                           all 4 lines combined (shared y-axis)
  row_curve_mlr_exact.png                 single-line, auto-scaled
  row_curve_mlr_jitter.png
  row_curve_tabpfn_exact.png
  row_curve_tabpfn_jitter.png

The combined plot keeps all lines on one set of axes so cross-model
comparison at a glance is possible (at the cost of the bigger model
dominating the y-range). The 4 per-(model, mode) plots each use their
own y-scale, so trends inside each curve are visible even when absolute
nRMSE differs by orders of magnitude between models.

nRMSE = RMSE / std(y_query). Dimensionless, cross-dataset comparable.
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
    ("mlr", "jitter"):   "#3a8dcc",
    ("tabpfn", "exact"): "#d62728",
    ("tabpfn", "jitter"):"#e65757",
}
_STYLES = {
    ("mlr", "exact"):    "--",
    ("mlr", "jitter"):   "-",
    ("tabpfn", "exact"): "--",
    ("tabpfn", "jitter"):"-",
}
# Marker also varies exact ↔ jitter so the two are distinguishable in the
# legend even when the dash-vs-solid difference is too small to notice.
_MARKERS = {
    ("mlr", "exact"):    "o",
    ("mlr", "jitter"):   "s",
    ("tabpfn", "exact"): "o",
    ("tabpfn", "jitter"):"s",
}
_PANEL_ORDER = [
    ("mlr", "exact"),
    ("mlr", "jitter"),
    ("tabpfn", "exact"),
    ("tabpfn", "jitter"),
]
# For the combined 4-line plot: vertical (points) offset per line, so the
# labels zigzag across 4 'lanes' instead of all piling on one line.
_COMBINED_OFFSET = {
    ("mlr", "exact"):    ("up",   14),
    ("mlr", "jitter"):   ("down", 14),
    ("tabpfn", "exact"): ("up",   42),
    ("tabpfn", "jitter"):("down", 42),
}


def _nrmse_series(recs: list[dict]) -> list[float]:
    """Pick nrmse per record (legacy r²-only fallback), rounded to 3
    decimals to match the label display precision. Floating-point noise
    below 1e-3 is suppressed so numerically identical lines (e.g.
    MLR/exact, OLS-invariant-to-duplication) actually render as flat."""
    out: list[float] = []
    for r in recs:
        if r.get("nrmse") is not None:
            v = round(float(r["nrmse"]), 3)
        elif "nrmse" in r:
            v = float("nan")
        else:
            v = round(float(np.sqrt(max(0.0, 1.0 - r["r2"]))), 3)
        out.append(v)
    return out


def _pivot(by_combo, skipped):
    """Reshape raw records to per-(model,mode) [(n_ctx, [ys...])]+skip_xs."""
    series: dict[tuple[str, str], list[tuple[int, list[float]]]] = {}
    skips: dict[tuple[str, str], list[int]] = {}
    for (model, mode, _k), recs in by_combo.items():
        n_ctx = int(recs[0]["n_ctx"])
        ys = _nrmse_series(recs)
        series.setdefault((model, mode), []).append((n_ctx, ys))
    for (model, mode, _k), recs in skipped.items():
        skips.setdefault((model, mode), []).append(int(recs[0]["n_ctx"]))
    for key in series:
        series[key].sort(key=lambda t: t[0])
    return series, skips


def _apply_ylim_with_floor(ax, means, *, pad_lo=0.15, pad_hi=0.18):
    """Expand y-limits around the data. When all points are equal (post
    rounding to 3 decimals), matplotlib's auto-scale gives span=0; fall back
    to a ±5%|mean| window so the flat line is visible.
    """
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    y_mid = float(np.nanmean(means)) if len(means) else 0.0
    if span <= 0 or span / max(abs(y_mid), 1.0) < 1e-6:
        half = max(abs(y_mid) * 0.05, 0.05)
        ax.set_ylim(y_mid - half, y_mid + half)
    else:
        ax.set_ylim(y0 - span * pad_lo, y1 + span * pad_hi)
    # Disable scientific "offset" notation that would otherwise print
    # labels like `1e-9+1.034e3` when the span is tiny.
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")


def _style_axes(ax, color=None, *, color_title=True):
    ax.grid(True, which="both", alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _set_rc():
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
        "axes.edgecolor": "#555555",
        "axes.linewidth": 0.8,
        "grid.color": "#cccccc",
        "grid.linestyle": ":",
        "grid.linewidth": 0.6,
    })


# --------------------------------------------------------------------------
# Combined plot (all 4 lines on one axis)
# --------------------------------------------------------------------------

def _plot_combined(ax, series, skips):
    """All 4 lines on one axes, shared y-scale. Writes per-point labels
    with zig-zag per-line offsets."""
    any_drawn = False
    for (model, mode), lanes in series.items():
        xs = np.array([t[0] for t in lanes])
        means = np.array([np.nanmean(t[1]) for t in lanes])
        stds = np.array([np.nanstd(t[1]) if len(t[1]) > 1 else 0.0
                         for t in lanes])
        color = _COLORS.get((model, mode), "gray")
        style = _STYLES.get((model, mode), "-")
        marker = _MARKERS.get((model, mode), "o")
        ax.plot(xs, means, style, color=color, linewidth=2,
                marker=marker, markersize=6,
                label=f"{model.upper()}/{mode}")
        ax.fill_between(xs, means - stds, means + stds,
                        color=color, alpha=0.15)
        direction, base = _COMBINED_OFFSET.get((model, mode), ("up", 14))
        sign = +1 if direction == "up" else -1
        for i, (x, ym) in enumerate(zip(xs, means)):
            if not np.isfinite(ym):
                continue
            dy = sign * (base + (i % 2) * 14)
            ax.annotate(
                f"({int(x)}, {ym:.3f})",
                xy=(x, ym), xytext=(0, dy), textcoords="offset points",
                ha="center", va="center", fontsize=7, color=color,
                arrowprops=dict(arrowstyle="-", color=color,
                                lw=0.4, alpha=0.5, shrinkA=0, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec=color, lw=0.5, alpha=0.9),
            )
        any_drawn = True

    # MLR/exact flat-line hint on the combined axis.
    if ("mlr", "exact") in series:
        lanes = series[("mlr", "exact")]
        x_last = lanes[-1][0]
        flat = float(np.nanmean(lanes[-1][1]))
        ax.axhline(flat, color=_COLORS[("mlr", "exact")],
                   alpha=0.35, linestyle=":")
        ax.annotate(
            "OLS invariant to uniform duplication",
            xy=(x_last, flat), xytext=(-6, -16),
            textcoords="offset points",
            ha="right", va="top", fontsize=8,
            color=_COLORS[("mlr", "exact")],
        )

    # Skipped markers on top.
    if any_drawn:
        all_means = np.concatenate([
            np.array([np.nanmean(t[1]) for t in lanes])
            for lanes in series.values()
        ])
        y_top = float(np.nanmax(all_means))
    else:
        y_top = 1.0
    for (model, mode), xs in skips.items():
        color = _COLORS.get((model, mode), "gray")
        for xk in xs:
            ax.plot([xk], [y_top * 1.05 if y_top > 0 else 1.0],
                    marker="s", mfc="none", mec=color, ms=10,
                    linestyle="none", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("n_ctx (context size per model invocation)")
    ax.set_ylabel("nRMSE = RMSE / std(y_query)")
    if any_drawn:
        all_means = np.concatenate([
            np.array([np.nanmean(t[1]) for t in lanes])
            for lanes in series.values()
        ])
        _apply_ylim_with_floor(ax, all_means)
    ax.legend(loc="center right", frameon=True, framealpha=0.92,
              edgecolor="#aaaaaa",
              handlelength=3.5,   # show enough line for dash/solid to be obvious
              handleheight=1.2,
              borderpad=0.7,
              labelspacing=0.7)
    _style_axes(ax)


# --------------------------------------------------------------------------
# Single-panel plot (one (model, mode), own y-scale)
# --------------------------------------------------------------------------

def _plot_single(ax, model, mode, series, skips):
    """Render one (model, mode) on its own axes with auto-scale."""
    color = _COLORS.get((model, mode), "gray")
    lanes = series.get((model, mode), [])
    skip_xs = skips.get((model, mode), [])

    if not lanes and not skip_xs:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="#888", fontsize=14)
        for sp in ax.spines.values():
            sp.set_color("#cccccc")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{model.upper()} / {mode}", fontsize=13,
                     color="#888", loc="left")
        return

    xs = np.array([t[0] for t in lanes]) if lanes else np.array([])
    means = np.array([np.nanmean(t[1]) for t in lanes]) if lanes else np.array([])
    stds = (np.array([np.nanstd(t[1]) if len(t[1]) > 1 else 0.0
                      for t in lanes]) if lanes else np.array([]))

    if lanes:
        ax.plot(xs, means, "-", color=color, linewidth=2,
                marker="o", markersize=6)
        ax.fill_between(xs, means - stds, means + stds,
                        color=color, alpha=0.18)
        # Zig-zag labels (only one line, simple even/odd alternation).
        for i, (x, ym) in enumerate(zip(xs, means)):
            if not np.isfinite(ym):
                continue
            dy = 16 if (i % 2 == 0) else -20
            ax.annotate(
                f"({int(x)}, {ym:.3f})",
                xy=(x, ym), xytext=(0, dy), textcoords="offset points",
                ha="center", va="center", fontsize=9, color=color,
                arrowprops=dict(arrowstyle="-", color=color,
                                lw=0.5, alpha=0.55, shrinkA=0, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.3",
                          fc="white", ec=color, lw=0.7, alpha=0.94),
            )
        if (model, mode) == ("mlr", "exact"):
            flat = float(means[-1])
            ax.axhline(flat, color=color, alpha=0.35, linestyle=":")
            ax.annotate(
                "OLS invariant to uniform duplication",
                xy=(xs[-1], flat), xytext=(-6, -22),
                textcoords="offset points",
                ha="right", va="top", fontsize=9, color=color,
            )
        _apply_ylim_with_floor(ax, means)

    if skip_xs:
        y_ref = float(np.nanmean(means)) if lanes else 1.0
        for xk in skip_xs:
            ax.plot([xk], [y_ref * 1.05 if y_ref > 0 else 1.0],
                    marker="s", mfc="none", mec=color, ms=10,
                    linestyle="none", zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("n_ctx (context size per model invocation)")
    ax.set_ylabel("nRMSE = RMSE / std(y_query)")
    ax.set_title(f"{model.upper()} / {mode}", fontsize=13, loc="left",
                 color=color, fontweight="bold")
    _style_axes(ax)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def plot_row_curves(
    dataset: str,
    jsonl_path: Path,
    out_dir: Path,
) -> Path:
    """Write 5 PNGs into `out_dir`:

      row_curve.png                    all 4 lines combined
      row_curve_<model>_<mode>.png     4 per-combo plots, each auto-scaled

    Returns the combined plot path.
    """
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

    series, skips = _pivot(by_combo, skipped)
    _set_rc()
    dpi = CONFIG["viz"]["dpi"]

    # 1) combined
    figsize = CONFIG["viz"]["figsize_curve"]
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _plot_combined(ax, series, skips)
    ax.set_title(
        f"{dataset}: nRMSE vs context size — all (model, mode) combined\n"
        f"(mean ± std over seeds;  n_ctx = k × n_tr for proportional, "
        f"k × (N−1) for loo)",
        fontsize=12,
    )
    fig.tight_layout()
    combined_path = out_dir / "row_curve.png"
    fig.savefig(combined_path)
    plt.close(fig)

    # 2) four individual panels, each in its own PNG
    single_figsize = (figsize[0] * 0.75, figsize[1] * 0.75)
    for model, mode in _PANEL_ORDER:
        fig, ax = plt.subplots(figsize=single_figsize, dpi=dpi)
        _plot_single(ax, model, mode, series, skips)
        fig.suptitle(
            f"{dataset}  —  {model.upper()} / {mode}\n"
            f"(n_ctx = k × n_tr for proportional, k × (N−1) for loo)",
            fontsize=11, y=0.98,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / f"row_curve_{model}_{mode}.png")
        plt.close(fig)

    return combined_path
