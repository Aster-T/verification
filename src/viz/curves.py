"""
Row-probe curves for one dataset.

Reads `results/<dataset>/row/metrics.jsonl`, groups by (model, mode),
aggregates across seeds, and writes SEVEN PNGs into `out_dir`:

  row_curve.png                           all 4 lines combined (shared y-axis)
  row_curve_mlr_exact.png                 single-line, auto-scaled
  row_curve_mlr_jitter.png
  row_curve_tabpfn_exact.png
  row_curve_tabpfn_jitter.png
  row_curve_mlr.png                       MLR exact + jitter overlaid
  row_curve_tabpfn.png                    TabPFN exact + jitter overlaid

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


# Four fully distinct colors (Tableau-style) — picked so that overlapping
# lines remain distinguishable even when markers, hues, or line styles
# coincide. No two series share a hue.
_COLORS = {
    ("mlr", "exact"):    "#1f77b4",  # blue
    ("mlr", "jitter"):   "#ff7f0e",  # orange
    ("tabpfn", "exact"): "#2ca02c",  # green
    ("tabpfn", "jitter"):"#d62728",  # red
}
# Four distinct line styles + four distinct markers. Either alone is enough
# to identify a series; together they're robust to b/w printing or color
# blindness.
_STYLES = {
    ("mlr", "exact"):    "--",
    ("mlr", "jitter"):   "-",
    ("tabpfn", "exact"): "-.",
    ("tabpfn", "jitter"):":",
}
_MARKERS = {
    ("mlr", "exact"):    "o",
    ("mlr", "jitter"):   "s",
    ("tabpfn", "exact"): "^",
    ("tabpfn", "jitter"):"D",
}
_PANEL_ORDER = [
    ("mlr", "exact"),
    ("mlr", "jitter"),
    ("tabpfn", "exact"),
    ("tabpfn", "jitter"),
]
# Combined-plot label layout: every point gets a label. Each series has a
# near/far dy pair so consecutive labels along the SAME line zigzag in y
# and don't overlap horizontally; the four series also occupy distinct
# vertical bands so labels from different lines don't fight when two lines
# pass close in y.
_COMBINED_LABEL = {
    ("mlr", "exact"):    {"dy_near": +28, "dy_far": +44},
    ("mlr", "jitter"):   {"dy_near": -16, "dy_far": -32},
    ("tabpfn", "exact"): {"dy_near": +14, "dy_far": +30},
    ("tabpfn", "jitter"):{"dy_near": -36, "dy_far": -52},
}
# Per-model plot (one model, both modes) — every point gets a label.
# Each line has a near/far dy so consecutive labels along the SAME line
# zigzag and don't horizontally overlap; exact labels sit above the curve,
# jitter below, so the two lines don't fight over the same vertical band.
_PER_MODEL_LABEL = {
    "exact":  {"dy_near": +18, "dy_far": +34},
    "jitter": {"dy_near": -22, "dy_far": -38},
}


# ----------------------------------------------------------------------
# Metric configs — drive which jsonl field is plotted, what the y axis /
# title / file prefix should say. plot_row_curves() emits one full set of
# 7 PNGs per metric whose data exists in the records (so e.g. MLR-only
# datasets like forest-fires get just the nrmse set, while ship-all gets
# nrmse + mape + mape_tanker).
# ----------------------------------------------------------------------
class _MetricCfg:
    __slots__ = ("field", "file_prefix", "ylabel", "title_label")

    def __init__(self, field: str, file_prefix: str,
                 ylabel: str, title_label: str) -> None:
        self.field = field
        self.file_prefix = file_prefix
        self.ylabel = ylabel
        self.title_label = title_label


_METRIC_NRMSE = _MetricCfg(
    field="nrmse",
    file_prefix="row_curve",
    ylabel="nRMSE = RMSE / std(y_query)",
    title_label="nRMSE",
)
_METRIC_MAPE = _MetricCfg(
    field="mape",
    file_prefix="mape_curve",
    ylabel="MAPE = mean(|y - ŷ| / |y|)  on rows where y ≠ 0",
    title_label="MAPE",
)
_METRIC_MAPE_TANKER = _MetricCfg(
    field="mape_tanker",
    file_prefix="mape_tanker_curve",
    ylabel="MAPE on tanker subset (9 vessels)",
    title_label="MAPE (tanker)",
)


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


def _field_series(recs: list[dict], field: str) -> list[float]:
    """Pick `field` from each record as a float, NaN when missing/null.
    Same rounding-to-3-decimals as _nrmse_series so the OLS-invariant
    flat lines render flat after numerical noise."""
    out: list[float] = []
    for r in recs:
        v = r.get(field)
        if v is None:
            out.append(float("nan"))
            continue
        try:
            out.append(round(float(v), 3))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _pivot(by_combo, skipped, metric_field: str = "nrmse"):
    """Reshape raw records to per-(model,mode) [(n_ctx, [ys...])]+skip_xs.
    `metric_field` selects which jsonl field to plot — 'nrmse' (default)
    keeps the legacy r²-fallback path; anything else reads the field
    directly."""
    series: dict[tuple[str, str], list[tuple[int, list[float]]]] = {}
    skips: dict[tuple[str, str], list[int]] = {}
    series_fn = (
        _nrmse_series if metric_field == "nrmse"
        else (lambda recs: _field_series(recs, metric_field))
    )
    for (model, mode, _k), recs in by_combo.items():
        n_ctx = int(recs[0]["n_ctx"])
        ys = series_fn(recs)
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

def _plot_combined(ax, series, skips, *,
                   ylabel: str = "nRMSE = RMSE / std(y_query)"):
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
        ax.plot(xs, means, style, color=color, linewidth=1.8,
                marker=marker, markersize=7, markeredgecolor="white",
                markeredgewidth=0.6,
                label=f"{model.upper()}/{mode}")
        ax.fill_between(xs, means - stds, means + stds,
                        color=color, alpha=0.12)
        cfg = _COMBINED_LABEL.get((model, mode),
                                  {"dy_near": +18, "dy_far": +34})
        for i, (x, ym) in enumerate(zip(xs, means)):
            if not np.isfinite(ym):
                continue
            dy = cfg["dy_near"] if (i % 2 == 0) else cfg["dy_far"]
            ax.annotate(
                f"({int(x)}, {ym:.3f})",
                xy=(x, ym), xytext=(0, dy),
                textcoords="offset points",
                ha="center", va="center", fontsize=7, color=color,
                arrowprops=dict(arrowstyle="-", color=color,
                                lw=0.4, alpha=0.5, shrinkA=0, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec=color, lw=0.5, alpha=0.92),
            )
        any_drawn = True

    # MLR/exact flat-line hint. Place the caption at the top-left of the
    # axes so it doesn't collide with the dense per-point labels along the
    # flat segment.
    if ("mlr", "exact") in series:
        lanes = series[("mlr", "exact")]
        flat = float(np.nanmean(lanes[-1][1]))
        ax.axhline(flat, color=_COLORS[("mlr", "exact")],
                   alpha=0.35, linestyle=":")
        ax.text(
            0.015, 0.97,
            f"OLS invariant to uniform duplication  (≈ {flat:.3f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=_COLORS[("mlr", "exact")],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=_COLORS[("mlr", "exact")], lw=0.5, alpha=0.9),
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

    # Linear axis: n_ctx = k * n_tr (or k * (N-1)) is an arithmetic series
    # when k_list is, so points sit at equal spacing — no exponential form,
    # no clustering on the right side of a log axis.
    ax.set_xscale("linear")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.set_xlabel("n_ctx (context size per model invocation)")
    ax.set_ylabel(ylabel)
    if any_drawn:
        all_means = np.concatenate([
            np.array([np.nanmean(t[1]) for t in lanes])
            for lanes in series.values()
        ])
        _apply_ylim_with_floor(ax, all_means)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92,
              edgecolor="#aaaaaa",
              handlelength=3.5,   # show enough line for dash/solid to be obvious
              handleheight=1.2,
              borderpad=0.7,
              labelspacing=0.7)
    _style_axes(ax)


# --------------------------------------------------------------------------
# Per-model plot (one model, exact + jitter on the same axes)
# --------------------------------------------------------------------------

def _plot_per_model(ax, model, series, skips, *,
                    ylabel: str = "nRMSE = RMSE / std(y_query)"):
    """Render exact + jitter for ONE model on shared axes.

    Auto-scales tight on this model's data so the exact-vs-jitter delta is
    visible without the cross-model y-range that the combined plot inherits.
    Same color/style/marker conventions as the combined view (line solid for
    jitter, dashed for exact; circles vs squares).
    """
    relevant = [(model, "exact"), (model, "jitter")]
    drawn_keys: list[tuple[str, str]] = []
    for key in relevant:
        if key not in series:
            continue
        lanes = series[key]
        xs = np.array([t[0] for t in lanes])
        means = np.array([np.nanmean(t[1]) for t in lanes])
        stds = np.array([np.nanstd(t[1]) if len(t[1]) > 1 else 0.0
                         for t in lanes])
        color = _COLORS.get(key, "gray")
        style = _STYLES.get(key, "-")
        marker = _MARKERS.get(key, "o")
        ax.plot(xs, means, style, color=color, linewidth=1.8,
                marker=marker, markersize=7, markeredgecolor="white",
                markeredgewidth=0.6,
                label=f"{model.upper()}/{key[1]}")
        ax.fill_between(xs, means - stds, means + stds,
                        color=color, alpha=0.14)
        cfg = _PER_MODEL_LABEL.get(
            key[1], {"dy_near": +18, "dy_far": +34}
        )
        for i, (x, ym) in enumerate(zip(xs, means)):
            if not np.isfinite(ym):
                continue
            # Every point gets a label; alternate near/far within the line
            # so adjacent labels don't sit on top of each other.
            dy = cfg["dy_near"] if (i % 2 == 0) else cfg["dy_far"]
            ax.annotate(
                f"({int(x)}, {ym:.3f})",
                xy=(x, ym), xytext=(0, dy),
                textcoords="offset points",
                ha="center", va="center", fontsize=8, color=color,
                arrowprops=dict(arrowstyle="-", color=color,
                                lw=0.4, alpha=0.5, shrinkA=0, shrinkB=2),
                bbox=dict(boxstyle="round,pad=0.25",
                          fc="white", ec=color, lw=0.5, alpha=0.92),
            )
        drawn_keys.append(key)

    # MLR/exact stays flat by OLS-on-tile invariance — show the same caption
    # as on the combined plot so the per-model view tells the same story.
    if model == "mlr" and ("mlr", "exact") in series:
        lanes = series[("mlr", "exact")]
        flat = float(np.nanmean(lanes[-1][1]))
        ax.axhline(flat, color=_COLORS[("mlr", "exact")],
                   alpha=0.35, linestyle=":")
        ax.text(
            0.015, 0.97,
            f"OLS invariant to uniform duplication  (≈ {flat:.3f})",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            color=_COLORS[("mlr", "exact")],
            bbox=dict(boxstyle="round,pad=0.3", fc="white",
                      ec=_COLORS[("mlr", "exact")], lw=0.5, alpha=0.9),
        )

    # Skipped markers: small open squares at the top of the plot.
    if drawn_keys:
        all_means = np.concatenate([
            np.array([np.nanmean(t[1]) for t in series[k]])
            for k in drawn_keys
        ])
        y_top = float(np.nanmax(all_means))
    else:
        y_top = 1.0
    for key in relevant:
        for xk in skips.get(key, []):
            color = _COLORS.get(key, "gray")
            ax.plot([xk], [y_top * 1.05 if y_top > 0 else 1.0],
                    marker="s", mfc="none", mec=color, ms=10,
                    linestyle="none", zorder=5)

    ax.set_xscale("linear")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.set_xlabel("n_ctx (context size per model invocation)")
    ax.set_ylabel(ylabel)
    if drawn_keys:
        all_means = np.concatenate([
            np.array([np.nanmean(t[1]) for t in series[k]])
            for k in drawn_keys
        ])
        _apply_ylim_with_floor(ax, all_means)
    if drawn_keys:
        ax.legend(loc="center right", frameon=True, framealpha=0.92,
                  edgecolor="#aaaaaa",
                  handlelength=3.5, handleheight=1.2,
                  borderpad=0.7, labelspacing=0.7)
    _style_axes(ax)


# --------------------------------------------------------------------------
# Single-panel plot (one (model, mode), own y-scale)
# --------------------------------------------------------------------------

def _plot_single(ax, model, mode, series, skips, *,
                 ylabel: str = "nRMSE = RMSE / std(y_query)"):
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

    # Linear axis: n_ctx = k * n_tr (or k * (N-1)) is an arithmetic series
    # when k_list is, so points sit at equal spacing — no exponential form,
    # no clustering on the right side of a log axis.
    ax.set_xscale("linear")
    ax.ticklabel_format(axis="x", useOffset=False, style="plain")
    ax.set_xlabel("n_ctx (context size per model invocation)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model.upper()} / {mode}", fontsize=13, loc="left",
                 color=color, fontweight="bold")
    _style_axes(ax)


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------

def _has_metric(records: list[dict], field: str) -> bool:
    """True iff at least one non-skipped record carries a non-null value
    for `field`. mape lives on every record we now produce, but old
    metrics.jsonl files predate it; mape_tanker only exists for ship-all
    and ship-selected — both must be detected, not assumed."""
    for r in records:
        if r.get("skipped"):
            continue
        if r.get(field) is not None:
            return True
    return False


def _plot_one_metric(
    dataset: str,
    by_combo: dict,
    skipped: dict,
    out_dir: Path,
    figsize: tuple[float, float],
    dpi: int,
    cfg: _MetricCfg,
) -> Path:
    """Emit the 7-PNG bundle for ONE metric. Returns the combined path."""
    series, skips = _pivot(by_combo, skipped, metric_field=cfg.field)

    # 1) combined
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _plot_combined(ax, series, skips, ylabel=cfg.ylabel)
    ax.set_title(
        f"{dataset}: {cfg.title_label} vs context size — all (model, mode) combined\n"
        f"(mean ± std over seeds;  n_ctx = k × n_tr for proportional, "
        f"k × (N−1) for loo)",
        fontsize=12,
    )
    fig.tight_layout()
    combined_path = out_dir / f"{cfg.file_prefix}.png"
    fig.savefig(combined_path)
    plt.close(fig)

    # 2) four individual panels
    single_figsize = (figsize[0] * 0.75, figsize[1] * 0.75)
    for model, mode in _PANEL_ORDER:
        fig, ax = plt.subplots(figsize=single_figsize, dpi=dpi)
        _plot_single(ax, model, mode, series, skips, ylabel=cfg.ylabel)
        fig.suptitle(
            f"{dataset}  —  {model.upper()} / {mode}  ·  {cfg.title_label}\n"
            f"(n_ctx = k × n_tr for proportional, k × (N−1) for loo)",
            fontsize=11, y=0.98,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / f"{cfg.file_prefix}_{model}_{mode}.png")
        plt.close(fig)

    # 3) per-model overlay (exact + jitter on tight per-model y-axis)
    for model in ("mlr", "tabpfn"):
        if (model, "exact") not in series and (model, "jitter") not in series:
            continue
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        _plot_per_model(ax, model, series, skips, ylabel=cfg.ylabel)
        fig.suptitle(
            f"{dataset}  —  {model.upper()} (exact + jitter)  ·  {cfg.title_label}\n"
            f"(n_ctx = k × n_tr for proportional, k × (N−1) for loo)",
            fontsize=11, y=0.98,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        fig.savefig(out_dir / f"{cfg.file_prefix}_{model}.png")
        plt.close(fig)

    return combined_path


def plot_row_curves(
    dataset: str,
    jsonl_path: Path,
    out_dir: Path,
) -> Path:
    """Write the row-probe curve PNGs into `out_dir`. Always emits the
    nRMSE bundle (7 files, prefix `row_curve`); additionally emits MAPE
    and MAPE_tanker bundles when the records carry those fields.

    Returns the path to row_curve.png (the nRMSE combined plot) for
    backward-compatible callers; other bundles are written alongside.
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

    _set_rc()
    figsize = CONFIG["viz"]["figsize_curve"]
    dpi = CONFIG["viz"]["dpi"]

    # nRMSE always gets plotted — the legacy r²-fallback in _nrmse_series
    # means even pre-mape jsonls produce a curve here.
    combined_path = _plot_one_metric(
        dataset, by_combo, skipped, out_dir, figsize, dpi, _METRIC_NRMSE,
    )

    # Optional bundles, only when the data is there.
    for cfg in (_METRIC_MAPE, _METRIC_MAPE_TANKER):
        if _has_metric(records, cfg.field):
            _plot_one_metric(
                dataset, by_combo, skipped, out_dir, figsize, dpi, cfg,
            )

    return combined_path
