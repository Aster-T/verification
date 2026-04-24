"""
Column heatmaps for one dataset.

Reads mlr.npz and (optionally) tabpfn.npz from results/column/<dataset>/,
emits four PNGs under `out_dir`:
  {dataset}_mlr_wvec.png
  {dataset}_mlr_wouter.png
  {dataset}_tabpfn_colattn.png          (skipped if tabpfn.npz missing)
  {dataset}_side_by_side.png            (MLR w_vec + w_outer + TabPFN col_attn)

All three heatmaps share a symmetric colorbar with
  vmax = max(|w_outer|.max(), |col_attn|.max())
  vmin = -vmax
so readers can compare magnitudes directly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.configs import CONFIG  # noqa: E402


def _heatmap(ax, mat, feature_names, vmin, vmax, title, cmap):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(len(feature_names)))
    ax.set_yticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(feature_names, fontsize=8)
    ax.set_title(title, fontsize=10)
    return im


def plot_column_heatmaps(
    dataset: str,
    column_dir: Path,
    out_dir: Path,
) -> Path:
    """
    Produce the column-probe heatmaps.

    RETURNS: path to `{dataset}_side_by_side.png`.
    """
    column_dir = Path(column_dir) / dataset
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mlr_path = column_dir / "mlr.npz"
    tp_path = column_dir / "tabpfn.npz"
    assert mlr_path.exists(), f"missing {mlr_path}"

    mlr = np.load(mlr_path, allow_pickle=True)
    w_vec = mlr["w_vec"]
    w_outer = mlr["w_outer"]
    feature_names = [str(x) for x in mlr["feature_names"].tolist()]

    if tp_path.exists():
        tp = np.load(tp_path, allow_pickle=True)
        col_attn = tp["col_attn"]
        col_attn_per_layer = tp["col_attn_per_layer"]
    else:
        col_attn = None
        col_attn_per_layer = None

    abs_outer = np.abs(w_outer).max()
    abs_attn = np.abs(col_attn).max() if col_attn is not None else 0.0
    vmax = max(float(abs_outer), float(abs_attn))
    vmin = -vmax
    cmap = CONFIG["viz"]["heatmap_cmap"]
    dpi = CONFIG["viz"]["dpi"]

    # -- w_vec plot (bar + heatstrip)
    fig, ax = plt.subplots(figsize=(8, 3), dpi=dpi)
    xs = np.arange(len(w_vec))
    ax.bar(xs, w_vec, color=["#3b82f6" if v >= 0 else "#ef4444" for v in w_vec])
    ax.set_xticks(xs)
    ax.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("MLR w (standardized)")
    ax.set_title(f"{dataset}: MLR coefficients")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    wvec_path = out_dir / f"{dataset}_mlr_wvec.png"
    fig.savefig(wvec_path)
    plt.close(fig)

    # -- w_outer heatmap
    fig, ax = plt.subplots(figsize=CONFIG["viz"]["figsize_heatmap_single"], dpi=dpi)
    im = _heatmap(
        ax, w_outer, feature_names, vmin, vmax,
        "MLR W⊗W (rank-1, NOT interactions)", cmap,
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    wouter_path = out_dir / f"{dataset}_mlr_wouter.png"
    fig.savefig(wouter_path)
    plt.close(fig)

    # -- TabPFN heatmap (if present)
    if col_attn is not None:
        fig, ax = plt.subplots(figsize=CONFIG["viz"]["figsize_heatmap_single"], dpi=dpi)
        im = _heatmap(
            ax, col_attn, feature_names, vmin, vmax,
            "TabPFN col-attn (mean over B/H/L, n_estimators=1)", cmap,
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        colattn_path = out_dir / f"{dataset}_tabpfn_colattn.png"
        fig.savefig(colattn_path)
        plt.close(fig)

        # -- Per-layer facet
        fig, axes = plt.subplots(
            4,
            int(np.ceil(col_attn_per_layer.shape[0] / 4)),
            figsize=(16, 12), dpi=dpi,
        )
        axes_flat = axes.ravel()
        for i, layer_attn in enumerate(col_attn_per_layer):
            ax = axes_flat[i]
            ax.imshow(layer_attn, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(f"L{i}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{dataset}: TabPFN col-attn per layer")
        fig.tight_layout()
        perlayer_path = out_dir / f"{dataset}_tabpfn_per_layer.png"
        fig.savefig(perlayer_path)
        plt.close(fig)

    # -- Side-by-side (3 panels: w_vec as heatstrip, w_outer, col_attn)
    ncols = 3 if col_attn is not None else 2
    fig, axes = plt.subplots(
        1, ncols, figsize=CONFIG["viz"]["figsize_heatmap_triptych"], dpi=dpi,
    )
    # panel 1: w_vec as single-row heatmap
    ax0 = axes[0]
    im0 = ax0.imshow(
        w_vec.reshape(1, -1), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
    )
    ax0.set_xticks(range(len(feature_names)))
    ax0.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax0.set_yticks([0]); ax0.set_yticklabels(["w"])
    ax0.set_title("MLR w (std-space)", fontsize=10)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    # panel 2: w_outer
    ax1 = axes[1]
    im1 = _heatmap(
        ax1, w_outer, feature_names, vmin, vmax, "MLR W⊗W (rank-1)", cmap,
    )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # panel 3: col_attn (optional)
    if col_attn is not None:
        ax2 = axes[2]
        im2 = _heatmap(
            ax2, col_attn, feature_names, vmin, vmax,
            "TabPFN col-attn", cmap,
        )
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(f"{dataset}: MLR vs TabPFN column probe", fontsize=12)
    fig.tight_layout()
    side_path = out_dir / f"{dataset}_side_by_side.png"
    fig.savefig(side_path)
    plt.close(fig)
    return side_path
