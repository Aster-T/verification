"""
Column heatmaps for one dataset.

Reads mlr.npz + (optionally) tabpfn.npz from `column_dir`, writes:
  <out_dir>/side_by_side.png       MLR w-strip + MLR w⊗w + TabPFN col-attn
  <out_dir>/tabpfn_per_layer.png   per-layer TabPFN attention grid (only if
                                   tabpfn.npz is present)

All heatmaps share a symmetric colorbar with
  vmax = max(|w_outer|.max(), |col_attn|.max()); vmin = -vmax
so magnitudes are directly comparable.
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
    Produce the column-probe heatmaps under `out_dir`.

    ARGS:
      dataset:    display name (used only in titles / errors).
      column_dir: where mlr.npz / tabpfn.npz live (e.g. results/<ds>/column/).
      out_dir:    where PNGs are written (e.g. results/<ds>/viz/).

    RETURNS: path to side_by_side.png.
    """
    column_dir = Path(column_dir)
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

    # -- TabPFN per-layer facet (skipped without tabpfn.npz) --
    if col_attn_per_layer is not None:
        fig, axes = plt.subplots(
            4,
            int(np.ceil(col_attn_per_layer.shape[0] / 4)),
            figsize=(16, 12), dpi=dpi,
        )
        axes_flat = axes.ravel()
        last_i = -1
        for i, layer_attn in enumerate(col_attn_per_layer):
            ax = axes_flat[i]
            ax.imshow(layer_attn, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            ax.set_title(f"L{i}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
            last_i = i
        for j in range(last_i + 1, len(axes_flat)):
            axes_flat[j].set_visible(False)
        fig.suptitle(f"{dataset}: TabPFN col-attn per layer")
        fig.tight_layout()
        fig.savefig(out_dir / "tabpfn_per_layer.png")
        plt.close(fig)

    # -- Side-by-side (3 panels: w_vec strip, w_outer, col_attn) --
    ncols = 3 if col_attn is not None else 2
    fig, axes = plt.subplots(
        1, ncols, figsize=CONFIG["viz"]["figsize_heatmap_triptych"], dpi=dpi,
    )
    ax0 = axes[0]
    im0 = ax0.imshow(
        w_vec.reshape(1, -1), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
    )
    ax0.set_xticks(range(len(feature_names)))
    ax0.set_xticklabels(feature_names, rotation=45, ha="right", fontsize=8)
    ax0.set_yticks([0]); ax0.set_yticklabels(["w"])
    ax0.set_title("MLR w (std-space)", fontsize=10)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = axes[1]
    im1 = _heatmap(
        ax1, w_outer, feature_names, vmin, vmax, "MLR W⊗W (rank-1)", cmap,
    )
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    if col_attn is not None:
        ax2 = axes[2]
        im2 = _heatmap(
            ax2, col_attn, feature_names, vmin, vmax, "TabPFN col-attn", cmap,
        )
        fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    fig.suptitle(f"{dataset}: MLR vs TabPFN column probe", fontsize=12)
    fig.tight_layout()
    side_path = out_dir / "side_by_side.png"
    fig.savefig(side_path)
    plt.close(fig)
    return side_path
