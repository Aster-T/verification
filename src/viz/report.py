"""
Self-contained HTML report: embeds PNGs as base64 and pastes per-dataset
aggregated tables from the row jsonl.

No external CSS / JS; the <style> block is inline. Usage via scripts/build_report.py.
"""

from __future__ import annotations

import base64
from pathlib import Path
from statistics import mean, pstdev

from src.utils.io import read_jsonl

_STYLE = """
<style>
  html { background: #fafafa; color: #222; font-family: -apple-system,
    BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { max-width: 1100px; margin: 2rem auto; padding: 0 1.2rem; }
  h1 { font-size: 1.6rem; border-bottom: 2px solid #333; padding-bottom: 0.3em; }
  h2 { font-size: 1.3rem; border-bottom: 1px solid #bbb; padding-bottom: 0.2em;
       margin-top: 2.2em; }
  h3 { font-size: 1.05rem; margin-top: 1.4em; }
  nav a { margin-right: 1em; text-decoration: none; color: #1f4ea8; }
  nav { border: 1px solid #ddd; padding: 0.5em 0.8em; border-radius: 4px;
        background: #fff; }
  img { max-width: 100%; height: auto; display: block; margin: 0.5em 0; }
  table { border-collapse: collapse; font-size: 0.9em; margin: 0.5em 0 1.5em; }
  th, td { border: 1px solid #bbb; padding: 0.25em 0.6em; text-align: right; }
  th { background: #eee; }
  td.label { text-align: left; background: #f4f4f4; }
  details { margin: 0.8em 0; }
  summary { cursor: pointer; font-weight: 600; }
  .skipped { background: #fff7e0; }
  .note { color: #555; font-size: 0.9em; }
</style>
"""


def _img_tag(p: Path) -> str:
    if not p.exists():
        return f'<div class="note">Image missing: {p.name}</div>'
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{data}" alt="{p.name}">'


def _aggregate_row_table(jsonl_path: Path) -> str:
    """
    Render a per-(split_mode, model, mode, k) summary table. Columns:
      split, model, mode, k, n_ctx, n_query, n_features,
      nRMSE (mean ± std over seeds), R², RMSE, MAE, skipped.
    """
    records = read_jsonl(jsonl_path) if jsonl_path.exists() else []
    if not records:
        return '<div class="note">No row_probe records to summarize.</div>'

    by_key: dict[tuple, dict[str, object]] = {}
    for r in records:
        split = r.get("split_mode", "proportional")
        key = (split, r["model"], r["mode"], r["k"])
        row = by_key.setdefault(key, {
            "nrmse": [], "r2": [], "rmse": [], "mae": [], "skipped": 0,
            "n_ctx": r.get("n_ctx"),
            # Accept both new 'n_query' and legacy 'n_te'.
            "n_query": r.get("n_query", r.get("n_te")),
            "n_folds": r.get("n_folds", 1),
            "n_features": r.get("n_features"),
        })
        if r.get("skipped"):
            row["skipped"] += 1  # type: ignore[operator]
        else:
            if r.get("nrmse") is not None:
                row["nrmse"].append(r["nrmse"])  # type: ignore[union-attr]
            row["r2"].append(r["r2"])  # type: ignore[union-attr]
            row["rmse"].append(r["rmse"])  # type: ignore[union-attr]
            row["mae"].append(r["mae"])  # type: ignore[union-attr]

    def fmt(xs: list[float]) -> str:
        if not xs:
            return "-"
        if len(xs) == 1:
            return f"{xs[0]:.4f}"
        return f"{mean(xs):.4f} ± {pstdev(xs):.4f}"

    def fmt_int(v: object) -> str:
        return str(v) if isinstance(v, int) else "-"

    rows = []
    for key in sorted(by_key.keys(), key=lambda t: (t[0], t[1], t[2], t[3])):
        split, model, mode, k = key
        stats = by_key[key]
        cls = "skipped" if stats["skipped"] and not stats["nrmse"] else ""  # type: ignore[index]
        rows.append(
            f'<tr class="{cls}"><td class="label">{split}</td>'
            f'<td class="label">{model}</td>'
            f'<td class="label">{mode}</td><td>{k}</td>'
            f'<td>{fmt_int(stats["n_ctx"])}</td>'
            f'<td>{fmt_int(stats["n_query"])}</td>'
            f'<td>{fmt_int(stats["n_folds"])}</td>'
            f'<td>{fmt_int(stats["n_features"])}</td>'
            f'<td>{fmt(stats["nrmse"])}</td>'  # type: ignore[arg-type]
            f'<td>{fmt(stats["r2"])}</td>'     # type: ignore[arg-type]
            f'<td>{fmt(stats["rmse"])}</td>'   # type: ignore[arg-type]
            f'<td>{fmt(stats["mae"])}</td>'    # type: ignore[arg-type]
            f'<td>{stats["skipped"] or ""}</td></tr>'
        )
    head = (
        "<thead><tr><th>split</th><th>model</th><th>mode</th><th>k</th>"
        "<th>n_ctx</th><th>n_query</th><th>n_folds</th><th>n_features</th>"
        "<th>nRMSE</th><th>R²</th><th>RMSE</th><th>MAE</th>"
        "<th>skipped</th></tr></thead>"
    )
    return f'<table>{head}<tbody>{"".join(rows)}</tbody></table>'


def build_report(
    datasets: list[str],
    results_root: Path,
    out_html: Path,
) -> Path:
    """
    Render a single HTML file embedding heatmaps, curves and aggregated tables
    for all `datasets`. Missing artefacts per dataset fall back to placeholders.
    """
    results_root = Path(results_root)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    nav = " | ".join(f'<a href="#{ds}">{ds}</a>' for ds in datasets)
    sections = []
    for ds in datasets:
        viz_dir = results_root / ds / "viz"
        side_by_side = viz_dir / "side_by_side.png"
        per_layer = viz_dir / "tabpfn_per_layer.png"
        row_curve = viz_dir / "row_curve.png"
        row_single = [
            viz_dir / f"row_curve_{m}_{md}.png"
            for (m, md) in [("mlr", "exact"), ("mlr", "jitter"),
                            ("tabpfn", "exact"), ("tabpfn", "jitter")]
        ]
        jsonl_path = results_root / ds / "row" / "metrics.jsonl"

        # Detect TabPFN skip: tabpfn.npz missing for this dataset.
        tp_path = results_root / ds / "column" / "tabpfn.npz"
        notes = []
        if not tp_path.exists():
            notes.append(
                '<div class="note">TabPFN column probe skipped; heatmap shows MLR only.</div>'
            )

        table_html = _aggregate_row_table(jsonl_path)

        sections.append(
            f"""
            <section id="{ds}">
              <h2>{ds}</h2>
              {"".join(notes)}
              <h3>列探索 (column probe)</h3>
              {_img_tag(side_by_side)}
              <details>
                <summary>逐层 TabPFN attention</summary>
                {_img_tag(per_layer)}
              </details>

              <h3>行探索 (row probe)</h3>
              {_img_tag(row_curve)}
              <details>
                <summary>各 (model, mode) 单独视图(独立 y 轴 scale)</summary>
                {''.join(_img_tag(p) for p in row_single)}
              </details>
              {table_html}
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TabPFN vs MLR: Rows &amp; Columns Probing</title>
{_STYLE}
</head>
<body>
  <h1>TabPFN vs MLR: Rows &amp; Columns Probing</h1>
  <nav>{nav}</nav>
  {''.join(sections)}
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    return out_html
