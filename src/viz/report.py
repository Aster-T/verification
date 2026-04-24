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
    records = read_jsonl(jsonl_path) if jsonl_path.exists() else []
    if not records:
        return '<div class="note">No row_probe records to summarize.</div>'

    by_key: dict[tuple, dict[str, list]] = {}
    for r in records:
        key = (r["model"], r["mode"], r["k"])
        row = by_key.setdefault(key, {"r2": [], "rmse": [], "mae": [], "skipped": 0})
        if r.get("skipped"):
            row["skipped"] += 1
        else:
            row["r2"].append(r["r2"])
            row["rmse"].append(r["rmse"])
            row["mae"].append(r["mae"])

    def fmt(xs):
        if not xs:
            return "-"
        if len(xs) == 1:
            return f"{xs[0]:.4f}"
        return f"{mean(xs):.4f} ± {pstdev(xs):.4f}"

    rows = []
    for key in sorted(by_key.keys(), key=lambda t: (t[0], t[1], t[2])):
        model, mode, k = key
        stats = by_key[key]
        cls = "skipped" if stats["skipped"] and not stats["r2"] else ""
        rows.append(
            f'<tr class="{cls}"><td class="label">{model}</td>'
            f'<td class="label">{mode}</td><td>{k}</td>'
            f'<td>{fmt(stats["r2"])}</td>'
            f'<td>{fmt(stats["rmse"])}</td>'
            f'<td>{fmt(stats["mae"])}</td>'
            f'<td>{stats["skipped"] or ""}</td></tr>'
        )
    head = (
        "<thead><tr><th>model</th><th>mode</th><th>k</th><th>R²</th>"
        "<th>RMSE</th><th>MAE</th><th>skipped</th></tr></thead>"
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
    viz_dir = results_root / "viz"

    nav = " | ".join(f'<a href="#{ds}">{ds}</a>' for ds in datasets)
    sections = []
    for ds in datasets:
        side_by_side = viz_dir / f"{ds}_side_by_side.png"
        per_layer = viz_dir / f"{ds}_tabpfn_per_layer.png"
        row_curve = viz_dir / f"{ds}_row_curve.png"
        jsonl_path = results_root / "row" / f"{ds}.jsonl"

        # Detect Phase 2 skip: tabpfn.npz missing for this dataset.
        tp_path = results_root / "column" / ds / "tabpfn.npz"
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
