"""
Self-contained HTML report: embeds PNGs as base64 and pastes per-dataset
aggregated tables from the row jsonl.

No external CSS / JS; the <style> / <script> blocks are inline. The
report supports light / dark theme (button in the header; respects
`prefers-color-scheme` and persists the choice in localStorage).

Usage via scripts/build_report.py.
"""

from __future__ import annotations

import base64
from pathlib import Path
from statistics import mean, pstdev

from src.utils.io import read_jsonl

_STYLE = """
<style>
  :root, [data-theme="light"] {
    --bg:            #f6f7f9;
    --bg-card:       #ffffff;
    --bg-inset:      #f1f3f5;
    --bg-hover:      #eef1f4;
    --fg:            #1f2328;
    --fg-muted:      #6b7280;
    --fg-soft:       #4b5563;
    --border:        #e3e7ec;
    --border-soft:   #edf0f3;
    --accent:        #2563eb;
    --accent-hover:  #1d4ed8;
    --accent-soft:   #dbeafe;
    --warn-soft:     #fef3c7;
    --warn-border:   #fcd34d;
    --table-head:    #f1f3f5;
    --table-stripe:  #fafbfc;
    --shadow:        0 1px 2px rgba(15, 23, 42, 0.04),
                     0 2px 8px rgba(15, 23, 42, 0.04);
  }
  [data-theme="dark"] {
    --bg:            #0d1117;
    --bg-card:       #161b22;
    --bg-inset:      #0d1117;
    --bg-hover:      #1f2630;
    --fg:            #e6edf3;
    --fg-muted:      #8b949e;
    --fg-soft:       #adbac7;
    --border:        #30363d;
    --border-soft:   #21262d;
    --accent:        #58a6ff;
    --accent-hover:  #79b8ff;
    --accent-soft:   #1f2e4a;
    --warn-soft:     #3d2f0a;
    --warn-border:   #7c5e14;
    --table-head:    #1c2128;
    --table-stripe:  #161b22;
    --shadow:        0 1px 2px rgba(0,0,0,0.25),
                     0 3px 12px rgba(0,0,0,0.3);
  }

  * { box-sizing: border-box; }
  html {
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, "Noto Sans", "PingFang SC",
                 "Microsoft YaHei", sans-serif;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
    transition: background 0.15s ease, color 0.15s ease;
  }
  body {
    max-width: 1400px;
    margin: 0 auto;
    padding: 2rem 1.5rem 4rem;
  }

  /* ---------- Header ---------- */
  header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 0.5rem 0 1.25rem;
    border-bottom: 2px solid var(--border);
    margin-bottom: 1.75rem;
  }
  header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.01em;
  }
  .theme-toggle {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--fg-soft);
    padding: 0.45rem 0.9rem;
    border-radius: 0.5rem;
    cursor: pointer;
    font-size: 0.875rem;
    font-weight: 500;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    transition: all 0.15s ease;
    user-select: none;
  }
  .theme-toggle:hover {
    background: var(--bg-hover);
    color: var(--fg);
    border-color: var(--accent);
  }

  /* ---------- Nav (sticky chips) ---------- */
  nav {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    padding: 0.75rem 1rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 0.6rem;
    margin-bottom: 2rem;
    position: sticky;
    top: 0.75rem;
    z-index: 10;
    backdrop-filter: saturate(1.8) blur(8px);
    box-shadow: var(--shadow);
  }
  nav a {
    display: inline-block;
    padding: 0.3rem 0.7rem;
    border-radius: 0.4rem;
    color: var(--accent);
    text-decoration: none;
    font-size: 0.88rem;
    font-weight: 500;
    transition: background 0.12s, color 0.12s;
  }
  nav a:hover {
    background: var(--accent-soft);
    color: var(--accent-hover);
  }

  /* ---------- Section (one per dataset) ---------- */
  section {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1.75rem 2rem;
    margin-bottom: 1.75rem;
    box-shadow: var(--shadow);
    scroll-margin-top: 5rem;
  }
  section h2 {
    font-size: 1.35rem;
    font-weight: 600;
    margin: 0 0 0.25rem;
    color: var(--fg);
    letter-spacing: -0.01em;
  }
  section h3 {
    font-size: 1rem;
    font-weight: 600;
    margin: 1.75rem 0 0.9rem;
    color: var(--fg-soft);
    padding-left: 0.75rem;
    border-left: 3px solid var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.04em;
  }

  /* ---------- Images ---------- */
  img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0.5rem auto;
    border-radius: 0.5rem;
    border: 1px solid var(--border-soft);
    background: var(--bg-card);
  }
  figure {
    margin: 0;
    padding: 0;
  }
  figcaption {
    text-align: center;
    font-size: 0.8rem;
    color: var(--fg-muted);
    margin-top: 0.2rem;
  }

  /* 2x2 facet grid for per-(model,mode) plots */
  .facet-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.9rem;
    margin: 0.9rem 0;
  }
  .facet-grid img { margin: 0; }
  @media (max-width: 900px) {
    .facet-grid { grid-template-columns: 1fr; }
  }

  /* ---------- Table ---------- */
  .table-wrap { overflow-x: auto; margin: 0.5rem 0 0.5rem; }
  table {
    border-collapse: collapse;
    font-size: 0.86rem;
    width: 100%;
    font-variant-numeric: tabular-nums;
  }
  th, td {
    border-bottom: 1px solid var(--border-soft);
    padding: 0.45rem 0.75rem;
    text-align: right;
    white-space: nowrap;
  }
  th {
    background: var(--table-head);
    font-weight: 600;
    color: var(--fg-soft);
    position: sticky;
    top: 0;
  }
  tbody tr:nth-child(even) td { background: var(--table-stripe); }
  tbody tr:hover td { background: var(--bg-hover); }
  td.label {
    text-align: left;
    font-weight: 500;
    color: var(--fg);
  }
  tr.skipped td { background: var(--warn-soft); }

  /* ---------- Details / notes ---------- */
  details {
    margin: 0.9rem 0;
    padding: 0.65rem 1rem;
    background: var(--bg-inset);
    border: 1px solid var(--border-soft);
    border-radius: 0.5rem;
    transition: background 0.12s;
  }
  details[open] { background: var(--bg-inset); }
  summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--fg-soft);
    user-select: none;
    list-style: none;
  }
  summary::before {
    content: "▸";
    display: inline-block;
    margin-right: 0.4rem;
    transition: transform 0.15s;
    color: var(--fg-muted);
  }
  details[open] > summary::before { transform: rotate(90deg); }
  details > *:not(summary) { margin-top: 0.8rem; }

  .note {
    color: var(--fg-muted);
    font-size: 0.85rem;
    padding: 0.5rem 0.75rem;
    background: var(--bg-inset);
    border-left: 3px solid var(--warn-border);
    border-radius: 0 0.35rem 0.35rem 0;
    margin: 0.6rem 0;
  }

  footer {
    color: var(--fg-muted);
    font-size: 0.8rem;
    text-align: center;
    padding: 1.5rem 0 0;
    border-top: 1px solid var(--border-soft);
    margin-top: 2rem;
  }

  @media (max-width: 680px) {
    header { flex-direction: column; align-items: flex-start; }
    section { padding: 1.25rem 1rem; }
    body { padding: 1rem 0.75rem 2rem; }
  }
</style>
"""

_SCRIPT = """
<script>
(function() {
  const KEY = 'probing-report-theme';
  const html = document.documentElement;
  const btn = document.getElementById('theme-toggle');
  if (!btn) return;

  function apply(theme) {
    html.setAttribute('data-theme', theme);
    btn.innerHTML = theme === 'dark'
      ? '<span aria-hidden="true">☀</span> Light'
      : '<span aria-hidden="true">☾</span> Dark';
    btn.setAttribute('aria-pressed', theme === 'dark');
  }

  // Initial resolve: saved > system > light.
  let initial = localStorage.getItem(KEY);
  if (!initial) {
    initial = (window.matchMedia &&
               window.matchMedia('(prefers-color-scheme: dark)').matches)
              ? 'dark' : 'light';
  }
  apply(initial);

  btn.addEventListener('click', () => {
    const next = (html.getAttribute('data-theme') === 'dark') ? 'light' : 'dark';
    localStorage.setItem(KEY, next);
    apply(next);
  });
})();
</script>
"""


def _img_tag(p: Path, alt: str | None = None) -> str:
    if not p.exists():
        return f'<div class="note">Image missing: {p.name}</div>'
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    label = alt if alt is not None else p.name
    return f'<img src="data:image/png;base64,{data}" alt="{label}">'


def _aggregate_row_table(jsonl_path: Path) -> str:
    """
    Render a per-(split_mode, model, mode, k) summary table. Columns:
      split, model, mode, k, n_ctx, n_query, n_folds, n_features,
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
            return "—"
        if len(xs) == 1:
            return f"{xs[0]:.4f}"
        return f"{mean(xs):.4f} ± {pstdev(xs):.4f}"

    def fmt_int(v: object) -> str:
        return str(v) if isinstance(v, int) else "—"

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
    return (
        '<div class="table-wrap">'
        f'<table>{head}<tbody>{"".join(rows)}</tbody></table>'
        '</div>'
    )


def build_report(
    datasets: list[str],
    results_root: Path,
    out_html: Path,
) -> Path:
    """
    Render a single HTML file embedding heatmaps, curves and aggregated tables
    for all `datasets`. Missing artefacts per dataset fall back to placeholders.
    Includes a light/dark theme toggle in the header.
    """
    results_root = Path(results_root)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    nav = "".join(f'<a href="#{ds}">{ds}</a>' for ds in datasets)
    sections = []
    for ds in datasets:
        viz_dir = results_root / ds / "viz"
        side_by_side = viz_dir / "side_by_side.png"
        per_layer = viz_dir / "tabpfn_per_layer.png"
        row_curve = viz_dir / "row_curve.png"
        row_single = [
            (viz_dir / f"row_curve_{m}_{md}.png", f"{m.upper()} / {md}")
            for (m, md) in [("mlr", "exact"), ("mlr", "jitter"),
                            ("tabpfn", "exact"), ("tabpfn", "jitter")]
        ]
        jsonl_path = results_root / ds / "row" / "metrics.jsonl"

        # Detect TabPFN skip: tabpfn.npz missing for this dataset.
        tp_path = results_root / ds / "column" / "tabpfn.npz"
        notes: list[str] = []
        if not tp_path.exists():
            notes.append(
                '<div class="note">TabPFN column probe not run; '
                'heatmaps show MLR only.</div>'
            )

        # Skip entire column block if MLR column artefacts don't exist either.
        column_npz = results_root / ds / "column" / "mlr.npz"
        has_column = column_npz.exists()

        column_block = ""
        if has_column:
            column_block = f"""
              <h3>列探索 (column probe)</h3>
              {_img_tag(side_by_side, "Column probe — side by side")}
              <details>
                <summary>逐层 TabPFN attention</summary>
                {_img_tag(per_layer, "Per-layer TabPFN attention")}
              </details>
            """

        table_html = _aggregate_row_table(jsonl_path)

        facet_imgs = "".join(_img_tag(p, label) for p, label in row_single)

        sections.append(
            f"""
            <section id="{ds}">
              <h2>{ds}</h2>
              {"".join(notes)}
              {column_block}

              <h3>行探索 (row probe)</h3>
              {_img_tag(row_curve, "Row curve — all combined")}
              <details>
                <summary>各 (model, mode) 单独视图 (独立 y 轴 scale)</summary>
                <div class="facet-grid">{facet_imgs}</div>
              </details>
              {table_html}
            </section>
            """
        )

    html = f"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TabPFN vs MLR — Rows &amp; Columns Probing</title>
{_STYLE}
</head>
<body>
  <header>
    <h1>TabPFN vs MLR — Rows &amp; Columns Probing</h1>
    <button class="theme-toggle" id="theme-toggle"
            type="button" aria-pressed="false">
      <span aria-hidden="true">☾</span> Dark
    </button>
  </header>
  <nav>{nav}</nav>
  {''.join(sections)}
  <footer>
    Generated by <code>scripts/build_report.py</code> ·
    TabPFN vs MLR probing
  </footer>
{_SCRIPT}
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")
    return out_html
