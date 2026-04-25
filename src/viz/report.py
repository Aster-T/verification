"""
Self-contained HTML report: embeds PNGs as base64 and pastes per-dataset
aggregated tables from row jsonls.

Layout knowledge:

  sigma_root/
    <ds>/
      row/metrics.jsonl   <- LOO row probe (test_size-independent)
      column/{mlr,tabpfn}.npz, viz/...
      viz/row_curve*.png  <- LOO plots
    test_size_<ts>/<ds>/
      row/metrics.jsonl   <- proportional row probe per test_size
      viz/row_curve*.png  <- per-test_size plots
    report.html

A single HTML covers all test_sizes via a top-of-page dropdown. Column-probe
and LOO sections are always visible (test_size-independent). Proportional
row-probe sections come in N copies (one per discovered test_size) wrapped in
`<div class="ts-content" data-ts=...>` and gated by a JS toggle.

No external CSS / JS; the <style> / <script> blocks are inline. Theme toggle
(light/dark) and test_size dropdown both persist their choice in localStorage.

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
    flex-wrap: wrap;
  }
  header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.01em;
  }
  .header-controls {
    display: flex;
    gap: 0.6rem;
    align-items: center;
    flex-wrap: wrap;
  }
  .ts-picker {
    display: flex;
    gap: 0.4rem;
    align-items: center;
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--fg-soft);
    padding: 0.35rem 0.7rem;
    border-radius: 0.5rem;
    font-size: 0.875rem;
  }
  .ts-picker label { font-weight: 500; }
  .ts-picker select {
    background: transparent;
    border: none;
    color: var(--fg);
    font-size: 0.875rem;
    font-weight: 600;
    padding: 0.1rem 0.2rem;
    cursor: pointer;
    font-family: inherit;
  }
  .ts-picker select:focus { outline: 1px solid var(--accent); }
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
  .ts-content { /* JS toggles via display: none */ }
  .ts-tag {
    display: inline-block;
    margin-left: 0.6rem;
    padding: 0.1rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
    background: var(--accent-soft);
    color: var(--accent);
    border-radius: 0.35rem;
    letter-spacing: 0.02em;
    text-transform: none;
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
  const html = document.documentElement;

  // ---- theme toggle ----
  const THEME_KEY = 'probing-report-theme';
  const tbtn = document.getElementById('theme-toggle');
  function applyTheme(theme) {
    html.setAttribute('data-theme', theme);
    if (tbtn) {
      tbtn.innerHTML = theme === 'dark'
        ? '<span aria-hidden="true">☀</span> Light'
        : '<span aria-hidden="true">☾</span> Dark';
      tbtn.setAttribute('aria-pressed', theme === 'dark');
    }
  }
  let initialTheme = localStorage.getItem(THEME_KEY);
  if (!initialTheme) {
    initialTheme = (window.matchMedia &&
                    window.matchMedia('(prefers-color-scheme: dark)').matches)
                   ? 'dark' : 'light';
  }
  applyTheme(initialTheme);
  if (tbtn) {
    tbtn.addEventListener('click', () => {
      const next = (html.getAttribute('data-theme') === 'dark') ? 'light' : 'dark';
      localStorage.setItem(THEME_KEY, next);
      applyTheme(next);
    });
  }

  // ---- test_size dropdown ----
  const TS_KEY = 'probing-report-test-size';
  const tsSelect = document.getElementById('ts-select');
  function applyTestSize(ts) {
    document.body.setAttribute('data-current-ts', ts);
    document.querySelectorAll('.ts-content').forEach(el => {
      el.style.display = (el.dataset.ts === ts) ? '' : 'none';
    });
    if (tsSelect && tsSelect.value !== ts) tsSelect.value = ts;
  }
  if (tsSelect) {
    const available = Array.from(tsSelect.options).map(o => o.value);
    let initialTs = localStorage.getItem(TS_KEY);
    if (!available.includes(initialTs)) initialTs = available[0];
    applyTestSize(initialTs);
    tsSelect.addEventListener('change', () => {
      localStorage.setItem(TS_KEY, tsSelect.value);
      applyTestSize(tsSelect.value);
    });
  }
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
      nRMSE (mean +- std over seeds), R^2, RMSE, MAE, skipped.
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


def _row_curve_block(viz_dir: Path) -> str:
    """Return the HTML fragment for the row-curve combined plot + facet grid
    rendered out of a viz directory. Empty string when the combined PNG is
    missing."""
    row_curve = viz_dir / "row_curve.png"
    if not row_curve.exists():
        return '<div class="note">No row_curve.png in this viz dir.</div>'
    facets = "".join(
        _img_tag(viz_dir / f"row_curve_{m}_{md}.png", f"{m.upper()} / {md}")
        for (m, md) in [("mlr", "exact"), ("mlr", "jitter"),
                        ("tabpfn", "exact"), ("tabpfn", "jitter")]
    )
    return (
        f'{_img_tag(row_curve, "Row curve - all combined")}'
        '<details>'
        '<summary>各 (model, mode) 单独视图 '
        '(独立 y 轴 scale)</summary>'
        f'<div class="facet-grid">{facets}</div>'
        '</details>'
    )


def _ts_label(ts_dir: Path) -> str:
    """test_size_0.5 -> '0.5'."""
    return ts_dir.name[len("test_size_"):]


def build_report(
    datasets: list[str],
    sigma_root: Path,
    test_size_dirs: list[Path],
    out_html: Path,
) -> Path:
    """
    Render a single HTML file embedding all artefacts under `sigma_root`.

    LOO and column-probe blocks (under `sigma_root/<ds>/`) are always visible.
    Proportional blocks under each `test_size_<ts>/<ds>/` are wrapped in
    `<div class="ts-content" data-ts=...>` and shown one at a time via the
    header dropdown.

    `test_size_dirs` is the list of `sigma_root/test_size_<ts>` directories
    that exist (in numeric order). Pass [] to skip the dropdown entirely.
    """
    sigma_root = Path(sigma_root)
    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)

    nav = "".join(f'<a href="#{ds}">{ds}</a>' for ds in datasets)

    sections: list[str] = []
    any_prop_data = False
    for ds in datasets:
        sigma_ds = sigma_root / ds
        loo_jsonl = sigma_ds / "row" / "metrics.jsonl"
        sigma_viz = sigma_ds / "viz"
        column_dir = sigma_ds / "column"

        notes: list[str] = []
        if column_dir.exists() and not (column_dir / "tabpfn.npz").exists():
            notes.append(
                '<div class="note">TabPFN column probe not run; '
                'heatmaps show MLR only.</div>'
            )

        column_block = ""
        if (column_dir / "mlr.npz").exists():
            column_block = (
                '<h3>列探索 (column probe)</h3>'
                f'{_img_tag(sigma_viz / "side_by_side.png", "Column probe - side by side")}'
                '<details>'
                '<summary>逐层 TabPFN attention</summary>'
                f'{_img_tag(sigma_viz / "tabpfn_per_layer.png", "Per-layer TabPFN attention")}'
                '</details>'
            )

        loo_block = ""
        if loo_jsonl.exists():
            loo_block = (
                '<h3>行探索 (row probe — LOO)</h3>'
                f'{_row_curve_block(sigma_viz)}'
                f'{_aggregate_row_table(loo_jsonl)}'
            )

        prop_blocks: list[str] = []
        for ts_dir in test_size_dirs:
            prop_jsonl = ts_dir / ds / "row" / "metrics.jsonl"
            if not prop_jsonl.exists():
                continue
            any_prop_data = True
            ts = _ts_label(ts_dir)
            prop_blocks.append(
                f'<div class="ts-content" data-ts="{ts}">'
                f'<h3>行探索 (row probe — proportional)'
                f'<span class="ts-tag">test_size = {ts}</span></h3>'
                f'{_row_curve_block(ts_dir / ds / "viz")}'
                f'{_aggregate_row_table(prop_jsonl)}'
                '</div>'
            )

        if not (column_block or loo_block or prop_blocks):
            sections.append(
                f'<section id="{ds}"><h2>{ds}</h2>'
                '<div class="note">No artefacts found for this dataset.</div>'
                '</section>'
            )
            continue

        sections.append(
            f'<section id="{ds}">'
            f'<h2>{ds}</h2>'
            f'{"".join(notes)}'
            f'{column_block}'
            f'{loo_block}'
            f'{"".join(prop_blocks)}'
            '</section>'
        )

    # Build the dropdown only if proportional data exists somewhere.
    ts_picker_html = ""
    if any_prop_data and test_size_dirs:
        opts = "".join(
            f'<option value="{_ts_label(d)}">{_ts_label(d)}</option>'
            for d in test_size_dirs
        )
        ts_picker_html = (
            '<div class="ts-picker">'
            '<label for="ts-select">test_size:</label>'
            f'<select id="ts-select">{opts}</select>'
            '</div>'
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
    <div class="header-controls">
      {ts_picker_html}
      <button class="theme-toggle" id="theme-toggle"
              type="button" aria-pressed="false">
        <span aria-hidden="true">☾</span> Dark
      </button>
    </div>
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
