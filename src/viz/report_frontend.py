"""Single-page frontend served by `scripts/serve_report.py`.

Pure HTML + inline CSS + inline JS (no external assets, no build step).
The page boots by fetching `/manifest.json`, then renders filters, image
gallery, and collapsible per-slice data tables. Comparison uses a sticky
panel at the top of the page; clicking "+ Compare" on any image pins it
there. Clearing localStorage resets all UI state.

Why a single string constant rather than reading from a `.html` file:
keeping the HTML inline makes the server a one-import, no-extra-resources
script. The string is large (~600 lines) but stable; edit it like a
template.
"""
from __future__ import annotations

FRONTEND_HTML = r"""<!doctype html>
<html lang="en" data-theme="light">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TabPFN vs MLR — Probing report</title>
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
    --shadow:        0 1px 2px rgba(15,23,42,0.04),
                     0 2px 8px rgba(15,23,42,0.04);
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
    line-height: 1.55;
    -webkit-font-smoothing: antialiased;
  }
  body { max-width: 1480px; margin: 0 auto; padding: 1.25rem 1rem 4rem; }

  header {
    display: flex; justify-content: space-between; align-items: center;
    gap: 1rem; margin-bottom: 1rem;
  }
  header h1 {
    font-size: 1.4rem; margin: 0; letter-spacing: -0.01em;
  }
  .header-controls { display: flex; gap: 0.5rem; align-items: center; }

  button {
    cursor: pointer; font: inherit;
    background: var(--bg-card); color: var(--fg);
    border: 1px solid var(--border); border-radius: 0.4rem;
    padding: 0.35rem 0.75rem;
    transition: background 0.12s, border-color 0.12s;
  }
  button:hover { background: var(--bg-hover); border-color: var(--accent); }
  button[disabled] { opacity: 0.45; cursor: not-allowed; }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  button.primary:hover { background: var(--accent-hover); }

  /* ---------- two-column layout ---------- */
  .layout {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 380px;
    gap: 1.25rem;
    align-items: start;
  }
  .layout-main { min-width: 0; }   /* let the gallery shrink, not overflow */
  .layout-side {
    position: sticky;
    top: 0.75rem;
    max-height: calc(100vh - 1.5rem);
    overflow-y: auto;
    /* Subtle scrollbar so the rail looks calm. */
    scrollbar-width: thin;
  }
  @media (max-width: 1100px) {
    /* Below this width the right rail eats too much space — collapse to
       single column and let the compare panel stack at the top. */
    .layout { grid-template-columns: 1fr; }
    .layout-side { position: static; max-height: none; }
  }

  /* ---------- compare panel (right rail) ---------- */
  .compare-panel {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 0.6rem;
    box-shadow: var(--shadow);
    padding: 0.6rem 0.9rem 0.85rem;
    margin-bottom: 0;
  }
  .compare-header {
    display: flex; justify-content: space-between; align-items: center;
    gap: 0.75rem; margin-bottom: 0.4rem;
  }
  .compare-header h2 { font-size: 1rem; margin: 0; color: var(--fg-soft); }
  .compare-header .pill {
    background: var(--accent-soft); color: var(--accent);
    border-radius: 999px; padding: 0.05rem 0.55rem;
    font-size: 0.78rem; font-weight: 600; margin-left: 0.4rem;
  }
  .compare-grid {
    /* Vertical stack so every pinned card stays at the full container width
       — same readable size as when there's only one. Scroll vertically when
       multiple cards are pinned. Use the lightbox (click any image) to zoom
       a single chart for detailed inspection. */
    display: flex; flex-direction: column;
    gap: 0.7rem;
  }
  .compare-grid .empty-hint {
    color: var(--fg-muted); font-size: 0.85rem; padding: 0.5rem 0.25rem;
  }
  .compare-card {
    background: var(--bg-inset); border: 1px solid var(--border-soft);
    border-radius: 0.5rem; padding: 0.4rem; display: flex; flex-direction: column;
  }
  .compare-card .meta {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.3rem; font-size: 0.78rem; color: var(--fg-muted);
    gap: 0.3rem;
  }
  .compare-card .meta span:first-child {
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    flex: 1; min-width: 0;
  }
  .compare-card .x-btn {
    padding: 0.05rem 0.4rem; font-size: 0.75rem;
    background: transparent; border-color: transparent;
    color: var(--fg-muted);
  }
  .compare-card .x-btn:hover { color: var(--accent); }
  .compare-card img {
    width: 100%; height: auto; border-radius: 0.35rem;
    background: var(--bg-card); display: block;
    cursor: zoom-in;
  }

  /* ---------- lightbox (click any image to zoom) ---------- */
  .image-card img { cursor: zoom-in; }
  .lightbox-backdrop {
    position: fixed; inset: 0; z-index: 1000;
    background: rgba(0, 0, 0, 0.92);
    display: none;
    padding: 2rem 2.5rem;
  }
  .lightbox-backdrop.open {
    display: flex;
    justify-content: center; align-items: center;
  }
  .lightbox-backdrop img {
    max-width: 100%; max-height: 100%;
    object-fit: contain;
    border-radius: 0.4rem;
    box-shadow: 0 10px 40px rgba(0, 0, 0, 0.6);
    background: #fff;
  }
  .lightbox-close {
    position: absolute; top: 1rem; right: 1.25rem;
    background: rgba(255, 255, 255, 0.15);
    color: #fff;
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 0.4rem;
    padding: 0.4rem 0.85rem;
    font-size: 0.85rem;
    cursor: pointer;
    user-select: none;
    font-family: inherit;
  }
  .lightbox-close:hover { background: rgba(255, 255, 255, 0.25); }
  .lightbox-caption {
    position: absolute; left: 1.25rem; bottom: 1rem;
    color: rgba(255, 255, 255, 0.85);
    font-size: 0.85rem;
    background: rgba(0, 0, 0, 0.4);
    padding: 0.35rem 0.7rem;
    border-radius: 0.35rem;
    max-width: calc(100% - 2.5rem);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }

  /* ---------- filters ---------- */
  .filters {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 0.6rem; padding: 0.85rem 1rem; margin-bottom: 1rem;
    box-shadow: var(--shadow);
  }
  .filters > .filters-toolbar {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.7rem; flex-wrap: wrap; gap: 0.5rem;
  }
  .filters-toolbar h2 { font-size: 1rem; margin: 0; color: var(--fg-soft); }
  .filters-toolbar .summary {
    color: var(--fg-muted); font-size: 0.85rem;
    font-variant-numeric: tabular-nums;
  }
  .filter-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.85rem;
  }
  .filter-group {
    border: 1px solid var(--border-soft); border-radius: 0.45rem;
    padding: 0.5rem 0.7rem; background: var(--bg-inset);
  }
  .filter-group .group-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.35rem;
  }
  .filter-group .group-head .title {
    font-weight: 600; font-size: 0.85rem; color: var(--fg-soft);
    text-transform: uppercase; letter-spacing: 0.04em;
  }
  .filter-group .group-head .group-actions {
    display: flex; gap: 0.4rem;
  }
  .filter-group .group-head .group-actions a {
    cursor: pointer; color: var(--accent); font-size: 0.78rem;
    text-decoration: none;
  }
  .filter-group .group-head .group-actions a:hover { text-decoration: underline; }
  .filter-options {
    display: flex; flex-wrap: wrap; gap: 0.3rem 0.6rem;
  }
  .filter-options label {
    display: inline-flex; align-items: center; gap: 0.3rem;
    font-size: 0.85rem; color: var(--fg); cursor: pointer;
    user-select: none;
  }
  .filter-options input[type="checkbox"] { margin: 0; cursor: pointer; }

  /* ---------- gallery ---------- */
  .gallery {
    /* Two cards per row at the standard viewport. Below ~900px the second
       column collapses so cards never get unreadable on narrow screens. */
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.85rem; margin-bottom: 2rem;
  }
  @media (max-width: 900px) {
    .gallery { grid-template-columns: 1fr; }
  }
  .image-card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 0.55rem; padding: 0.5rem; display: flex; flex-direction: column;
    box-shadow: var(--shadow);
  }
  .image-card .caption {
    display: flex; justify-content: space-between; align-items: flex-start;
    gap: 0.5rem; margin-bottom: 0.4rem; font-size: 0.82rem;
  }
  .image-card .caption .meta {
    color: var(--fg-soft); flex: 1; min-width: 0; line-height: 1.3;
  }
  .image-card .caption .meta strong { color: var(--fg); }
  .image-card .caption .meta .tags {
    display: flex; flex-wrap: wrap; gap: 0.25rem; margin-top: 0.2rem;
    color: var(--fg-muted); font-size: 0.75rem;
  }
  .image-card .caption .meta .tags span {
    background: var(--bg-inset); padding: 0.05rem 0.4rem; border-radius: 0.25rem;
  }
  .image-card .compare-btn { font-size: 0.78rem; flex-shrink: 0; }
  .image-card .compare-btn.added {
    background: var(--accent); border-color: var(--accent); color: #fff;
  }
  .image-card img {
    width: 100%; height: auto; border-radius: 0.4rem;
    background: var(--bg-inset); display: block;
  }

  .empty-state {
    text-align: center; color: var(--fg-muted); font-size: 0.9rem;
    padding: 3rem 1rem; background: var(--bg-card);
    border: 1px dashed var(--border); border-radius: 0.6rem;
  }

  /* ---------- macro summary ---------- */
  .macro-section {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 0.6rem; padding: 0.85rem 1rem; margin-bottom: 1rem;
    box-shadow: var(--shadow);
  }
  .macro-section > .head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.6rem;
  }
  .macro-section > .head h2 { font-size: 1rem; margin: 0; color: var(--fg-soft); }
  .macro-section > .head .hint {
    color: var(--fg-muted); font-size: 0.82rem;
  }
  .macro-cards {
    display: grid; gap: 0.65rem;
    /* auto-fit collapses empty tracks so a single card takes the full row
       instead of being stuck at one third of the row. */
    grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
  }

  /* ---------- feature distribution section ---------- */
  .feature-dist-section {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 0.6rem; padding: 0.85rem 1rem; margin-bottom: 1rem;
    box-shadow: var(--shadow);
  }
  .feature-dist-section > .head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.6rem; flex-wrap: wrap; gap: 0.4rem;
  }
  .feature-dist-section > .head h2 {
    font-size: 1rem; margin: 0; color: var(--fg-soft);
  }
  .feature-dist-section > .head .hint {
    color: var(--fg-muted); font-size: 0.82rem;
  }
  .feature-dist-grid {
    display: grid; gap: 0.85rem;
    /* Two cards per row at the standard viewport, dropping to one when the
       layout-main column is narrow. Each card holds a tall box-plot grid,
       so we let the image rule its own height (max-width: 100%). */
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  @media (max-width: 1100px) {
    .feature-dist-grid { grid-template-columns: 1fr; }
  }
  .feature-dist-grid .image-card { box-shadow: none; }
  .feature-dist-grid .empty-hint {
    color: var(--fg-muted); font-size: 0.85rem; padding: 0.5rem 0.25rem;
    grid-column: 1 / -1;
  }
  .macro-card {
    background: var(--bg-inset); border: 1px solid var(--border-soft);
    border-radius: 0.5rem; padding: 0.55rem 0.7rem 0.4rem;
  }
  .macro-card .macro-head {
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 0.5rem; margin-bottom: 0.4rem; font-size: 0.88rem;
    color: var(--fg);
  }
  .macro-card .macro-head .ds { font-weight: 600; }
  .macro-card .macro-head .sig {
    font-size: 0.78rem; color: var(--fg-muted);
    background: var(--bg-card); padding: 0.05rem 0.4rem;
    border-radius: 0.3rem; border: 1px solid var(--border-soft);
  }
  .macro-card .macro-head .n {
    font-size: 0.75rem; color: var(--fg-muted);
    font-variant-numeric: tabular-nums;
  }
  .macro-card table {
    width: 100%; border-collapse: collapse; font-size: 0.78rem;
    background: var(--bg-card); border-radius: 0.35rem; overflow: hidden;
  }
  .macro-card thead th {
    text-align: left; padding: 0.25rem 0.45rem;
    background: var(--table-head); color: var(--fg-soft);
    font-weight: 600; font-size: 0.72rem;
    text-transform: uppercase; letter-spacing: 0.04em;
    border-bottom: 1px solid var(--border);
  }
  .macro-card tbody td {
    padding: 0.22rem 0.45rem;
    border-bottom: 1px solid var(--border-soft);
    font-variant-numeric: tabular-nums;
    /* Don't break long "mean ± std" cells across two lines; let the row
       grow horizontally if needed (the card itself sits in an overflow
       wrapper below). */
    white-space: nowrap;
  }
  .macro-card .table-wrap { overflow-x: auto; }
  .macro-card tbody tr:last-child td { border-bottom: none; }
  .macro-card td.label { color: var(--fg-muted); font-variant-numeric: normal; }
  .macro-card.err {
    background: var(--warn-soft); border-color: var(--warn-border);
    color: var(--fg);
    padding: 0.5rem 0.7rem; font-size: 0.82rem;
  }
  .macro-card details.macro-breakdown {
    margin-top: 0.5rem; padding: 0.35rem 0.5rem 0.4rem;
    background: var(--bg-card); border: 1px solid var(--border-soft);
    border-radius: 0.35rem;
  }
  .macro-card details.macro-breakdown > summary {
    cursor: pointer; font-size: 0.78rem; color: var(--fg-soft);
    user-select: none; list-style: none; font-weight: 500;
  }
  .macro-card details.macro-breakdown > summary::before {
    content: "▸"; display: inline-block; margin-right: 0.35rem;
    transition: transform 0.15s; color: var(--fg-muted);
  }
  .macro-card details.macro-breakdown[open] > summary::before { transform: rotate(90deg); }
  .macro-card details.macro-breakdown > table { margin-top: 0.35rem; }
  .macro-card details.macro-breakdown tr.grp td {
    background: var(--bg-inset); color: var(--fg-soft);
    font-weight: 600; font-size: 0.72rem; text-transform: uppercase;
    letter-spacing: 0.04em; padding: 0.2rem 0.45rem;
  }

  /* ---------- tables ---------- */
  .tables-section {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 0.6rem; padding: 1rem; box-shadow: var(--shadow);
    margin-bottom: 2rem;
  }
  .tables-section > h2 {
    font-size: 1.05rem; margin: 0 0 0.4rem; color: var(--fg-soft);
  }
  .tables-section > .hint {
    color: var(--fg-muted); font-size: 0.82rem; margin-bottom: 0.7rem;
  }
  details.table-block {
    margin: 0.4rem 0; padding: 0.5rem 0.8rem;
    background: var(--bg-inset); border: 1px solid var(--border-soft);
    border-radius: 0.45rem;
  }
  details.table-block > summary {
    cursor: pointer; font-weight: 500; color: var(--fg-soft);
    font-size: 0.88rem; user-select: none; list-style: none;
  }
  details.table-block > summary::before {
    content: "▸"; display: inline-block; margin-right: 0.4rem;
    transition: transform 0.15s; color: var(--fg-muted);
  }
  details.table-block[open] > summary::before { transform: rotate(90deg); }
  details.table-block > .table-body { margin-top: 0.6rem; }

  .table-wrap { overflow-x: auto; }
  table {
    width: 100%; border-collapse: collapse; font-size: 0.84rem;
    background: var(--bg-card); border-radius: 0.4rem; overflow: hidden;
  }
  thead th {
    text-align: left; padding: 0.4rem 0.6rem;
    background: var(--table-head); color: var(--fg-soft);
    font-weight: 600; border-bottom: 1px solid var(--border);
    font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
    white-space: nowrap;
  }
  tbody td {
    padding: 0.35rem 0.6rem; border-bottom: 1px solid var(--border-soft);
    font-variant-numeric: tabular-nums; vertical-align: top;
  }
  tbody tr:nth-child(even) td { background: var(--table-stripe); }
  tbody tr.skipped td { color: var(--fg-muted); font-style: italic; }
  tbody td.label { color: var(--fg-muted); font-variant-numeric: normal; }

  .loading { color: var(--fg-muted); font-size: 0.85rem; }
  .err { color: #c62828; font-size: 0.85rem; }

  footer {
    color: var(--fg-muted); font-size: 0.78rem; text-align: center;
    padding: 1.5rem 0 0; border-top: 1px solid var(--border-soft);
  }
</style>
</head>
<body>
  <header>
    <h1>TabPFN vs MLR — Probing report</h1>
    <div class="header-controls">
      <button id="theme-toggle" type="button">☾ Dark</button>
    </div>
  </header>

  <div class="layout">
    <main class="layout-main">
      <section class="filters" id="filters">
        <div class="filters-toolbar">
          <h2>Filters</h2>
          <div class="summary" id="filter-summary">…</div>
        </div>
        <div class="filter-grid" id="filter-grid"></div>
      </section>

      <section class="macro-section" id="macro-section">
        <div class="head">
          <h2>宏指标</h2>
          <span class="hint">每个 (σ · dataset) 一张卡，按当前筛选下的全部记录聚合。</span>
        </div>
        <div class="macro-cards" id="macro-cards"></div>
      </section>

      <section class="feature-dist-section" id="feature-dist-section">
        <div class="head">
          <h2>特征分布</h2>
          <span class="hint">每个 dataset 一张 box plot（与 σ / test_size / jitter_scale 无关）。由 <code>scripts/plot_feature_distributions.py</code> 产出。</span>
        </div>
        <div class="feature-dist-grid" id="feature-dist-grid"></div>
      </section>

      <section id="gallery" class="gallery"></section>

      <section class="tables-section">
        <h2>Data tables</h2>
        <p class="hint">展开查看每个 (dataset · σ · test_size) 切片的聚合指标表。表数据按需加载。</p>
        <div id="tables"></div>
      </section>
    </main>

    <aside class="layout-side">
      <section id="compare" class="compare-panel">
        <div class="compare-header">
          <h2>Compare<span class="pill" id="compare-count">0</span></h2>
          <button id="compare-clear" disabled>Clear all</button>
        </div>
        <div class="compare-grid" id="compare-grid">
          <div class="empty-hint">Click "+ Compare" on any image to the left to pin it here.</div>
        </div>
      </section>
    </aside>
  </div>

  <div class="lightbox-backdrop" id="lightbox" aria-hidden="true" role="dialog">
    <button class="lightbox-close" type="button" id="lightbox-close">Close (Esc)</button>
    <img alt="">
    <div class="lightbox-caption" id="lightbox-caption"></div>
  </div>

  <footer>scripts/serve_report.py · live from results/</footer>

<script>
(function () {
  "use strict";

  // ---------- State ----------
  const STATE = {
    manifest: null,
    filters: { sigma: new Set(), test_size: new Set(), dataset: new Set(),
               jitter_scale: new Set(), chart: new Set() },
    compare: [],   // [{key, src, label}]
    macroCache: new Map(),  // url → {data, ts}
    macroToken: 0,          // incremented to invalidate stale fetches
  };
  const FILTERS_KEY = "probing-report-filters-v1";
  const COMPARE_KEY = "probing-report-compare-v1";
  const THEME_KEY   = "probing-report-theme-v1";

  // ---------- Theme toggle ----------
  function applyTheme(t) {
    document.documentElement.setAttribute("data-theme", t);
    const b = document.getElementById("theme-toggle");
    b.textContent = t === "dark" ? "☀ Light" : "☾ Dark";
  }
  (function initTheme() {
    let t = localStorage.getItem(THEME_KEY);
    if (!t) {
      t = (window.matchMedia &&
           window.matchMedia("(prefers-color-scheme: dark)").matches)
          ? "dark" : "light";
    }
    applyTheme(t);
    document.getElementById("theme-toggle").addEventListener("click", () => {
      const next = document.documentElement.getAttribute("data-theme") === "dark"
                   ? "light" : "dark";
      localStorage.setItem(THEME_KEY, next);
      applyTheme(next);
    });
  })();

  // ---------- Persistence ----------
  function saveFilters() {
    const obj = {};
    for (const dim of Object.keys(STATE.filters)) {
      obj[dim] = Array.from(STATE.filters[dim]);
    }
    localStorage.setItem(FILTERS_KEY, JSON.stringify(obj));
  }
  function loadFilters() {
    try {
      const raw = localStorage.getItem(FILTERS_KEY);
      if (!raw) return null;
      const obj = JSON.parse(raw);
      const out = {};
      for (const k of Object.keys(STATE.filters)) {
        out[k] = new Set(Array.isArray(obj[k]) ? obj[k] : []);
      }
      return out;
    } catch (e) { return null; }
  }
  function saveCompare() {
    localStorage.setItem(COMPARE_KEY, JSON.stringify(STATE.compare));
  }
  function loadCompare() {
    try {
      const raw = localStorage.getItem(COMPARE_KEY);
      if (!raw) return [];
      const arr = JSON.parse(raw);
      return Array.isArray(arr) ? arr : [];
    } catch (e) { return []; }
  }

  // ---------- Manifest fetch ----------
  async function init() {
    try {
      const r = await fetch("/manifest.json");
      if (!r.ok) throw new Error("HTTP " + r.status);
      STATE.manifest = await r.json();
    } catch (e) {
      document.getElementById("gallery").innerHTML =
        '<div class="empty-state err">Failed to load manifest: ' +
        escapeHtml(String(e)) + '</div>';
      return;
    }
    initFilters();
    renderFilterUI();
    STATE.compare = loadCompare();
    renderCompare();
    renderGallery();
    renderFeatureDistributions();
    renderTables();
    renderMacro();
  }

  function initFilters() {
    const m = STATE.manifest;
    const scales = (m.jitter_scales && m.jitter_scales.length)
      ? m.jitter_scales : ["absolute"];
    const saved = loadFilters();
    if (saved && saved.sigma.size && saved.dataset.size) {
      // restore but intersect with what's currently available
      STATE.filters.sigma        = new Set([...saved.sigma].filter(v => m.sigmas.includes(v)));
      STATE.filters.test_size    = new Set([...saved.test_size].filter(v => m.test_sizes.includes(v)));
      STATE.filters.dataset      = new Set([...saved.dataset].filter(v => m.datasets.includes(v)));
      STATE.filters.jitter_scale = new Set([...(saved.jitter_scale || [])].filter(v => scales.includes(v)));
      STATE.filters.chart        = new Set([...saved.chart].filter(v => m.chart_types.some(c => c.id === v)));
      // if any dim ended up empty after the intersect, fall through to defaults
      if (STATE.filters.sigma.size && STATE.filters.test_size.size &&
          STATE.filters.dataset.size && STATE.filters.jitter_scale.size &&
          STATE.filters.chart.size) return;
    }
    // ----- defaults: just enough to be useful, not so much it floods -----
    STATE.filters.sigma        = new Set(m.sigmas.slice(0, 1));
    STATE.filters.test_size    = new Set(m.test_sizes.includes("loo") ? ["loo"] : m.test_sizes.slice(0, 1));
    STATE.filters.dataset      = new Set(m.datasets);
    // Default to "absolute" only — keeps the legacy view unchanged on first
    // load. User flips on per_col_std (or both, for ablation A/B) explicitly.
    STATE.filters.jitter_scale = new Set(scales.includes("absolute") ? ["absolute"] : scales.slice(0, 1));
    STATE.filters.chart        = new Set(m.chart_types.some(c => c.id === "row_curve") ? ["row_curve"] : m.chart_types.slice(0, 1).map(c => c.id));
    saveFilters();
  }

  // ---------- Filter UI ----------
  function renderFilterUI() {
    const m = STATE.manifest;
    const scales = (m.jitter_scales && m.jitter_scales.length)
      ? m.jitter_scales : ["absolute"];
    const groups = [
      {dim: "sigma",        title: "σ (jitter)",   values: m.sigmas.map(v => ({id: v, label: v}))},
      {dim: "test_size",    title: "test_size",    values: m.test_sizes.map(v => ({id: v, label: v === "loo" ? "LOO" : v}))},
      {dim: "dataset",      title: "Dataset",      values: m.datasets.map(v => ({id: v, label: v}))},
      {dim: "jitter_scale", title: "Jitter scale", values: scales.map(v => ({id: v, label: v}))},
      {dim: "chart",        title: "Chart type",   values: m.chart_types.map(c => ({id: c.id, label: c.label}))},
    ];
    const grid = document.getElementById("filter-grid");
    grid.innerHTML = "";
    for (const g of groups) {
      const opts = g.values.map(v => {
        const checked = STATE.filters[g.dim].has(v.id) ? "checked" : "";
        return `<label><input type="checkbox" data-dim="${g.dim}" value="${escapeAttr(v.id)}" ${checked}> ${escapeHtml(v.label)}</label>`;
      }).join("");
      const el = document.createElement("div");
      el.className = "filter-group";
      el.innerHTML = `
        <div class="group-head">
          <div class="title">${escapeHtml(g.title)}</div>
          <div class="group-actions">
            <a data-act="all" data-dim="${g.dim}">all</a>
            <a data-act="none" data-dim="${g.dim}">none</a>
          </div>
        </div>
        <div class="filter-options">${opts || '<span class="loading">empty</span>'}</div>`;
      grid.appendChild(el);
    }
    grid.addEventListener("change", onFilterChange);
    grid.addEventListener("click", onFilterAction);
    updateSummary();
  }

  function onFilterChange(e) {
    const cb = e.target;
    if (!(cb.matches && cb.matches('input[type="checkbox"][data-dim]'))) return;
    const dim = cb.dataset.dim;
    if (cb.checked) STATE.filters[dim].add(cb.value);
    else STATE.filters[dim].delete(cb.value);
    saveFilters();
    updateSummary();
    renderGallery();
    renderFeatureDistributions();
    renderTables();
    renderMacro();
  }

  function onFilterAction(e) {
    const a = e.target;
    if (!a.dataset || !a.dataset.act) return;
    const dim = a.dataset.dim;
    const set = STATE.filters[dim];
    if (a.dataset.act === "all") {
      // re-derive available values from manifest dimension
      let avail;
      if (dim === "chart") {
        avail = STATE.manifest.chart_types.map(c => c.id);
      } else if (dim === "jitter_scale") {
        avail = (STATE.manifest.jitter_scales && STATE.manifest.jitter_scales.length)
          ? STATE.manifest.jitter_scales : ["absolute"];
      } else {
        avail = STATE.manifest[dim === "sigma" ? "sigmas"
                : (dim === "dataset" ? "datasets" : "test_sizes")];
      }
      avail.forEach(v => set.add(v));
    } else {
      set.clear();
    }
    document.querySelectorAll(`input[data-dim="${dim}"]`).forEach(cb => {
      cb.checked = set.has(cb.value);
    });
    saveFilters();
    updateSummary();
    renderGallery();
    renderFeatureDistributions();
    renderTables();
    renderMacro();
  }

  function updateSummary() {
    const m = STATE.manifest;
    const matchImg = m.images.filter(passesImageFilter).length;
    const matchTbl = m.tables.filter(passesTableFilter).length;
    document.getElementById("filter-summary").textContent =
      `${matchImg} / ${m.images.length} images · ${matchTbl} / ${m.tables.length} tables`;
  }

  // ---------- Filter predicates ----------
  // Items written by older servers may not carry `jitter_scale`; treat
  // missing as "absolute" (the only scale the legacy pipeline produced).
  function itemScale(it) {
    return it.jitter_scale || "absolute";
  }
  function passesImageFilter(it) {
    const f = STATE.filters;
    return f.sigma.has(it.sigma) && f.test_size.has(it.test_size)
        && f.dataset.has(it.dataset) && f.chart.has(it.chart)
        && f.jitter_scale.has(itemScale(it));
  }
  function passesTableFilter(it) {
    const f = STATE.filters;
    return f.sigma.has(it.sigma) && f.test_size.has(it.test_size)
        && f.dataset.has(it.dataset)
        && f.jitter_scale.has(itemScale(it));
  }

  // ---------- Feature distributions (dataset-level box plots) ----------
  function renderFeatureDistributions() {
    const grid = document.getElementById("feature-dist-grid");
    if (!grid) return;
    const m = STATE.manifest;
    const all = Array.isArray(m.feature_distributions)
      ? m.feature_distributions : [];
    // Gated by the dataset filter only — feature distributions don't
    // depend on σ / test_size / jitter_scale.
    const items = all.filter(it => STATE.filters.dataset.has(it.dataset))
                     .slice()
                     .sort((a, b) => a.dataset.localeCompare(b.dataset));
    if (!items.length) {
      grid.innerHTML = all.length
        ? '<div class="empty-hint">No dataset selected. Pick at least one in the dataset filter.</div>'
        : '<div class="empty-hint">No <code>results/feature_distributions/&lt;dataset&gt;.png</code> on disk yet. Run <code>scripts/plot_feature_distributions.py --local-all</code>.</div>';
      return;
    }
    const compareSet = new Set(STATE.compare.map(c => c.key));
    grid.innerHTML = items.map(it => {
      const key = it.url;
      const inCompare = compareSet.has(key);
      return `
        <div class="image-card">
          <div class="caption">
            <div class="meta">
              <strong>${escapeHtml(it.dataset)}</strong>
              <div>feature distribution</div>
            </div>
            <button class="compare-btn ${inCompare ? "added" : ""}"
                    data-key="${escapeAttr(key)}"
                    data-url="${escapeAttr(it.url)}"
                    data-label="${escapeAttr(it.label)}">
              ${inCompare ? "✓ Added" : "+ Compare"}
            </button>
          </div>
          <img loading="lazy" src="${escapeAttr(it.url)}" alt="${escapeAttr(it.label)}">
        </div>`;
    }).join("");
    grid.querySelectorAll(".compare-btn").forEach(b => {
      b.addEventListener("click", onCompareClick);
    });
  }

  // ---------- Gallery ----------
  function renderGallery() {
    const gallery = document.getElementById("gallery");
    const m = STATE.manifest;
    const items = m.images.filter(passesImageFilter);
    items.sort(imageSortKey);
    if (!items.length) {
      gallery.innerHTML = '<div class="empty-state">没有图匹配当前筛选 — 试试在左上角放宽几个维度。</div>';
      return;
    }
    const compareSet = new Set(STATE.compare.map(c => c.key));
    const html = items.map(it => {
      const key = it.url;  // url is unique per image
      const inCompare = compareSet.has(key);
      const scale = itemScale(it);
      const scaleTag = scale === "absolute"
        ? "" : `<span>scale ${escapeHtml(scale)}</span>`;
      const tags = `
        <div class="tags">
          <span>σ ${escapeHtml(it.sigma)}</span>
          <span>${it.test_size === "loo" ? "LOO" : "test_size " + escapeHtml(it.test_size)}</span>
          ${scaleTag}
        </div>`;
      return `
        <div class="image-card">
          <div class="caption">
            <div class="meta">
              <strong>${escapeHtml(it.dataset)}</strong>
              <div>${escapeHtml(it.label)}</div>
              ${tags}
            </div>
            <button class="compare-btn ${inCompare ? "added" : ""}"
                    data-key="${escapeAttr(key)}"
                    data-url="${escapeAttr(it.url)}"
                    data-label="${escapeAttr(captionLabel(it))}">
              ${inCompare ? "✓ Added" : "+ Compare"}
            </button>
          </div>
          <img loading="lazy" src="${escapeAttr(it.url)}" alt="${escapeAttr(it.label)}">
        </div>`;
    }).join("");
    gallery.innerHTML = html;
    gallery.querySelectorAll(".compare-btn").forEach(b => {
      b.addEventListener("click", onCompareClick);
    });
  }

  function captionLabel(it) {
    const scale = itemScale(it);
    const scalePart = scale === "absolute" ? "" : ` · scale=${scale}`;
    return `${it.dataset} · σ=${it.sigma} · ${it.test_size === "loo" ? "LOO" : "ts=" + it.test_size}${scalePart} · ${it.label}`;
  }

  function imageSortKey(a, b) {
    if (a.dataset !== b.dataset) return a.dataset.localeCompare(b.dataset);
    if (a.sigma   !== b.sigma)   return a.sigma.localeCompare(b.sigma);
    const sa = itemScale(a), sb = itemScale(b);
    if (sa !== sb) return sa.localeCompare(sb);
    if (a.test_size !== b.test_size) {
      if (a.test_size === "loo") return -1;
      if (b.test_size === "loo") return 1;
      return parseFloat(a.test_size) - parseFloat(b.test_size);
    }
    return a.chart.localeCompare(b.chart);
  }

  // ---------- Compare panel ----------
  function onCompareClick(e) {
    const b = e.currentTarget;
    const key = b.dataset.key;
    const idx = STATE.compare.findIndex(c => c.key === key);
    if (idx >= 0) {
      STATE.compare.splice(idx, 1);
    } else {
      STATE.compare.push({key, src: b.dataset.url, label: b.dataset.label});
    }
    saveCompare();
    renderCompare();
    // also refresh button state in gallery
    document.querySelectorAll(`.compare-btn[data-key="${escapeAttr(key)}"]`).forEach(btn => {
      const has = STATE.compare.some(c => c.key === key);
      btn.classList.toggle("added", has);
      btn.textContent = has ? "✓ Added" : "+ Compare";
    });
  }
  function renderCompare() {
    const grid = document.getElementById("compare-grid");
    const count = STATE.compare.length;
    document.getElementById("compare-count").textContent = count;
    document.getElementById("compare-clear").disabled = count === 0;
    if (!count) {
      grid.innerHTML = '<div class="empty-hint">Click "+ Compare" on any image below to pin it here, side by side.</div>';
      return;
    }
    grid.innerHTML = STATE.compare.map((c, i) => `
      <div class="compare-card">
        <div class="meta">
          <span>${escapeHtml(c.label)}</span>
          <button class="x-btn" data-idx="${i}" title="remove">✕</button>
        </div>
        <img src="${escapeAttr(c.src)}" alt="${escapeAttr(c.label)}">
      </div>
    `).join("");
    grid.querySelectorAll(".x-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        const i = parseInt(btn.dataset.idx, 10);
        const removed = STATE.compare.splice(i, 1)[0];
        saveCompare();
        renderCompare();
        if (removed) {
          document.querySelectorAll(`.compare-btn[data-key="${escapeAttr(removed.key)}"]`).forEach(gb => {
            gb.classList.remove("added");
            gb.textContent = "+ Compare";
          });
        }
      });
    });
  }
  document.getElementById("compare-clear").addEventListener("click", () => {
    if (!STATE.compare.length) return;
    if (!confirm("Clear all pinned images?")) return;
    const keys = STATE.compare.map(c => c.key);
    STATE.compare = [];
    saveCompare();
    renderCompare();
    keys.forEach(k => {
      document.querySelectorAll(`.compare-btn[data-key="${escapeAttr(k)}"]`).forEach(gb => {
        gb.classList.remove("added");
        gb.textContent = "+ Compare";
      });
    });
  });

  // ---------- Tables ----------
  function renderTables() {
    const host = document.getElementById("tables");
    const items = STATE.manifest.tables.filter(passesTableFilter);
    items.sort((a, b) => a.label.localeCompare(b.label));
    if (!items.length) {
      host.innerHTML = '<div class="loading">没有表匹配当前筛选。</div>';
      return;
    }
    host.innerHTML = items.map(t => `
      <details class="table-block" data-jsonl="${escapeAttr(t.jsonl)}">
        <summary>${escapeHtml(t.label)}</summary>
        <div class="table-body"><div class="loading">点击展开后加载…</div></div>
      </details>
    `).join("");
    host.querySelectorAll("details.table-block").forEach(d => {
      d.addEventListener("toggle", () => {
        if (!d.open) return;
        if (d.dataset.loaded === "1") return;
        loadTableInto(d);
      });
    });
  }

  async function loadTableInto(detailsEl) {
    const body = detailsEl.querySelector(".table-body");
    body.innerHTML = '<div class="loading">loading…</div>';
    try {
      const url = "/table?jsonl=" + encodeURIComponent(detailsEl.dataset.jsonl);
      const r = await fetch(url);
      if (!r.ok) throw new Error("HTTP " + r.status);
      const data = await r.json();
      body.innerHTML = renderTableHtml(data);
      detailsEl.dataset.loaded = "1";
    } catch (e) {
      body.innerHTML = '<div class="err">load failed: ' + escapeHtml(String(e)) + '</div>';
    }
  }

  function renderTableHtml(data) {
    if (!data.rows || !data.rows.length) {
      return '<div class="loading">no records</div>';
    }
    const head = ["split", "model", "mode", "k",
                  "n_ctx", "n_query", "n_folds", "n_features",
                  "nRMSE", "R²", "RMSE", "MAE", "MAPE"];
    if (data.has_tanker) head.push("MAPE_tanker");
    head.push("skipped");
    const headHtml = "<thead><tr>" + head.map(h => `<th>${escapeHtml(h)}</th>`).join("") + "</tr></thead>";

    const fmt = s => {
      if (!s || s.n === 0) return "—";
      const dp = decimalPlaces();
      if (s.n === 1) return s.mean.toFixed(dp);
      return `${s.mean.toFixed(dp)} ± ${s.std.toFixed(dp)}`;
    };
    const fmtInt = v => (v === null || v === undefined) ? "—" : String(v);
    const rows = data.rows.map(r => {
      const tankerCell = data.has_tanker ? `<td>${escapeHtml(fmt(r.mape_tanker))}</td>` : "";
      const cls = (r.skipped_count && r.nrmse.n === 0) ? "skipped" : "";
      return `<tr class="${cls}">
        <td class="label">${escapeHtml(r.split)}</td>
        <td class="label">${escapeHtml(r.model)}</td>
        <td class="label">${escapeHtml(r.mode)}</td>
        <td>${escapeHtml(String(r.k))}</td>
        <td>${escapeHtml(fmtInt(r.n_ctx))}</td>
        <td>${escapeHtml(fmtInt(r.n_query))}</td>
        <td>${escapeHtml(fmtInt(r.n_folds))}</td>
        <td>${escapeHtml(fmtInt(r.n_features))}</td>
        <td>${escapeHtml(fmt(r.nrmse))}</td>
        <td>${escapeHtml(fmt(r.r2))}</td>
        <td>${escapeHtml(fmt(r.rmse))}</td>
        <td>${escapeHtml(fmt(r.mae))}</td>
        <td>${escapeHtml(fmt(r.mape))}</td>
        ${tankerCell}
        <td>${r.skipped_count || ""}</td>
      </tr>`;
    }).join("");
    return `<div class="table-wrap"><table>${headHtml}<tbody>${rows}</tbody></table></div>`;
  }

  // ---------- Macro summary ----------
  function computeMacroPairs() {
    // Returns one entry per (σ, jitter_scale, dataset) covered by the
    // current filters — splitting by scale so absolute vs per_col_std
    // ablations show up as separate cards instead of being averaged
    // together. Each entry buckets jsonls by test_size for the per-ts
    // drilldown.
    const f = STATE.filters;
    const pairs = new Map();
    for (const t of STATE.manifest.tables) {
      if (!f.sigma.has(t.sigma)) continue;
      if (!f.test_size.has(t.test_size)) continue;
      if (!f.dataset.has(t.dataset)) continue;
      const scale = itemScale(t);
      if (!f.jitter_scale.has(scale)) continue;
      const key = t.sigma + "|" + scale + "|" + t.dataset;
      let entry = pairs.get(key);
      if (!entry) {
        entry = { sigma: t.sigma, dataset: t.dataset,
                  jitter_scale: scale,
                  byTs: new Map(), allJsonls: [] };
        pairs.set(key, entry);
      }
      if (!entry.byTs.has(t.test_size)) entry.byTs.set(t.test_size, []);
      entry.byTs.get(t.test_size).push(t.jsonl);
      entry.allJsonls.push(t.jsonl);
    }
    return Array.from(pairs.values()).sort((a, b) => {
      if (a.dataset !== b.dataset) return a.dataset.localeCompare(b.dataset);
      if (a.sigma   !== b.sigma)   return a.sigma.localeCompare(b.sigma);
      return a.jitter_scale.localeCompare(b.jitter_scale);
    });
  }

  function macroUrl(jsonls) {
    return "/macro?" + jsonls.slice().sort()
      .map(j => "jsonl=" + encodeURIComponent(j)).join("&");
  }

  async function fetchMacro(jsonls) {
    const url = macroUrl(jsonls);
    const cached = STATE.macroCache.get(url);
    if (cached) return cached;
    const r = await fetch(url);
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    STATE.macroCache.set(url, data);
    return data;
  }

  function tsOrderKey(ts) {
    if (ts === "loo") return -Infinity;
    const v = parseFloat(ts);
    return isFinite(v) ? v : Infinity;
  }

  async function renderMacro() {
    const host = document.getElementById("macro-cards");
    const pairs = computeMacroPairs();
    if (!pairs.length) {
      host.innerHTML = '<div class="loading">没有 (σ · dataset) 在当前筛选范围内。</div>';
      return;
    }
    const myToken = ++STATE.macroToken;
    host.innerHTML = '<div class="loading">computing macro… ('
      + pairs.length + ' card' + (pairs.length === 1 ? "" : "s") + ')</div>';

    // Fetch overall + per-test_size in parallel; cache hits make this
    // near-instant after the first render.
    const cards = await Promise.all(pairs.map(async pair => {
      try {
        const tsList = Array.from(pair.byTs.keys())
          .sort((a, b) => tsOrderKey(a) - tsOrderKey(b));
        const overallP = fetchMacro(pair.allJsonls);
        const perTsP = Promise.all(tsList.map(async ts =>
          ({ ts, data: await fetchMacro(pair.byTs.get(ts)) })));
        const [overall, perTs] = await Promise.all([overallP, perTsP]);
        return renderMacroCard(pair, overall, perTs);
      } catch (e) {
        return '<div class="macro-card err">'
          + escapeHtml(pair.dataset) + ' · σ=' + escapeHtml(pair.sigma)
          + ': ' + escapeHtml(String(e)) + '</div>';
      }
    }));
    if (myToken !== STATE.macroToken) return;
    host.innerHTML = cards.join("");
  }

  function macroNum(x) {
    if (x === null || x === undefined || Number.isNaN(x)) return "—";
    const dp = decimalPlaces();
    const a = Math.abs(x);
    // Scientific for very large / very small magnitudes — keeps cells short.
    if (a !== 0 && (a >= 1e4 || a < 1e-3)) {
      return x.toExponential(dp).replace("e+", "e").replace("e-0", "e-");
    }
    return x.toFixed(dp);
  }
  function macroFmt(s) {
    if (!s || s.n === 0) return "—";
    if (s.n === 1) return macroNum(s.mean);
    return macroNum(s.mean) + " ± " + macroNum(s.std);
  }

  function macroTableHtml(data, firstColLabel) {
    const models = Object.keys(data.per_model);
    if (!models.length) return '<div class="loading">no records</div>';
    const cols = ["nRMSE", "MAE", "MAPE"];
    const keys = ["nrmse", "mae", "mape"];
    if (data.has_tanker) { cols.push("MAPE_tanker"); keys.push("mape_tanker"); }
    const head = "<thead><tr><th>" + escapeHtml(firstColLabel) + "</th>"
      + cols.map(c => "<th>" + escapeHtml(c) + "</th>").join("")
      + "<th>n</th></tr></thead>";
    const rows = models.map(m => {
      const v = data.per_model[m];
      return "<tr><td class='label'>" + escapeHtml(m) + "</td>"
        + keys.map(k => "<td>" + escapeHtml(macroFmt(v[k])) + "</td>").join("")
        + "<td>" + (v.n || 0) + "</td></tr>";
    }).join("");
    return '<div class="table-wrap"><table>' + head + "<tbody>" + rows + "</tbody></table></div>";
  }

  function renderMacroCard(pair, overall, perTs) {
    const skipNote = overall.n_skipped
      ? ' <span class="n">' + overall.n_skipped + ' skipped</span>' : "";
    const scaleTag = (pair.jitter_scale && pair.jitter_scale !== "absolute")
      ? '<span class="sig">scale=' + escapeHtml(pair.jitter_scale) + '</span>'
      : "";
    const headHtml = '<div class="macro-head">'
      + '<span class="ds">' + escapeHtml(pair.dataset) + '</span>'
      + '<span class="sig">σ=' + escapeHtml(pair.sigma) + '</span>'
      + scaleTag
      + '<span class="n">' + overall.n_records + ' rec'
      + (overall.n_records === 1 ? "" : "s") + skipNote + '</span>'
      + '</div>';

    const overallHtml = macroTableHtml(overall, "model");

    // Per-test_size breakdown: one big table where each model gets its
    // own grouped section. Collapsed by default; gives the user the
    // "ship-all under each test_size" drilldown without extra cards.
    let perTsHtml = "";
    if (perTs.length > 1) {
      const cols = ["nRMSE", "MAE", "MAPE"];
      const keys = ["nrmse", "mae", "mape"];
      const showTanker = perTs.some(g => g.data.has_tanker);
      if (showTanker) { cols.push("MAPE_tanker"); keys.push("mape_tanker"); }
      const allModels = new Set();
      for (const g of perTs) Object.keys(g.data.per_model).forEach(m => allModels.add(m));
      const models = Array.from(allModels).sort();
      const rows = [];
      for (const m of models) {
        rows.push('<tr class="grp"><td colspan="' + (cols.length + 2)
          + '">' + escapeHtml(m) + '</td></tr>');
        for (const g of perTs) {
          const v = g.data.per_model[m];
          const tsLabel = g.ts === "loo" ? "LOO" : "ts=" + g.ts;
          if (!v) {
            rows.push("<tr><td class='label'>" + escapeHtml(tsLabel) + "</td>"
              + cols.map(() => "<td>—</td>").join("") + "<td>0</td></tr>");
            continue;
          }
          rows.push("<tr><td class='label'>" + escapeHtml(tsLabel) + "</td>"
            + keys.map(k => "<td>" + escapeHtml(macroFmt(v[k])) + "</td>").join("")
            + "<td>" + (v.n || 0) + "</td></tr>");
        }
      }
      const head = "<thead><tr><th>test_size</th>"
        + cols.map(c => "<th>" + escapeHtml(c) + "</th>").join("")
        + "<th>n</th></tr></thead>";
      perTsHtml = '<details class="macro-breakdown"><summary>'
        + '按 test_size 展开（' + perTs.length + '）</summary>'
        + '<div class="table-wrap"><table>' + head + '<tbody>'
        + rows.join("") + '</tbody></table></div>'
        + '</details>';
    }

    return '<div class="macro-card">'
      + headHtml + overallHtml + perTsHtml + '</div>';
  }

  // ---------- Tiny escape helpers ----------
  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
  }
  function escapeAttr(s) { return escapeHtml(s); }

  // ---------- Numeric display precision (sourced from CONFIG.viz.decimal_places via /manifest.json) ----------
  function decimalPlaces() {
    const dp = STATE.manifest && STATE.manifest.decimal_places;
    return Number.isInteger(dp) && dp >= 0 ? dp : 4;
  }

  // ---------- Lightbox: click any chart to zoom; Esc/click outside to close ----------
  (function setupLightbox() {
    const lb = document.getElementById("lightbox");
    if (!lb) return;
    const lbImg = lb.querySelector("img");
    const lbCap = document.getElementById("lightbox-caption");
    function open(src, alt) {
      lbImg.src = src;
      lbImg.alt = alt || "";
      lbCap.textContent = alt || "";
      lb.classList.add("open");
      lb.setAttribute("aria-hidden", "false");
    }
    function close() {
      lb.classList.remove("open");
      lb.setAttribute("aria-hidden", "true");
      lbImg.removeAttribute("src");
      lbCap.textContent = "";
    }
    document.addEventListener("click", (e) => {
      const t = e.target;
      if (t.tagName === "IMG" && t.closest(".image-card, .compare-card")) {
        open(t.currentSrc || t.src, t.alt);
      }
    });
    lb.addEventListener("click", (e) => {
      // Click on backdrop or close button closes; click on the image itself stays open.
      if (e.target === lb || e.target.id === "lightbox-close") close();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && lb.classList.contains("open")) close();
    });
  })();

  // boot
  init();
})();
</script>
</body>
</html>
"""
