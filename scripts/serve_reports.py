"""
Serve `results/` over HTTP so other machines on the network can browse the
generated reports in a browser.

Usage:
  python scripts/serve_reports.py [--port 8000] [--bind 0.0.0.0]
                                  [--results-root results]

Then point a browser at  http://<server-ip>:8000/ .

The script does NOT write anything into `results/` — the index page that
lists available reports is rendered in memory on each request, so the
results-tree layout produced by run_row_probe.py / build_report.py is
left strictly intact.

By default it binds to 0.0.0.0 so any host that can reach the server's
network interface (and whose firewall lets through the port) can connect.
Bind to 127.0.0.1 if you only want local access via SSH-tunnel:
  ssh -L 8000:localhost:8000 server
  python scripts/serve_reports.py --bind 127.0.0.1
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import http.server
import socket
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


_INDEX_STYLE = """
* { box-sizing: border-box; }
html { background: #f6f7f9; color: #1f2328;
       font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                    "Helvetica Neue", Arial, "Noto Sans", "PingFang SC",
                    "Microsoft YaHei", sans-serif;
       line-height: 1.55; }
body { max-width: 880px; margin: 0 auto; padding: 2rem 1.5rem 4rem; }
h1 { font-size: 1.6rem; margin: 0 0 0.4rem; letter-spacing: -0.01em; }
.sub { color: #6b7280; font-size: 0.92rem; margin-bottom: 1.6rem; }
.card { background: #fff; border: 1px solid #e3e7ec;
        border-radius: 0.6rem; padding: 0.4rem 0.4rem;
        box-shadow: 0 1px 2px rgba(15,23,42,0.04),
                    0 2px 8px rgba(15,23,42,0.04); }
.empty { padding: 1rem 1.2rem; color: #6b7280; font-size: 0.92rem; }
ul { list-style: none; margin: 0; padding: 0; }
li { border-bottom: 1px solid #edf0f3; }
li:last-child { border-bottom: none; }
a { display: flex; justify-content: space-between; align-items: center;
    padding: 0.7rem 1rem; text-decoration: none; color: #2563eb;
    transition: background 0.12s, color 0.12s;
    border-radius: 0.45rem; }
a:hover { background: #eef1f4; color: #1d4ed8; }
.path { font-weight: 500; font-family: ui-monospace, SFMono-Regular,
        "SF Mono", Menlo, monospace; font-size: 0.9rem; }
.meta { color: #6b7280; font-size: 0.8rem;
        font-variant-numeric: tabular-nums; }
footer { color: #6b7280; font-size: 0.78rem; text-align: center;
         margin-top: 2rem; }
code { background: #f1f3f5; padding: 0.05rem 0.3rem; border-radius: 0.25rem;
       font-size: 0.86em; }
"""


def _format_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024 or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes //= 1024
    return f"{nbytes} GB"


def _discover_reports(root: Path) -> list[Path]:
    """All report.html files anywhere under root, sorted by relative path."""
    if not root.exists():
        return []
    found = sorted(root.rglob("report.html"))
    return found


def _render_index(root: Path) -> bytes:
    reports = _discover_reports(root)
    if reports:
        items = []
        for p in reports:
            rel = p.relative_to(root).as_posix()
            stat = p.stat()
            mtime = dt.datetime.fromtimestamp(stat.st_mtime).strftime(
                "%Y-%m-%d %H:%M"
            )
            size = _format_size(stat.st_size)
            items.append(
                f'<li><a href="{html.escape(rel)}">'
                f'<span class="path">{html.escape(rel)}</span>'
                f'<span class="meta">{mtime} · {size}</span>'
                "</a></li>"
            )
        body = f'<ul>{"".join(items)}</ul>'
    else:
        body = (
            '<div class="empty">No <code>report.html</code> found under '
            f'<code>{html.escape(str(root))}</code>. Run '
            "<code>scripts/run_row.sh</code> or "
            "<code>scripts/build_report.py</code> first.</div>"
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Probing reports — {html.escape(str(root))}</title>
<style>{_INDEX_STYLE}</style>
</head>
<body>
  <h1>Probing reports</h1>
  <div class="sub">Serving <code>{html.escape(str(root))}</code></div>
  <div class="card">{body}</div>
  <footer>scripts/serve_reports.py · pick a report; the page does the rest</footer>
</body>
</html>
"""
    return page.encode("utf-8")


def _make_handler(results_root: Path) -> type[http.server.SimpleHTTPRequestHandler]:
    """Build a SimpleHTTPRequestHandler subclass that serves results_root and
    intercepts '/' to render a dynamic index of available reports."""
    root_str = str(results_root)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
            super().__init__(*args, directory=root_str, **kwargs)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                body = _render_index(results_root)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, fmt: str, *args) -> None:  # type: ignore[no-untyped-def]
            sys.stderr.write(
                f"[{self.log_date_time_string()}] {self.address_string()} "
                f"{fmt % args}\n"
            )

    return Handler


def _guess_lan_ip() -> str | None:
    """Best-effort: which IP would the host use to reach the public internet?
    Doesn't actually send packets — connect() on UDP just sets local addr."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, default=8000,
                   help="TCP port to listen on (default: 8000).")
    p.add_argument("--bind", default="0.0.0.0",
                   help="Bind address. 0.0.0.0 = all interfaces (default), "
                        "127.0.0.1 = localhost only.")
    p.add_argument("--results-root", type=Path, default=REPO / "results",
                   help="Directory to serve. Default: <repo>/results")
    args = p.parse_args()

    root = Path(args.results_root).resolve()
    if not root.exists():
        sys.stderr.write(f"results-root does not exist: {root}\n")
        sys.exit(1)

    handler_cls = _make_handler(root)
    # ThreadingHTTPServer so a slow client doesn't block the next request.
    httpd = http.server.ThreadingHTTPServer((args.bind, args.port), handler_cls)
    httpd.allow_reuse_address = True

    print(f"Serving {root}", flush=True)
    print(f"  bind={args.bind}  port={args.port}", flush=True)
    print("Browse via:", flush=True)
    print(f"  http://localhost:{args.port}/", flush=True)
    lan_ip = _guess_lan_ip()
    if lan_ip and lan_ip != "127.0.0.1":
        print(f"  http://{lan_ip}:{args.port}/   (LAN; clients use this)",
              flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
