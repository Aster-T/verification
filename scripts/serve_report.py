"""Serve the unified, on-demand probing report.

Usage:
    python scripts/serve_report.py [--port 8000] [--bind 127.0.0.1]
                                   [--results-root results]

Then open http://localhost:8000/ . No HTML files are written to disk — the
page, the manifest, the table data, and the PNGs are all served live from
whatever currently sits under `results/`.

Endpoints:
    GET /                       single-page frontend (HTML + inline CSS/JS)
    GET /manifest.json          manifest of every (sigma, test_size, dataset,
                                chart) tuple available on disk
    GET /results/<rel-path>     a PNG (or any file) under results/
    GET /table?jsonl=<rel-path> aggregated table JSON for one metrics.jsonl

Stop with Ctrl+C.
"""
from __future__ import annotations

import argparse
import http.server
import json
import logging
import mimetypes
import socket
import sys
from pathlib import Path
from urllib.parse import parse_qs, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from src.viz.report_frontend import FRONTEND_HTML  # noqa: E402
from src.viz.report_server import aggregate_table, build_manifest  # noqa: E402

logger = logging.getLogger(__name__)


def _safe_under(rel: str, root: Path) -> Path | None:
    """Resolve `rel` relative to `root`, rejecting path traversal. Returns
    None if the resolved target falls outside `root` (or if normalization
    fails). The caller treats None as 'send 400'."""
    try:
        target = (root / rel).resolve()
        target.relative_to(root.resolve())
    except (ValueError, OSError):
        return None
    return target


def _make_handler(results_root: Path):
    root_resolved = results_root.resolve()

    class Handler(http.server.BaseHTTPRequestHandler):
        # Tighter logging — show path + status, drop the noisy default.
        def log_message(self, fmt: str, *args) -> None:  # type: ignore[no-untyped-def]
            sys.stderr.write(
                f"[{self.log_date_time_string()}] "
                f"{self.address_string()} {fmt % args}\n"
            )

        def do_GET(self) -> None:  # noqa: N802 (stdlib API)
            try:
                parsed = urlparse(self.path)
                path = parsed.path
                if path in ("/", "/index.html"):
                    self._send_bytes(
                        FRONTEND_HTML.encode("utf-8"),
                        "text/html; charset=utf-8",
                    )
                    return
                if path == "/manifest.json":
                    manifest = build_manifest(root_resolved)
                    self._send_json(manifest)
                    return
                if path == "/table":
                    qs = parse_qs(parsed.query)
                    rel = (qs.get("jsonl") or [""])[0]
                    if not rel:
                        self.send_error(400, "missing 'jsonl' param")
                        return
                    target = _safe_under(rel, root_resolved)
                    if target is None or not target.exists():
                        self.send_error(404, f"no such jsonl: {rel}")
                        return
                    self._send_json(aggregate_table(target))
                    return
                if path.startswith("/results/"):
                    rel = path[len("/results/"):]
                    target = _safe_under(rel, root_resolved)
                    if target is None or not target.is_file():
                        self.send_error(404, f"not found: {rel}")
                        return
                    ctype, _ = mimetypes.guess_type(target.name)
                    self._send_file(target, ctype or "application/octet-stream")
                    return
                self.send_error(404, f"unknown path: {path}")
            except (BrokenPipeError, ConnectionResetError):
                # Client went away mid-response. Nothing useful to do; stay
                # alive for the next request.
                return

        # ---- helpers ---------------------------------------------------

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, obj) -> None:  # type: ignore[no-untyped-def]
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self._send_bytes(body, "application/json; charset=utf-8")

        def _send_file(self, path: Path, content_type: str) -> None:
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            # PNGs in results/ are regenerated by rebuild_reports.py — let
            # the browser cache for a short window but always revalidate.
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)

    return Handler


def _guess_lan_ip() -> str | None:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return None
    finally:
        s.close()


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--bind", default="127.0.0.1",
                   help="Bind address (default: 127.0.0.1; use 0.0.0.0 for LAN).")
    p.add_argument("--results-root", type=Path, default=REPO / "results")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    root = Path(args.results_root)
    if not root.exists():
        sys.stderr.write(f"results-root does not exist: {root}\n")
        sys.exit(1)

    handler_cls = _make_handler(root)
    httpd = http.server.ThreadingHTTPServer((args.bind, args.port), handler_cls)
    httpd.allow_reuse_address = True

    print(f"Serving {root.resolve()}", flush=True)
    print(f"  bind={args.bind}  port={args.port}", flush=True)
    print(f"  http://localhost:{args.port}/", flush=True)
    if args.bind == "0.0.0.0":
        ip = _guess_lan_ip()
        if ip:
            print(f"  http://{ip}:{args.port}/   (LAN)", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
