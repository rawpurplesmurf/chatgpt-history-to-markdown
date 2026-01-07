#!/usr/bin/env python3
"""
Local web server for browsing the generated Markdown archive.

Routes:
  /              -> web/index.html
  /search        -> JSON search endpoint (requires search index)
  /app/<path>    -> web/<path>
  /markdown/<path> -> markdown/<path> (generated output)
"""

import argparse
import json
import logging
import mimetypes
import os
import shutil
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

logger = logging.getLogger(__name__)

try:
    from search_index import SearchIndex
except Exception:  # pragma: no cover - optional at runtime
    SearchIndex = None


def _is_within(path: Path, base: Path) -> bool:
    try:
        path = path.resolve()
        base = base.resolve()
    except FileNotFoundError:
        return False
    return path == base or base in path.parents


class ArchiveServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_cls, repo_root: Path):
        self.repo_root = repo_root
        self.web_dir = repo_root / "web"
        self.markdown_dir = repo_root / "markdown"
        self.search_index = None
        self.search_lock = threading.Lock()
        super().__init__(server_address, handler_cls)

    def get_search_index(self):
        if self.search_index is not None:
            return self.search_index
        if SearchIndex is None:
            return None
        with self.search_lock:
            if self.search_index is None:
                self.search_index = SearchIndex.try_load(self.markdown_dir)
        return self.search_index


class ArchiveHandler(BaseHTTPRequestHandler):
    server_version = "ChatGPTHistoryServer/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        request_path = unquote(parsed.path)

        if request_path == "/":
            self._serve_static(self.server.web_dir / "index.html", self.server.web_dir)
            return

        if request_path == "/search":
            self._handle_search(parsed)
            return

        if request_path.startswith("/app/"):
            rel = request_path[len("/app/") :]
            self._serve_static(self.server.web_dir / rel, self.server.web_dir)
            return

        if request_path.startswith("/markdown/"):
            rel = request_path[len("/markdown/") :]
            self._serve_static(self.server.markdown_dir / rel, self.server.markdown_dir)
            return

        self.send_error(404, "Not found")

    def _handle_search(self, parsed) -> None:
        params = parse_qs(parsed.query)
        query = ""
        if "q" in params:
            query = params["q"][0]
        elif "query" in params:
            query = params["query"][0]
        query = (query or "").strip()

        if not query:
            self._send_json({"error": "Missing query parameter."}, status=400)
            return

        limit = 20
        if "limit" in params:
            try:
                limit = int(params["limit"][0])
            except (TypeError, ValueError):
                limit = 20
        limit = max(1, min(50, limit))

        mode = params.get("mode", ["hybrid"])[0]
        scope = params.get("scope", ["both"])[0]
        sort = params.get("sort", ["relevance"])[0]
        role = params.get("role", [""])[0]
        date_from = params.get("from", [""])[0]
        date_to = params.get("to", [""])[0]
        snippet_tokens = 12
        if "snippet" in params:
            try:
                snippet_tokens = int(params["snippet"][0])
            except (TypeError, ValueError):
                snippet_tokens = 12

        index = self.server.get_search_index()
        if not index:
            self._send_json(
                {"error": "Search index not available. Run the converter to build it."},
                status=503,
            )
            return

        results = index.search(
            query,
            limit=limit,
            mode=mode,
            scope=scope,
            role=role,
            date_from=date_from,
            date_to=date_to,
            sort=sort,
            snippet_tokens=snippet_tokens,
        )
        payload = {"query": query, "count": len(results), "results": results}
        self._send_json(payload, status=200)

    def _send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(data)
        except BrokenPipeError:
            return

    def _serve_static(self, path: Path, base: Path) -> None:
        base = base.resolve()
        try:
            resolved = path.resolve()
        except FileNotFoundError:
            self.send_error(404, "Not found")
            return

        if not _is_within(resolved, base):
            self.send_error(403, "Forbidden")
            return

        if resolved.is_dir():
            self.send_error(404, "Not found")
            return

        if not resolved.exists():
            self.send_error(404, "Not found")
            return

        content_type, _ = mimetypes.guess_type(str(resolved))
        if resolved.suffix.lower() == ".md":
            content_type = "text/plain"
        if not content_type:
            content_type = "application/octet-stream"

        try:
            file_size = os.path.getsize(resolved)
            self.send_response(200)
            if content_type.startswith("text/"):
                self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            else:
                self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(file_size))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            with open(resolved, "rb") as f:
                shutil.copyfileobj(f, self.wfile)
        except BrokenPipeError:
            # Client closed connection mid-response.
            return

    def log_message(self, format: str, *args) -> None:  # noqa: A002
        logger.info("%s - %s", self.address_string(), format % args)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Serve the ChatGPT Markdown archive locally.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the viewer in your default browser after starting.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    web_dir = repo_root / "web"
    if not web_dir.exists():
        logger.error("Missing web/ directory at %s", web_dir)
        return 2

    markdown_dir = repo_root / "markdown"
    if not markdown_dir.exists():
        logger.warning(
            "No markdown/ directory found yet. Run the converter first to generate markdown/index.md."
        )

    httpd = ArchiveServer((args.host, args.port), ArchiveHandler, repo_root=repo_root)
    url = f"http://{args.host}:{args.port}/"
    logger.info("Serving ChatGPT archive at %s", url)

    if args.open:
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception as e:
            logger.warning("Failed to open browser: %s", e)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
