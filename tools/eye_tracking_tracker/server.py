#!/usr/bin/env python3
"""Local-only tracker for the separate eye-finding/tracking AI project."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
from urllib.parse import urlsplit

WEB = Path(__file__).resolve().parent
STATUS = WEB / "status.json"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            body = STATUS.read_bytes()
            content_type = "application/json; charset=utf-8"
        elif path in ("/", "/index.html"):
            body = (WEB / "index.html").read_bytes()
            content_type = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, _format, *_args):
        pass


if __name__ == "__main__":
    host = "127.0.0.1"
    port = int(os.environ.get("EYE_TRACKING_TRACKER_PORT", "8770"))
    print(f"Eye-tracking project tracker: http://{host}:{port}/", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
