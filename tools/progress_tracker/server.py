#!/usr/bin/env python3
"""Local-only read-only progress dashboard for Golden Retina development."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent
STATUS = ROOT / "results" / "technical_dev_20261108_rerun" / "status.json"
CANDIDATES = ROOT / "configs" / "expanded_foundation_model_candidates_20261108.json"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            try:
                status = json.loads(STATUS.read_text(encoding="utf-8"))
                status["candidate_roster"] = json.loads(CANDIDATES.read_text(encoding="utf-8")).get("candidates", [])
                body = json.dumps(status, ensure_ascii=False).encode("utf-8")
            except OSError:
                self.send_error(503, "Progress data is not available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            body = (WEB / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, _format, *_args):
        pass


if __name__ == "__main__":
    address = ("127.0.0.1", int(os.environ.get("GOLDEN_RETINA_TRACKER_PORT", "8768")))
    print(f"Golden Retina tracker: http://{address[0]}:{address[1]}/", flush=True)
    ThreadingHTTPServer(address, Handler).serve_forever()
