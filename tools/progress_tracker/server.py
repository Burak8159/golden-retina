#!/usr/bin/env python3
"""Local-only persistent dashboard for Golden Retina AI development."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import tempfile
from datetime import datetime, timezone

WEB = Path(__file__).resolve().parent
STATUS = WEB / "status.json"


def read_status():
    status = json.loads(STATUS.read_text(encoding="utf-8"))
    budget = status.get("budget", {})
    started = budget.get("accrual_start_utc")
    rate = budget.get("conservative_hourly_rate_try")
    if budget.get("accrual_active") and started and rate is not None:
        try:
            start = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            elapsed_hours = max(0.0, (datetime.now(timezone.utc) - start.astimezone(timezone.utc)).total_seconds() / 3600)
            budget["gpu_hours_used"] = round(elapsed_hours, 2)
            budget["gpu_hours_remaining"] = round(max(0.0, float(budget.get("gpu_hours_total", 8)) - elapsed_hours), 2)
            budget["estimated_spent_try"] = round(elapsed_hours * float(rate), 2)
        except (TypeError, ValueError):
            pass
    return status


def write_status(status):
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=WEB, prefix=".status-", suffix=".tmp", delete=False
    ) as stream:
        json.dump(status, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(STATUS)


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            try:
                body = json.dumps(read_status(), ensure_ascii=False).encode("utf-8")
            except (OSError, json.JSONDecodeError):
                self.send_error(503, "Project state is unavailable")
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

    def do_POST(self):
        if self.path != "/api/status":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 1 <= length <= 1_000_000:
                self.send_error(413, "Invalid project-state payload size")
                return
            status = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(status, dict) or not isinstance(status.get("blocks"), list):
                self.send_error(400, "Expected an object with a blocks list")
                return
            write_status(status)
        except (ValueError, OSError, json.JSONDecodeError):
            self.send_error(400, "Could not save project state")
            return
        self.send_response(204)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def log_message(self, _format, *_args):
        pass


if __name__ == "__main__":
    address = ("127.0.0.1", int(os.environ.get("GOLDEN_RETINA_TRACKER_PORT", "8768")))
    print(f"Golden Retina tracker: http://{address[0]}:{address[1]}/", flush=True)
    ThreadingHTTPServer(address, Handler).serve_forever()
