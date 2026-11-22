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
    # Keep the visible location card synced with the canonical live-work field.
    # Older updates sometimes changed working_now without refreshing the card.
    working = status.get("working_now") or {}
    location = status.get("current_location") or {}
    if working.get("updated_at", "") > location.get("updated_at", ""):
        status["current_location"] = working
    budget = status.get("budget", {})
    started = budget.get("accrual_start_utc")
    rate = budget.get("conservative_hourly_rate_try")
    if budget.get("accrual_active") and started and rate is not None:
        try:
            start = datetime.fromisoformat(started.replace("Z", "+00:00"))
            if start.tzinfo is None:
                start = start.replace(tzinfo=timezone.utc)
            elapsed_hours = max(0.0, (datetime.now(timezone.utc) - start.astimezone(timezone.utc)).total_seconds() / 3600)
            campaign_limit = float(budget.get("gpu_hours_total", 0))
            used_hours = min(campaign_limit, float(budget.get("gpu_hours_base", 0)) + elapsed_hours)
            budget["gpu_hours_used"] = round(used_hours, 2)
            budget["gpu_hours_remaining"] = round(max(0.0, float(budget.get("gpu_hours_total", 8)) - used_hours), 2)
            prior_estimate = float(budget.get("prior_estimated_spend_try", 0))
            network_estimate = float(budget.get("campaign_network_estimate_try", 0))
            current_campaign_hours = max(0.0, used_hours - float(budget.get("gpu_hours_base", 0)))
            budget["estimated_spent_try"] = round(prior_estimate + current_campaign_hours * float(rate) + network_estimate, 2)
            budget["estimated_spend_scope"] = "cumulative"
            budget["display_label"] = (
                f"Estimate only: prior ₺{prior_estimate:.2f} + "
                f"{current_campaign_hours:.2f} campaign L4 hours at ₺{float(rate):.2f}/h + "
                f"provisional ₺{network_estimate:.2f} network. "
                f"Estimate ₺{budget['estimated_spent_try']:.2f}; invoice and retained-storage "
                f"charges unverified. ₺{float(budget.get('protected_reserve_try', 0)):.0f} reserve protected."
            )
        except (TypeError, ValueError):
            pass
    # These values are rendered from the live status payload, so the dashboard
    # cannot keep showing a stale VM task or an old spend estimate.
    cloud = status.setdefault("cloud", {})
    if budget.get("display_label"):
        cloud["planning_estimate"] = budget["display_label"]
    loc = status.get("current_location") or {}
    if loc.get("what"):
        cloud["gpu"] = loc["what"]
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
