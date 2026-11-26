#!/usr/bin/env python3
"""Local-only persistent dashboard for Golden Retina AI development."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import atexit
import hashlib
import json
import os
import tempfile
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlsplit
from datetime import datetime, timezone

WEB = Path(__file__).resolve().parent
STATUS = WEB / "status.json"
DEVICE_CONFIG = WEB / "devices.local.json"
TRACKER_PORT = int(os.environ.get("GOLDEN_RETINA_TRACKER_PORT", "8768"))
_DEVICE_CACHE = {"at": 0.0, "value": None}
SSH_KEEPALIVE_SECONDS = 15 * 60
SSH_TRANSPORT_KEEPALIVE_SECONDS = 60
_SSH_LOCK = threading.Lock()
_KEEPALIVE_LOCK = threading.Lock()
_KEEPALIVE_STOP = threading.Event()
_KEEPALIVE_STATE = {
    "state": "not_configured",
    "interval_seconds": SSH_KEEPALIVE_SECONDS,
    "transport_keepalive_seconds": SSH_TRANSPORT_KEEPALIVE_SECONDS,
    "last_success": None,
    "detail": "Configure the RTX target to enable automatic SSH keepalive.",
}


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
                f"{current_campaign_hours:.2f} historical cloud GPU hours at ₺{float(rate):.2f}/h + "
                f"provisional ₺{network_estimate:.2f} network. "
                f"Estimate ₺{budget['estimated_spent_try']:.2f}; invoice and retained-storage "
                f"charges unverified. ₺{float(budget.get('protected_reserve_try', 0)):.0f} reserve protected."
            )
        except (TypeError, ValueError):
            pass
    # Keep the cloud campaign status separate from local/remote device state.
    # Its GPU line is a cloud-resource fact, not the current work-location text.
    cloud = status.setdefault("cloud", {})
    if budget.get("display_label"):
        cloud["planning_estimate"] = budget["display_label"]
    return status


def write_status(status):
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=WEB, prefix=".status-", suffix=".tmp", delete=False
    ) as stream:
        json.dump(status, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(STATUS)


def safe_ssh_target(value):
    value = str(value or "").strip()
    if not value:
        return ""
    if value.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.@:-]{1,180}", value):
        raise ValueError("Use a hostname or user@hostname; do not enter a command or password.")
    return value


def safe_ssh_identity(value):
    value = str(value or "").strip()
    if not value:
        return ""
    path = Path(value).expanduser().resolve()
    ssh_dir = (Path.home() / ".ssh").resolve()
    if path.parent != ssh_dir or not path.is_file():
        raise ValueError("SSH key must be an existing key file directly inside ~/.ssh.")
    return str(path)


def safe_host_key_alias(value):
    value = str(value or "").strip()
    if value and not re.fullmatch(r"[A-Za-z0-9_.:-]{1,180}", value):
        raise ValueError("Host-key alias must be a hostname or IP address.")
    return value


def read_device_config():
    try:
        data = json.loads(DEVICE_CONFIG.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                "rtx_target": safe_ssh_target(data.get("rtx_target")),
                "identity_file": safe_ssh_identity(data.get("identity_file")),
                "host_key_alias": safe_host_key_alias(data.get("host_key_alias")),
            }
    except (OSError, ValueError, json.JSONDecodeError):
        pass
    return {"rtx_target": "", "identity_file": "", "host_key_alias": ""}


def write_device_config(payload):
    existing = read_device_config()
    config = {
        "rtx_target": safe_ssh_target(payload.get("rtx_target")),
        "identity_file": safe_ssh_identity(payload.get("identity_file", existing.get("identity_file"))),
        "host_key_alias": safe_host_key_alias(payload.get("host_key_alias", existing.get("host_key_alias"))),
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=WEB, prefix=".devices-", suffix=".tmp", delete=False) as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(DEVICE_CONFIG)
    if existing.get("rtx_target") and (
        existing.get("rtx_target") != config["rtx_target"]
        or existing.get("identity_file") != config["identity_file"]
        or existing.get("host_key_alias") != config["host_key_alias"]
    ):
        close_ssh_master(existing)
    _DEVICE_CACHE.update(at=0.0, value=None)
    return config


def keepalive_snapshot():
    with _KEEPALIVE_LOCK:
        return dict(_KEEPALIVE_STATE)


def update_keepalive(state, detail, last_success=None):
    with _KEEPALIVE_LOCK:
        _KEEPALIVE_STATE.update(
            state=state,
            interval_seconds=SSH_KEEPALIVE_SECONDS,
            transport_keepalive_seconds=SSH_TRANSPORT_KEEPALIVE_SECONDS,
            detail=detail,
        )
        if last_success is not None:
            _KEEPALIVE_STATE["last_success"] = last_success


def ssh_base_options(identity_file="", host_key_alias=""):
    options = [
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=5",
        "-o", "ConnectionAttempts=1",
        "-o", f"ServerAliveInterval={SSH_TRANSPORT_KEEPALIVE_SECONDS}",
        "-o", "ServerAliveCountMax=15",
        "-o", "TCPKeepAlive=yes",
    ]
    if identity_file:
        options += ["-o", "IdentitiesOnly=yes", "-i", identity_file]
    if host_key_alias:
        options += ["-o", f"HostKeyAlias={host_key_alias}"]
    return options


def ssh_control_path(config):
    key = "|".join((config.get("rtx_target", ""), config.get("identity_file", ""), config.get("host_key_alias", "")))
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]
    return f"/tmp/gr-ssh-{os.getuid()}-{digest}.sock"


def close_ssh_master(config):
    target = config.get("rtx_target", "")
    if not target:
        return
    socket_path = ssh_control_path(config)
    command = ["ssh", *ssh_base_options(config.get("identity_file", ""), config.get("host_key_alias", "")), "-S", socket_path, "-O", "exit", target]
    try:
        subprocess.run(command, capture_output=True, text=True, timeout=4, check=False)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        Path(socket_path).unlink(missing_ok=True)
    except OSError:
        pass


def ensure_ssh_master(config):
    target = config.get("rtx_target", "")
    if not target:
        return False, "RTX SSH target is not configured."
    if not shutil.which("ssh"):
        return False, "OpenSSH client is missing on the Mac."
    socket_path = ssh_control_path(config)
    base = ssh_base_options(config.get("identity_file", ""), config.get("host_key_alias", ""))
    with _SSH_LOCK:
        check = ["ssh", *base, "-S", socket_path, "-O", "check", target]
        try:
            result = subprocess.run(check, capture_output=True, text=True, timeout=5, check=False)
            if result.returncode == 0:
                return True, "Persistent SSH master is alive."
            Path(socket_path).unlink(missing_ok=True)
            start = [
                "ssh", *base,
                "-M", "-N", "-f",
                "-o", "ControlMaster=yes",
                "-o", f"ControlPath={socket_path}",
                "-o", "ControlPersist=yes",
                target,
            ]
            started = subprocess.run(start, capture_output=True, text=True, timeout=15, check=False)
            if started.returncode:
                message = (started.stderr or "Could not start persistent SSH connection.").strip().splitlines()
                return False, (message[-1] if message else "SSH start failed.")[:180]
            check = ["ssh", *base, "-S", socket_path, "-O", "check", target]
            verified = subprocess.run(check, capture_output=True, text=True, timeout=5, check=False)
            if verified.returncode:
                return False, "SSH startup returned, but the persistent connection did not verify."
            return True, "Persistent SSH master is alive."
        except subprocess.TimeoutExpired:
            return False, "SSH connection timed out while opening the persistent session."
        except OSError as exc:
            return False, str(exc)[:180]


def run_ssh_command(config, remote_command, timeout=15):
    socket_path = ssh_control_path(config)
    command = [
        "ssh", *ssh_base_options(config.get("identity_file", ""), config.get("host_key_alias", "")),
        "-S", socket_path, "-o", "ControlMaster=auto",
        config["rtx_target"], remote_command,
    ]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)


def ssh_keepalive_once(config):
    ok, detail = ensure_ssh_master(config)
    if not ok:
        update_keepalive("reconnecting", detail)
        return False
    try:
        result = run_ssh_command(config, "hostname", timeout=12)
        if result.returncode:
            msg = (result.stderr or "Remote SSH heartbeat failed.").strip().splitlines()
            update_keepalive("reconnecting", (msg[-1] if msg else "Remote SSH heartbeat failed.")[:180])
            close_ssh_master(config)
            return False
        hostname = result.stdout.strip().splitlines()[0] if result.stdout.strip() else "RTX host"
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        update_keepalive("active", f"Automatic SSH heartbeat every 15 minutes · {hostname}", timestamp)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        update_keepalive("reconnecting", str(exc)[:180])
        close_ssh_master(config)
        return False


def ssh_keepalive_loop():
    retry_seconds = 60
    while not _KEEPALIVE_STOP.is_set():
        config = read_device_config()
        if not config.get("rtx_target"):
            update_keepalive("not_configured", "Configure the RTX target to enable automatic SSH keepalive.")
            delay = SSH_KEEPALIVE_SECONDS
        else:
            delay = SSH_KEEPALIVE_SECONDS if ssh_keepalive_once(config) else retry_seconds
        if _KEEPALIVE_STOP.wait(delay):
            break


def start_ssh_keepalive():
    thread = threading.Thread(target=ssh_keepalive_loop, name="golden-retina-ssh-keepalive", daemon=True)
    thread.start()
    return thread


def close_all_ssh_masters():
    close_ssh_master(read_device_config())


def ssh_probe(target, gpu=False, identity_file="", host_key_alias=""):
    if not target:
        return {"state": "not_configured", "detail": "Add the RTX SSH target in RTX connection settings."}
    if not shutil.which("ssh"):
        return {"state": "unavailable", "detail": "OpenSSH client is missing on the Mac."}
    remote_command = "hostname && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader" if gpu else "hostname"
    config = {
        "rtx_target": target,
        "identity_file": identity_file,
        "host_key_alias": host_key_alias,
    }
    try:
        ok, reason = ensure_ssh_master(config)
        if not ok:
            update_keepalive("reconnecting", reason)
            return {"state": "offline", "detail": reason[:180]}
        result = run_ssh_command(config, remote_command, timeout=15)
        if result.returncode:
            if gpu:
                online = run_ssh_command(config, "hostname", timeout=10)
                if online.returncode == 0:
                    hostname = online.stdout.strip().splitlines()[0] if online.stdout.strip() else target
                    ssh_keepalive_once(config)
                    return {"state": "online", "detail": f"SSH reachable · {hostname} · nvidia-smi telemetry unavailable"}
            msg = (result.stderr or "SSH probe failed").strip().splitlines()[-1]
            update_keepalive("reconnecting", msg[:180])
            close_ssh_master(config)
            return {"state": "offline", "detail": msg[:180]}
        hostname = result.stdout.strip().splitlines()[0] if result.stdout.strip() else target
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        update_keepalive("active", f"Automatic SSH heartbeat every 15 minutes · {hostname}", timestamp)
        if gpu:
            lines = result.stdout.strip().splitlines()
            detail = f"SSH reachable · {hostname} · GPU " + (lines[1][:160] if len(lines) > 1 else "telemetry unavailable")
        else:
            detail = f"SSH reachable · {hostname}"
        return {"state": "online", "detail": detail}
    except subprocess.TimeoutExpired:
        update_keepalive("reconnecting", "SSH device probe timed out; reconnect will retry automatically.")
        return {"state": "offline", "detail": "SSH probe timed out."}
    except OSError as exc:
        update_keepalive("reconnecting", str(exc)[:180])
        return {"state": "unavailable", "detail": str(exc)[:180]}


def device_state(force=False):
    if not force and _DEVICE_CACHE["value"] is not None and time.monotonic() - _DEVICE_CACHE["at"] < 15:
        return _DEVICE_CACHE["value"]
    config = read_device_config()
    tailscale = "not installed"
    tailscale_bin = shutil.which("tailscale")
    if tailscale_bin:
        try:
            result = subprocess.run([tailscale_bin, "status"], capture_output=True, text=True, timeout=3, check=False)
            tailscale = "connected" if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.SubprocessError):
            tailscale = "unavailable"
    value = {
        "mac": {"state": "online", "detail": "This Mac · Codex workstation and project dashboard"},
        "rtx": ssh_probe(
            config.get("rtx_target", ""), gpu=True,
            identity_file=config.get("identity_file", ""),
            host_key_alias=config.get("host_key_alias", ""),
        ),
        "config": config,
        "tailscale": tailscale,
        "keepalive": keepalive_snapshot(),
    }
    _DEVICE_CACHE.update(at=time.monotonic(), value=value)
    return value


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def is_local_dashboard_request(self):
        peer = self.client_address[0].split("%", 1)[0]
        if peer.startswith("::ffff:"):
            peer = peer[7:]
        if peer not in ("127.0.0.1", "::1"):
            return False
        allowed_hosts = {f"127.0.0.1:{TRACKER_PORT}", f"localhost:{TRACKER_PORT}", "127.0.0.1", "localhost"}
        if self.headers.get("Host", "").lower() not in allowed_hosts:
            return False
        origin = self.headers.get("Origin")
        allowed_origins = {f"http://127.0.0.1:{TRACKER_PORT}", f"http://localhost:{TRACKER_PORT}"}
        return not origin or origin in allowed_origins

    def do_GET(self):
        if not self.is_local_dashboard_request():
            self.send_error(403, "Local dashboard requests only")
            return
        parsed_path = urlsplit(self.path)
        if parsed_path.path == "/api/control/devices":
            force = parse_qs(parsed_path.query).get("refresh") == ["1"]
            body = json.dumps(device_state(force=force), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
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
        if not self.is_local_dashboard_request():
            self.send_error(403, "Local dashboard requests only")
            return
        if self.path.startswith("/api/control/"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 1 <= length <= 100_000:
                    self.send_error(413, "Invalid control payload size")
                    return
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                if self.path == "/api/control/devices":
                    response = {"ok": True, "config": write_device_config(payload)}
                else:
                    self.send_error(404)
                    return
                body = json.dumps(response, ensure_ascii=False).encode("utf-8")
                self.send_response(200 if response.get("ok") else 409)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
                body = json.dumps({"ok": False, "error": str(exc)[:300]}, ensure_ascii=False).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            return
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
    address = ("127.0.0.1", TRACKER_PORT)
    print(f"Golden Retina tracker: http://{address[0]}:{address[1]}/", flush=True)
    start_ssh_keepalive()
    atexit.register(close_all_ssh_masters)
    ThreadingHTTPServer(address, Handler).serve_forever()
