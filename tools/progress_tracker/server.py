#!/usr/bin/env python3
"""Local-only persistent dashboard for Golden Retina AI development."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import tempfile
import re
import signal
import shlex
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

WEB = Path(__file__).resolve().parent
STATUS = WEB / "status.json"
DEVICE_CONFIG = WEB / "devices.local.json"
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
TRACKER_PORT = int(os.environ.get("GOLDEN_RETINA_TRACKER_PORT", "8768"))
_DEVICE_CACHE = {"at": 0.0, "value": None}


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


def ollama_request(path, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_URL}{path}", data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def llm_state():
    try:
        installed = ollama_request("/api/tags", timeout=3).get("models", [])
        loaded = ollama_request("/api/ps", timeout=3).get("models", [])
        return {
            "online": True,
            "runtime": "Ollama / MLX",
            "installed": [m.get("name") for m in installed if m.get("name")],
            "loaded": [m.get("name") for m in loaded if m.get("name")],
            "loaded_bytes": sum(int(m.get("size", 0) or 0) for m in loaded),
            "error": None,
        }
    except (OSError, ValueError, urllib.error.URLError) as exc:
        return {"online": False, "runtime": "Ollama", "installed": [], "loaded": [], "loaded_bytes": 0, "error": str(exc)[:220]}


def unload_ollama_model(model):
    # Ollama unloads a runner with an empty generate prompt and keep_alive=0.
    # There is no /api/stop endpoint in the local REST API.
    return ollama_request(
        "/api/generate",
        {"model": model, "prompt": "", "stream": False, "keep_alive": 0},
        timeout=45,
    )


def launch_ollama_terminal(model):
    state = llm_state()
    if not state.get("online"):
        raise RuntimeError("The local Ollama service is not available.")
    if model not in state.get("installed", []):
        raise ValueError("Choose an installed local model.")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", model):
        raise ValueError("The installed model name contains unsupported characters.")
    ollama = shutil.which("ollama") or "/opt/homebrew/bin/ollama"
    if not Path(ollama).is_file():
        raise RuntimeError("The Ollama CLI was not found. Install or restore the `ollama` command first.")
    command = f"{shlex.quote(ollama)} run {shlex.quote(model)}"
    applescript_command = 'tell application "Terminal" to activate\n' + (
        'tell application "Terminal" to do script "' +
        command.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ") + '"'
    )
    try:
        result = subprocess.run(
            ["/usr/bin/osascript", "-e", applescript_command],
            capture_output=True, text=True, timeout=12, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Terminal did not respond in time. Open Terminal and run the command shown below.") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not open a native Terminal session: {exc}") from exc
    if result.returncode:
        detail = (result.stderr or result.stdout or "Terminal automation was denied").strip().splitlines()[-1]
        raise RuntimeError(
            f"macOS did not let the dashboard control Terminal ({detail}). Open Terminal and run: {command}"
        )
    return {"ok": True, "command": command, "message": f"Opened the native Ollama CLI in Terminal: {command}"}


def safe_ssh_target(value):
    value = str(value or "").strip()
    if not value:
        return ""
    if value.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.@:-]{1,180}", value):
        raise ValueError("Use a hostname or user@hostname; do not enter a command or password.")
    return value


def read_device_config():
    try:
        data = json.loads(DEVICE_CONFIG.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"thinkpad_target": "", "rtx_target": "", "rtx_via_thinkpad": True}


def write_device_config(payload):
    config = {
        "thinkpad_target": safe_ssh_target(payload.get("thinkpad_target")),
        "rtx_target": safe_ssh_target(payload.get("rtx_target")),
        "rtx_via_thinkpad": bool(payload.get("rtx_via_thinkpad", True)),
    }
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=WEB, prefix=".devices-", suffix=".tmp", delete=False) as stream:
        json.dump(config, stream, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(DEVICE_CONFIG)
    _DEVICE_CACHE.update(at=0.0, value=None)
    return config


def ssh_probe(target, jump=None, gpu=False):
    if not target:
        return {"state": "not_configured", "detail": "Add this machine’s SSH target in Device connections."}
    if not shutil.which("ssh"):
        return {"state": "unavailable", "detail": "OpenSSH client is missing on the Mac."}
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ConnectionAttempts=1"]
    if jump:
        command += ["-J", jump]
    remote_command = "hostname && nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader" if gpu else "hostname"
    command += [target, remote_command]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=7, check=False)
        if result.returncode:
            if gpu:
                fallback = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=4", "-o", "ConnectionAttempts=1"]
                if jump:
                    fallback += ["-J", jump]
                fallback += [target, "hostname"]
                online = subprocess.run(fallback, capture_output=True, text=True, timeout=7, check=False)
                if online.returncode == 0:
                    hostname = online.stdout.strip().splitlines()[0] if online.stdout.strip() else target
                    return {"state": "online", "detail": f"SSH reachable · {hostname} · nvidia-smi telemetry unavailable"}
            msg = (result.stderr or "SSH probe failed").strip().splitlines()[-1]
            return {"state": "offline", "detail": msg[:180]}
        hostname = result.stdout.strip().splitlines()[0] if result.stdout.strip() else target
        if gpu:
            lines = result.stdout.strip().splitlines()
            detail = f"SSH reachable · {hostname} · GPU " + (lines[1][:160] if len(lines) > 1 else "telemetry unavailable")
        else:
            detail = f"SSH reachable · {hostname}"
        return {"state": "online", "detail": detail}
    except subprocess.TimeoutExpired:
        return {"state": "offline", "detail": "SSH probe timed out."}
    except OSError as exc:
        return {"state": "unavailable", "detail": str(exc)[:180]}


def device_state():
    if _DEVICE_CACHE["value"] is not None and time.monotonic() - _DEVICE_CACHE["at"] < 15:
        return _DEVICE_CACHE["value"]
    config = read_device_config()
    thinkpad = config.get("thinkpad_target", "")
    tailscale = "not installed"
    tailscale_bin = shutil.which("tailscale")
    if tailscale_bin:
        try:
            result = subprocess.run([tailscale_bin, "status"], capture_output=True, text=True, timeout=3, check=False)
            tailscale = "connected" if result.returncode == 0 else "unavailable"
        except (OSError, subprocess.SubprocessError):
            tailscale = "unavailable"
    value = {
        "mac": {"state": "online", "detail": "This Mac · local dashboard"},
        "thinkpad": ssh_probe(thinkpad),
        "rtx": ssh_probe(config.get("rtx_target", ""), jump=thinkpad if config.get("rtx_via_thinkpad", True) else None, gpu=True),
        "config": config,
        "tailscale": tailscale,
    }
    _DEVICE_CACHE.update(at=time.monotonic(), value=value)
    return value


def force_quit_ollama():
    unloaded = []
    unload_errors = []
    try:
        initial_state = llm_state()
        for model in initial_state.get("loaded", []):
            try:
                unload_ollama_model(model)
                unloaded.append(model)
            except (OSError, ValueError, urllib.error.URLError) as exc:
                unload_errors.append(f"{model}: {str(exc)[:120]}")
    except (OSError, ValueError, urllib.error.URLError):
        initial_state = {"online": False, "loaded": []}
    lsof = shutil.which("lsof") or "/usr/sbin/lsof"
    if not Path(lsof).exists():
        state_after = llm_state()
        model_unloaded = state_after.get("online") and not state_after.get("loaded")
        return {
            "ok": False,
            "model_unloaded": bool(model_unloaded),
            "runtime_stopped": False,
            "message": ("The model is unloaded, but the server cannot be safely stopped because macOS lsof is unavailable." if model_unloaded else "Cannot verify that the Ollama process is safe to stop because macOS lsof is unavailable."),
            "detail": "; ".join(unload_errors)[:300] if unload_errors else None,
        }
    result = subprocess.run([lsof, "-nP", "-iTCP:11434", "-sTCP:LISTEN", "-Fpcu"], capture_output=True, text=True, timeout=5, check=False)
    if result.returncode not in (0, 1):
        state_after = llm_state()
        unloaded_ok = state_after.get("online") and not state_after.get("loaded")
        return {
            "ok": False,
            "model_unloaded": bool(unloaded_ok),
            "runtime_stopped": False,
            "message": "The model was unloaded, but macOS did not allow the dashboard to inspect the Ollama process." if unloaded_ok else "macOS did not allow the dashboard to inspect the Ollama process.",
            "detail": (result.stderr or "Process inspection failed").strip()[:180],
        }
    found = []
    pid = command = uid = None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            if pid and command and uid == os.getuid() and Path(command).name.lower() == "ollama":
                found.append(int(pid))
            pid, command, uid = line[1:], None, None
        elif line.startswith("c"):
            command = line[1:]
        elif line.startswith("u"):
            try:
                uid = int(line[1:])
            except ValueError:
                uid = None
    if pid and command and uid == os.getuid() and Path(command).name.lower() == "ollama":
        found.append(int(pid))
    if not found:
        state_after = llm_state()
        note = f" Unloaded: {', '.join(unloaded)}." if unloaded else ""
        if state_after.get("online"):
            return {
                "ok": True,
                "model_unloaded": not bool(state_after.get("loaded")),
                "runtime_stopped": False,
                "message": "No Ollama server process could be verified as safe to terminate. The local model is unloaded." + note if not state_after.get("loaded") else "Ollama is still responding, and no verified listener process was found.",
            }
        return {"ok": True, "model_unloaded": not bool(initial_state.get("loaded")), "runtime_stopped": True, "message": "Ollama is already stopped; nothing needed to be terminated." + note}
    try:
        for process_id in found:
            try:
                os.kill(process_id, signal.SIGTERM)
            except PermissionError as exc:
                state_after = llm_state()
                model_unloaded = state_after.get("online") and not state_after.get("loaded")
                message = (
                    "The model was unloaded and its model memory should be released, but macOS blocked the dashboard from force-quitting Ollama. "
                    "This dashboard process is running with restricted process-control permission. Relaunch the tracker from your normal Terminal and retry to stop the Ollama server too."
                    if model_unloaded else
                    "macOS blocked the dashboard from stopping Ollama. Relaunch the tracker from your normal Terminal and retry."
                )
                return {
                    "ok": False,
                    "partial": bool(model_unloaded),
                    "model_unloaded": bool(model_unloaded),
                    "runtime_stopped": False,
                    "permission_denied": True,
                    "pid": process_id,
                    "manual_command": (
                        f"pid={process_id}; [ \"$(lsof -nP -iTCP:11434 -sTCP:LISTEN -t 2>/dev/null)\" = \"$pid\" ] "
                        f"&& [ \"$(ps -p \"$pid\" -o uid= | tr -d ' ')\" = \"{os.getuid()}\" ] "
                        "&& [ \"$(ps -p \"$pid\" -o comm= | sed 's#.*/##')\" = ollama ] "
                        "&& kill -TERM \"$pid\" || echo 'Ollama listener changed; no process was stopped.'"
                    ),
                    "message": message,
                    "detail": str(exc),
                }
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            alive = []
            for process_id in found:
                try:
                    os.kill(process_id, 0)
                    alive.append(process_id)
                except ProcessLookupError:
                    pass
            if not alive:
                break
            time.sleep(.15)
        else:
            # Recheck the exact listener PID, executable and owner before force-killing.
            remaining = subprocess.run([lsof, "-nP", "-iTCP:11434", "-sTCP:LISTEN", "-Fpcu"], capture_output=True, text=True, timeout=5, check=False).stdout
            listener = None
            row = {}
            for line in remaining.splitlines() + ["END"]:
                if line == "END" or line.startswith("p"):
                    if row.get("pid") and row.get("command") and row.get("uid") == os.getuid():
                        if Path(row["command"]).name.lower() == "ollama" and int(row["pid"]) in found:
                            listener = int(row["pid"])
                    row = {"pid": line[1:]} if line.startswith("p") else {}
                elif line.startswith("c"):
                    row["command"] = line[1:]
                elif line.startswith("u"):
                    try:
                        row["uid"] = int(line[1:])
                    except ValueError:
                        row["uid"] = None
            if listener:
                try:
                    os.kill(listener, signal.SIGKILL)
                except PermissionError as exc:
                    return {
                        "ok": False,
                        "partial": True,
                        "model_unloaded": True,
                        "runtime_stopped": False,
                        "permission_denied": True,
                        "pid": listener,
                        "manual_command": (
                            f"pid={listener}; [ \"$(lsof -nP -iTCP:11434 -sTCP:LISTEN -t 2>/dev/null)\" = \"$pid\" ] "
                            f"&& [ \"$(ps -p \"$pid\" -o uid= | tr -d ' ')\" = \"{os.getuid()}\" ] "
                            "&& [ \"$(ps -p \"$pid\" -o comm= | sed 's#.*/##')\" = ollama ] "
                            "&& kill -KILL \"$pid\" || echo 'Ollama listener changed; no process was stopped.'"
                        ),
                        "message": "The model was unloaded, but macOS blocked the final force-quit signal. Relaunch the tracker from your normal Terminal and retry.",
                        "detail": str(exc),
                    }
            else:
                state_after = llm_state()
                if state_after.get("online"):
                    return {"ok": False, "model_unloaded": not bool(state_after.get("loaded")), "runtime_stopped": False, "message": "Ollama restarted or remained online after the stop request. Its model is unloaded; check its login service if you also want the server kept off."}
                return {"ok": True, "model_unloaded": not bool(initial_state.get("loaded")), "runtime_stopped": True, "message": "Ollama stopped."}
        # Confirm the listener actually disappeared instead of reporting a signal as success.
        time.sleep(.2)
        verify = subprocess.run([lsof, "-nP", "-iTCP:11434", "-sTCP:LISTEN", "-Fpcu"], capture_output=True, text=True, timeout=5, check=False)
        if verify.returncode == 0 and any(line.startswith("p") for line in verify.stdout.splitlines()):
            state_after = llm_state()
            if state_after.get("online"):
                return {"ok": False, "model_unloaded": not bool(state_after.get("loaded")), "runtime_stopped": False, "message": "Ollama restarted or remained online after the stop request. Its model is unloaded; check its login service if you also want the server kept off."}
        note = f" Active model unloaded: {', '.join(unloaded)}." if unloaded else ""
        return {"ok": True, "model_unloaded": True, "runtime_stopped": True, "message": "Ollama server on port 11434 was stopped and verified offline." + note}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"ok": False, "message": f"Could not stop the verified Ollama process: {exc}"}


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
        if self.path == "/api/control/llm":
            body = json.dumps(llm_state(), ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/control/devices":
            body = json.dumps(device_state(), ensure_ascii=False).encode("utf-8")
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
                elif self.path == "/api/control/llm/chat":
                    self.stream_local_chat(payload)
                    return
                elif self.path == "/api/control/llm/terminal":
                    response = launch_ollama_terminal(str(payload.get("model", "")))
                elif self.path == "/api/control/llm/stop":
                    response = force_quit_ollama()
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
            except (ValueError, RuntimeError, OSError, urllib.error.URLError, subprocess.SubprocessError) as exc:
                body = json.dumps({"ok": False, "error": str(exc)[:300]}, ensure_ascii=False).encode("utf-8")
                self.send_response(400)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    def stream_local_chat(self, payload):
        state = llm_state()
        model = str(payload.get("model", ""))
        messages = payload.get("messages")
        if not state.get("online"):
            raise RuntimeError("The local Ollama service is not available.")
        if model not in state.get("installed", []):
            raise ValueError("Choose an installed local model.")
        if not isinstance(messages, list) or not 1 <= len(messages) <= 40:
            raise ValueError("Send between 1 and 40 local conversation messages.")
        safe_messages = []
        for item in messages:
            if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
                raise ValueError("Only user and assistant messages are accepted.")
            content = str(item.get("content", ""))
            if not content or len(content) > 20_000:
                raise ValueError("Each conversation message must contain 1–20,000 characters.")
            safe_messages.append({"role": item["role"], "content": content})
        body = json.dumps({"model": model, "messages": safe_messages, "stream": True, "keep_alive": "5m"}).encode("utf-8")
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=body,
            headers={"Content-Type": "application/json"},
        )
        try:
            upstream = urllib.request.urlopen(request, timeout=900)
        except urllib.error.HTTPError as exc:
            detail = exc.read(1000).decode("utf-8", "replace")
            raise RuntimeError(f"Local Ollama rejected the request: {detail[:500]}") from exc
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson; charset=utf-8")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            for line in upstream:
                if not line:
                    continue
                self.wfile.write(f"{len(line):X}\r\n".encode("ascii"))
                self.wfile.write(line)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # A user pressing Stop/closing the page closes Ollama's stream too.
            return
        except (OSError, urllib.error.URLError) as exc:
            error_line = json.dumps({"error": f"Local Ollama stream interrupted: {str(exc)[:240]}"}).encode("utf-8") + b"\n"
            try:
                self.wfile.write(f"{len(error_line):X}\r\n".encode("ascii") + error_line + b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
        finally:
            upstream.close()
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
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
    ThreadingHTTPServer(address, Handler).serve_forever()
