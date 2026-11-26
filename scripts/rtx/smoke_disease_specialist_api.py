#!/usr/bin/env python3
"""Launch the RTX-only research API briefly and verify authenticated inference."""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def request(url: str, token: str | None = None, data: bytes | None = None, content_type: str | None = None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if content_type:
        headers["Content-Type"] = content_type
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=90) as response:
        return response.status, response.read()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--port", type=int, default=18765)
    args = ap.parse_args()
    if not args.image.is_file() or not args.registry.is_file():
        raise FileNotFoundError("RTX smoke image or private registry is missing")
    token = secrets.token_urlsafe(40)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(args.project_root / "src")
    env["GOLDEN_RETINA_SPECIALIST_REGISTRY"] = str(args.registry.resolve())
    env["GOLDEN_RETINA_API_TOKEN"] = token
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "golden_retina.disease_specialist_api:app",
         "--host", "127.0.0.1", "--port", str(args.port), "--log-level", "error"],
        cwd=args.project_root, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        for _ in range(120):
            if proc.poll() is not None:
                raise RuntimeError(f"API process exited with status {proc.returncode}")
            try:
                code, raw = request(base + "/healthz")
                if code == 200:
                    break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        else:
            raise TimeoutError("API health check did not become ready")
        health = json.loads(raw)
        code, card_raw = request(base + "/v1/model-card", token)
        card = json.loads(card_raw)
        boundary = "----GoldenRetina" + secrets.token_hex(12)
        body = bytearray()
        def field(name: str, value: str) -> None:
            body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n".encode())
        field("request_id", "rtx-smoke-20261127")
        field("intended_use", "research")
        body.extend(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"sample.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n".encode())
        body.extend(args.image.read_bytes())
        body.extend(f"\r\n--{boundary}--\r\n".encode())
        code, result_raw = request(base + "/v1/screen?request_id=rtx-smoke-20261127",
                                   token, bytes(body), f"multipart/form-data; boundary={boundary}")
        result = json.loads(result_raw)
        if health.get("model_count") != 2 or len(result.get("scores", [])) != 2:
            raise RuntimeError("Expected two enabled specialists and two raw scores")
        if result.get("status") != "RESEARCH_ONLY_NOT_CLINICALLY_VALIDATED":
            raise RuntimeError("Research-only response boundary is absent")
        if any(s.get("calibrated_probability") or s.get("clinical_diagnosis") for s in result["scores"]):
            raise RuntimeError("API output incorrectly marks a raw score as clinical/calibrated")
        try:
            request(base + "/v1/model-card")
            raise RuntimeError("Unauthenticated model-card request unexpectedly succeeded")
        except urllib.error.HTTPError as e:
            unauth_status = e.code
        print(json.dumps({"status": "smoke_pass", "health": health,
                          "model_card": [{k: m[k] for k in ("disease", "model_version", "checkpoint_sha256")}
                                         for m in card["models"]],
                          "inference": {"http_status": code, "score_count": len(result["scores"]),
                                        "diseases": sorted({s["disease"] for s in result["scores"]}),
                                        "raw_only": True, "unauthenticated_model_card_status": unauth_status},
                          "image_bytes_sent_to_mac": False}, indent=2))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


if __name__ == "__main__":
    main()
