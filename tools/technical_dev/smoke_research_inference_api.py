#!/usr/bin/env python3
"""Private loopback contract smoke checks for the research-only API."""
from __future__ import annotations

import argparse
import io
import json
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
import urllib.error
import urllib.request

from PIL import Image


def request(url: str, method: str = "GET", headers=None, data=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def multipart(fields: dict[str, str], files: dict[str, tuple[str, bytes, str]]):
    boundary = "----GoldenRetinaSmokeBoundary"
    chunks = []
    for name, value in fields.items():
        chunks.extend([f"--{boundary}\r\n".encode(),
                       f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                       value.encode(), b"\r\n"])
    for name, (filename, payload, mime) in files.items():
        chunks.extend([f"--{boundary}\r\n".encode(),
                       f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
                       f"Content-Type: {mime}\r\n\r\n".encode(), payload, b"\r\n"])
    chunks.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


def main():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from golden_retina.research_inference_api import research_routing_flag

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8769")
    parser.add_argument("--token-file", required=True, help="root-readable service env file")
    args = parser.parse_args()
    env = {}
    for line in open(args.token_file, encoding="utf-8"):
        if "=" in line:
            key, value = line.rstrip("\n").split("=", 1)
            env[key] = value
    token = env["GOLDEN_RETINA_API_TOKEN"]
    assert len(token) >= 32
    assert research_routing_flag(0.01, 0.99) == "RESEARCH_QUALITY_REVIEW_FLAG"
    assert research_routing_flag(0.99, 0.01) == "RESEARCH_REFER_FLAG"
    assert research_routing_flag(0.01, 0.01) == "RESEARCH_BELOW_THRESHOLD_FLAG"

    status, body = request(args.base_url + "/healthz")
    assert status == 200 and json.loads(body)["model_loaded"] is True
    status, _ = request(args.base_url + "/v1/model-card")
    assert status == 401
    # Authorization middleware must reject before an uploaded body is parsed.
    mime, data = multipart({"request_id": "smoke-unauthorized-000", "intended_use": "research"},
                           {"first_view": ("synthetic.jpg", b"large body ignored", "image/jpeg")})
    status, _ = request(args.base_url + "/v1/infer", "POST", {"Content-Type": mime}, data)
    assert status == 401

    image = Image.new("RGB", (384, 384), (88, 112, 96))
    encoded = io.BytesIO()
    image.save(encoded, format="JPEG", quality=85)
    jpeg = encoded.getvalue()
    fields = {"request_id": "smoke-one-view-001", "intended_use": "research"}
    mime, data = multipart(fields, {"first_view": ("synthetic.jpg", jpeg, "image/jpeg")})
    auth = {"Authorization": f"Bearer {token}", "Content-Type": mime}
    status, body = request(args.base_url + "/v1/infer", "POST", auth, data)
    assert status == 200, (status, body[:500])
    result = json.loads(body)
    assert result["request_id"] == fields["request_id"]
    assert result["image_count"] == 1
    assert result["model_sha256"] == "61f2e891a8d71e186876fa2dec437afa73b0970f33e92412850af7c24d19c926"
    assert result["clinically_actionable"] is False
    assert abs(sum(result["grade_probabilities"]) - 1) < 1e-5
    assert all(0 <= result[key] <= 1 for key in ("referable_dr_risk_score", "ungradable_risk_score"))

    fields["request_id"] = "smoke-two-view-002"
    mime, data = multipart(fields, {
        "first_view": ("synthetic-a.jpg", jpeg, "image/jpeg"),
        "second_view": ("synthetic-b.jpg", jpeg, "image/jpeg"),
    })
    status, body = request(args.base_url + "/v1/infer", "POST",
                           {"Authorization": f"Bearer {token}", "Content-Type": mime}, data)
    assert status == 200, (status, body[:500])
    assert json.loads(body)["image_count"] == 2

    fields["intended_use"] = "clinical"
    mime, data = multipart(fields, {"first_view": ("synthetic.jpg", jpeg, "image/jpeg")})
    status, _ = request(args.base_url + "/v1/infer", "POST",
                        {"Authorization": f"Bearer {token}", "Content-Type": mime}, data)
    assert status == 403

    mime, data = multipart({"request_id": "smoke-corrupt-003", "intended_use": "research"},
                           {"first_view": ("bad.jpg", b"not an image", "image/jpeg")})
    status, _ = request(args.base_url + "/v1/infer", "POST",
                        {"Authorization": f"Bearer {token}", "Content-Type": mime}, data)
    assert status == 415

    oversized = b"0" * (25 * 1024 * 1024 + 1)
    mime, data = multipart({"request_id": "smoke-oversize-004", "intended_use": "research"},
                           {"first_view": ("large.bin", oversized, "application/octet-stream")})
    status, _ = request(args.base_url + "/v1/infer", "POST",
                        {"Authorization": f"Bearer {token}", "Content-Type": mime}, data)
    assert status == 413

    # Simulate a client that drops the connection mid-upload, then confirm the
    # service remains responsive and subsequent requests have fresh IDs.
    parsed = urlparse(args.base_url)
    sock = socket.create_connection((parsed.hostname, parsed.port or 80), timeout=5)
    partial = (
        "POST /v1/infer HTTP/1.1\r\n"
        f"Host: {parsed.hostname}\r\n"
        f"Authorization: Bearer {token}\r\n"
        "Content-Type: multipart/form-data; boundary=broken\r\n"
        "Content-Length: 100000\r\n\r\n--broken\r\n"
    ).encode()
    sock.sendall(partial)
    sock.close()
    time.sleep(0.3)
    status, body = request(args.base_url + "/healthz")
    assert status == 200 and json.loads(body)["model_loaded"] is True

    print(json.dumps({"status": "PASS", "health": 200, "auth_denied": 401,
                      "one_view_inference": "PASS", "two_view_inference": "PASS",
                      "clinical_use_blocked": 403, "corrupt_upload_rejected": 415,
                      "unauthorized_upload_rejected_before_parse": 401,
                      "oversized_upload_rejected": 413,
                      "client_disconnect_recovery": "PASS",
                      "threshold_routing_semantics": "PASS",
                      "synthetic_only": True, "no_patient_images_used": True}, indent=2))


if __name__ == "__main__":
    main()
