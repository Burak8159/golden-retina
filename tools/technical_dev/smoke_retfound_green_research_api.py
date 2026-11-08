#!/usr/bin/env python3
"""Synthetic-only loopback contract checks for RETFound-Green research API."""
from __future__ import annotations

import argparse
import io
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image


def request(url: str, method="GET", headers=None, data=None):
    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def multipart(fields, files):
    boundary = "----GoldenRetinaResearchSmokeBoundary"
    chunks = []
    for name, value in fields.items():
        chunks += [f"--{boundary}\r\n".encode(),
                   f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
                   value.encode(), b"\r\n"]
    for name, (filename, payload, mime) in files.items():
        chunks += [f"--{boundary}\r\n".encode(),
                   f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode(),
                   f"Content-Type: {mime}\r\n\r\n".encode(), payload, b"\r\n"]
    chunks.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8768")
    ap.add_argument("--token-file", type=Path, required=True)
    args = ap.parse_args()
    env = dict(line.rstrip("\n").split("=", 1) for line in args.token_file.read_text().splitlines() if "=" in line)
    token = env["GOLDEN_RETINA_API_TOKEN"]
    assert len(token) >= 32

    status, body = request(args.base_url + "/healthz")
    health = json.loads(body)
    assert status == 200 and health["model_loaded"] is True
    assert request(args.base_url + "/v1/model-card")[0] == 401
    auth = {"Authorization": f"Bearer {token}"}
    status, body = request(args.base_url + "/v1/model-card", headers=auth)
    card = json.loads(body)
    assert status == 200 and card["checkpoint_sha256"] == "de221fc19b6e76fede71593d92363bac8d814ea3ef287b2cf586d9f97602e176"
    image = Image.new("RGB", (384, 384), (90, 100, 110))
    encoded = io.BytesIO(); image.save(encoded, format="JPEG", quality=85); jpg = encoded.getvalue()

    def infer(fields, payload, filename="synthetic.jpg"):
        mime, data = multipart(fields, {"image": (filename, payload, "image/jpeg")})
        return request(args.base_url + "/v1/infer", "POST", {**auth, "Content-Type": mime}, data)

    fields = {"request_id": "synthetic-valid-001", "intended_use": "research"}
    status, body = infer(fields, jpg)
    result = json.loads(body)
    assert status == 200 and result["status"] == "RESEARCH_ONLY_NOT_CLINICALLY_VALIDATED"
    assert result["request_id"] == fields["request_id"]
    assert result["model_version"] == "golden-retina-retfound-green-source-invariance-seed-20261111-epoch-8"
    assert len(result["grade_0_to_4_probabilities"]) == 5
    assert abs(sum(result["grade_0_to_4_probabilities"]) - 1.0) < 1e-5
    assert 0 <= result["referable_dr_score"] <= 1 and 0 <= result["hygd_glaucoma_score"] <= 1

    fields["intended_use"] = "clinical"
    assert infer(fields, jpg)[0] == 403
    assert infer({"request_id": "synthetic-corrupt-002", "intended_use": "research"}, b"not-an-image")[0] == 415
    status, _ = infer({"request_id": "synthetic-large-003", "intended_use": "research"}, b"x" * (25 * 1024 * 1024 + 1), "large.bin")
    assert status == 413
    sock = socket.create_connection(("127.0.0.1", 8768), timeout=5)
    sock.sendall(("POST /v1/infer HTTP/1.1\r\nHost: 127.0.0.1\r\n"
                  f"Authorization: Bearer {token}\r\n"
                  "Content-Type: multipart/form-data; boundary=disconnect\r\n"
                  "Content-Length: 100000\r\n\r\n--disconnect\r\n").encode())
    time.sleep(0.2)
    sock.close()
    assert request(args.base_url + "/healthz")[0] == 200
    print(json.dumps({"status": "PASS", "synthetic_only": True, "patient_data_used": False,
                      "health": 200, "auth_required": 401, "research_inference": 200,
                      "clinical_use_blocked": 403, "corrupt_upload": 415,
                      "oversized_upload": 413, "client_disconnect_recovery": True,
                      "service_recovers": True}, indent=2))


if __name__ == "__main__":
    main()
