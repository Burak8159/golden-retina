"""Versioned research API for separately trained retinal disease specialists.

Each model is loaded from a local, hash-pinned RTX registry. Uploaded images are
processed in memory and are not written to application-managed storage. Scores
are uncalibrated model outputs, never clinical risk probabilities or diagnoses.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from starlette.responses import JSONResponse

MAX_BYTES = 25 * 1024 * 1024
MAX_PIXELS = 50_000_000
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_REGISTRY = Path(os.environ.get("GOLDEN_RETINA_SPECIALIST_REGISTRY", "configs/rtx/disease_specialists.local.json"))


class Score(BaseModel):
    disease: str
    source_label: str
    model_version: str
    checkpoint_sha256: str
    raw_score_0_to_1: float
    calibrated_probability: bool = False
    clinical_diagnosis: bool = False


class ScreenResponse(BaseModel):
    request_id: str
    status: str = "RESEARCH_ONLY_NOT_CLINICALLY_VALIDATED"
    image_width: int
    image_height: int
    image_quality: str = "not_assessed"
    scores: list[Score]
    limitations: list[str]


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _authorized(value: str | None) -> bool:
    token = os.environ.get("GOLDEN_RETINA_API_TOKEN", "")
    return len(token) >= 32 and value is not None and value.startswith("Bearer ") and hmac.compare_digest(value[7:].strip(), token)


def _model(architecture: str, n_outputs: int):
    from torch import nn
    from torchvision import models
    if architecture == "resnet18":
        m = models.resnet18(weights=None); m.fc = nn.Linear(m.fc.in_features, n_outputs)
    elif architecture == "resnet50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, n_outputs)
    elif architecture == "convnext_tiny":
        m = models.convnext_tiny(weights=None); m.classifier[2] = nn.Linear(m.classifier[2].in_features, n_outputs)
    elif architecture == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_outputs)
    elif architecture == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None); m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, n_outputs)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return m


def create_app(registry_path: str | Path = DEFAULT_REGISTRY) -> FastAPI:
    state: dict[str, Any] = {"models": [], "device": None, "torch": None, "lock": threading.Lock()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import torch
        from torchvision import transforms
        registry_file = Path(registry_path)
        if not registry_file.is_file():
            state.update(models=[], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch=torch)
            yield
            return
        registry = json.loads(registry_file.read_text(encoding="utf-8"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = []
        for item in registry.get("models", []):
            if item.get("enabled") is not True:
                continue
            ckpt = Path(item["checkpoint_path"])
            if not ckpt.is_file() or not hmac.compare_digest(_digest(ckpt), item["checkpoint_sha256"]):
                raise RuntimeError(f"Model checkpoint missing or hash mismatch for {item.get('disease')}")
            payload = torch.load(ckpt, map_location="cpu", weights_only=True)
            labels = item["labels"]
            model = _model(item["architecture"], len(labels))
            model.load_state_dict(payload["state_dict"], strict=True)
            model.to(device).eval()
            size = int(item["image_size"])
            if item["preprocessing"] == "resize_square":
                ops = [transforms.Resize((size, size))]
            elif item["preprocessing"] == "resize_center_crop":
                ops = [transforms.Resize(256), transforms.CenterCrop(size)]
            else:
                raise RuntimeError(f"Unsupported preprocessing for {item.get('disease')}")
            ops += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
            loaded.append({**item, "model": model, "transform": transforms.Compose(ops)})
        state.update(models=loaded, device=device, torch=torch)
        yield
        state.update(models=[])

    app = FastAPI(title="Golden Retina Disease Specialist API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.url.path in {"/v1/model-card", "/v1/screen"}:
            if not _authorized(request.headers.get("authorization")):
                return JSONResponse(status_code=503 if len(os.environ.get("GOLDEN_RETINA_API_TOKEN", "")) < 32 else 401,
                                    content={"detail": "Research API authorization required"})
        return await call_next(request)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model_count": len(state["models"]), "clinical_use": False}

    @app.get("/v1/model-card")
    def model_card(authorization: str | None = Header(default=None)):
        if not _authorized(authorization): raise HTTPException(401, "Valid bearer token required")
        return {"status": "RESEARCH_ONLY_NOT_CLINICALLY_VALIDATED", "clinical_use": False,
                "models": [{k: m.get(k) for k in ("disease", "source_label", "training_source", "evidence_boundary", "model_version", "checkpoint_sha256", "architecture", "labels", "image_size", "preprocessing")}
                           for m in state["models"]],
                "score_semantics": "raw sigmoid output; not calibrated as disease probability or clinical risk"}

    @app.post("/v1/screen", response_model=ScreenResponse)
    async def screen(request_id: str, intended_use: str = "research", image: UploadFile = File(...),
                     authorization: str | None = Header(default=None)):
        if not _authorized(authorization): raise HTTPException(401, "Valid bearer token required")
        if intended_use != "research": raise HTTPException(403, "Research-only model bundle")
        if not request_id or len(request_id) > 100: raise HTTPException(422, "request_id must contain 1–100 characters")
        payload = await image.read(MAX_BYTES + 1)
        if not payload or len(payload) > MAX_BYTES: raise HTTPException(413, "Image is empty or exceeds 25 MiB")
        try:
            with Image.open(io.BytesIO(payload)) as im:
                w, h = im.size
                if w < 1 or h < 1 or w * h > MAX_PIXELS: raise HTTPException(413, "Image exceeds 50 megapixels")
                decoded = ImageOps.exif_transpose(im).convert("RGB"); decoded.load()
        except HTTPException: raise
        except (UnidentifiedImageError, OSError, ValueError): raise HTTPException(415, "Unsupported or unreadable image")
        torch = state["torch"]
        if not state["models"]: raise HTTPException(503, "No enabled research models are configured")
        scores=[]
        with state["lock"], torch.inference_mode():
            for item in state["models"]:
                x=item["transform"](decoded).unsqueeze(0).to(state["device"])
                with torch.autocast("cuda",dtype=torch.float16,enabled=state["device"].type=="cuda"):
                    logits=item["model"](x).float().reshape(-1)
                probs=torch.sigmoid(logits).cpu().tolist()
                for label, value in zip(item["labels"], probs):
                    scores.append(Score(disease=item["disease"],source_label=label,model_version=item["model_version"],
                                        checkpoint_sha256=item["checkpoint_sha256"],raw_score_0_to_1=float(value)))
        return ScreenResponse(request_id=request_id,image_width=w,image_height=h,scores=scores,limitations=[
            "Research scores only; none is a calibrated disease probability or clinical diagnosis.",
            "Each model is source-specific and may not transfer to another camera, population, or disease spectrum.",
            "Patient-level or independent clinical validation has not been established for these development runs.",
            "Do not use to guide care; a qualified clinician must assess the image.",
        ])

    return app


app = create_app()
