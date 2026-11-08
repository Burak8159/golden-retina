"""Loopback-only, authenticated research inference for the frozen RETFound-Green model.

Non-commercial research use only under the applicable checkpoint terms. This
service is not a clinical diagnostic product. It never writes uploaded images.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from starlette.responses import JSONResponse


MODEL_VERSION = "golden-retina-retfound-green-source-invariance-seed-20261111-epoch-8"
MODEL_SHA256 = "de221fc19b6e76fede71593d92363bac8d814ea3ef287b2cf586d9f97602e176"
BASE_SHA256 = "431de5dbc1bebbb32f60e2c0bcf8daa4f8bcbf06f7cb1e1dc97ec589713942e1"
MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_CHECKPOINT = "/data/golden-retina/workspace/results/technical_dev_20261108_rerun/retfound_green_source_invariance_01_v2/best.pt"
DEFAULT_BASE = "/data/golden-retina/cache/retfound_green/retfoundgreen_statedict.pth"


class ResearchScores(BaseModel):
    request_id: str
    model_version: str
    checkpoint_sha256: str
    status: str = "RESEARCH_ONLY_NOT_CLINICALLY_VALIDATED"
    grade_0_to_4_probabilities: list[float]
    referable_dr_score: float
    hygd_glaucoma_score: float
    intended_use: str = "non-commercial research only"
    limitations: list[str]


def _authorized(header: str | None, expected: str) -> bool:
    return bool(header and header.startswith("Bearer ") and hmac.compare_digest(header[7:].strip(), expected))


async def _decode(file: UploadFile):
    payload = await file.read(MAX_IMAGE_BYTES + 1)
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is empty or exceeds 25 MiB")
    try:
        with Image.open(io.BytesIO(payload)) as im:
            if im.width <= 0 or im.height <= 0 or im.width * im.height > 50_000_000:
                raise HTTPException(413, "Image exceeds the 50-megapixel limit")
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.load()
            return im
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(415, "Upload is not a supported readable image")


def create_app(checkpoint: str | Path = DEFAULT_CHECKPOINT, base_checkpoint: str | Path = DEFAULT_BASE) -> FastAPI:
    state: dict[str, Any] = {"model": None, "torch": None, "transform": None, "lock": threading.Lock()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import timm
        import torch
        from torch import nn
        from torchvision.transforms import v2 as T

        class InferenceModel(nn.Module):
            def __init__(self, backbone, dim, source_count):
                super().__init__()
                self.backbone = backbone
                self.grade = nn.Linear(dim, 5)
                self.refer = nn.Linear(dim, 1)
                self.gon = nn.Linear(dim, 1)
                self.quality = None
                self.source_head = nn.Linear(dim, source_count) if source_count > 1 else None

            def forward(self, x):
                z = self.backbone(x)
                if z.ndim > 2:
                    z = z.mean(dim=1)
                return self.grade(z), self.refer(z).squeeze(-1), self.gon(z).squeeze(-1)

        cp, bp = Path(checkpoint), Path(base_checkpoint)
        if not cp.is_file() or not bp.is_file():
            raise RuntimeError("Frozen research checkpoint or base weights are unavailable")
        cp_hash = hashlib.sha256(cp.read_bytes()).hexdigest()
        bp_hash = hashlib.sha256(bp.read_bytes()).hexdigest()
        if not hmac.compare_digest(cp_hash, MODEL_SHA256) or not hmac.compare_digest(bp_hash, BASE_SHA256):
            raise RuntimeError("Checkpoint hash does not match the frozen research model card")

        backbone = timm.create_model("vit_small_patch14_reg4_dinov2", img_size=392, num_classes=0)
        base_state = torch.load(bp, map_location="cpu", weights_only=True)
        if isinstance(base_state, dict) and "model" in base_state:
            base_state = base_state["model"]
        if isinstance(base_state, dict) and "state_dict" in base_state:
            base_state = base_state["state_dict"]
        base_state = {k.removeprefix("module."): v for k, v in base_state.items() if isinstance(v, torch.Tensor)}
        missing = backbone.load_state_dict(base_state, strict=False)
        if missing.missing_keys or missing.unexpected_keys:
            raise RuntimeError("Base weights do not match RETFound-Green architecture")
        backbone.global_pool = "avg"
        payload = torch.load(cp, map_location="cpu", weights_only=True)
        source_map = payload.get("source_to_idx", {})
        model = InferenceModel(backbone, backbone.num_features, len(source_map))
        model.load_state_dict(payload["state_dict"], strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        transform = T.Compose([T.Resize((392, 392), antialias=True), T.ToImage(),
                               T.ToDtype(torch.float32, scale=True),
                               T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        state.update(model=model, device=device, torch=torch, transform=transform)
        yield
        state["model"] = None

    app = FastAPI(title="Golden Retina RETFound-Green Research API", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def auth_and_size(request: Request, call_next):
        if request.url.path in {"/v1/infer", "/v1/model-card"}:
            token = os.environ.get("GOLDEN_RETINA_API_TOKEN", "")
            if len(token) < 32:
                return JSONResponse(status_code=503, content={"detail": "Research API authorization is not configured"})
            if not _authorized(request.headers.get("authorization"), token):
                return JSONResponse(status_code=401, content={"detail": "Valid bearer token required"})
        if request.url.path == "/v1/infer":
            length = request.headers.get("content-length")
            if length:
                try:
                    if int(length) > MAX_IMAGE_BYTES + 256 * 1024:
                        return JSONResponse(status_code=413, content={"detail": "Request exceeds image size limit"})
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
        return await call_next(request)

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model_loaded": state["model"] is not None,
                "model_version": MODEL_VERSION, "clinical_use": False}

    @app.get("/v1/model-card")
    def model_card():
        return {"model_version": MODEL_VERSION, "checkpoint_sha256": MODEL_SHA256,
                "backbone": "RETFound-Green ViT-S/14 reg4 DINOv2 architecture; last block and task heads adapted",
                "outputs": ["DR grade 0-4 probabilities", "referable-DR score", "HYGD glaucoma score"],
                "status": "RESEARCH_ONLY_NOT_CLINICALLY_VALIDATED", "clinical_use": False,
                "limitations": ["Development-selected checkpoint; no independent clinical validation.",
                                 "Scores are not diagnoses and are not calibrated for clinical decisions.",
                                 "RETFound-Green terms limit use to non-commercial research; no redistribution of weights."]}

    @app.post("/v1/infer", response_model=ResearchScores)
    async def infer(request_id: str = Form(...), intended_use: str = Form(default="research"),
                    image: UploadFile = File(...), authorization: str | None = Header(default=None)):
        token = os.environ.get("GOLDEN_RETINA_API_TOKEN", "")
        if not _authorized(authorization, token):
            raise HTTPException(401, "Valid bearer token required")
        if intended_use != "research":
            raise HTTPException(403, "This checkpoint is restricted to non-commercial research use")
        if not request_id or len(request_id) > 100:
            raise HTTPException(422, "request_id must contain 1-100 characters")
        im = await _decode(image)
        torch = state["torch"]
        batch = state["transform"](im).unsqueeze(0).to(state["device"])
        with state["lock"], torch.inference_mode():
            grade, refer, gon = state["model"](batch)
        return ResearchScores(request_id=request_id, model_version=MODEL_VERSION,
                              checkpoint_sha256=MODEL_SHA256,
                              grade_0_to_4_probabilities=torch.softmax(grade.float(), -1)[0].cpu().tolist(),
                              referable_dr_score=float(torch.sigmoid(refer.float())[0].cpu()),
                              hygd_glaucoma_score=float(torch.sigmoid(gon.float())[0].cpu()),
                              limitations=["For non-commercial research only; not for patient-care decisions.",
                                            "Development validation and exploratory exposed-cohort transfer only.",
                                            "Scores are not calibrated clinical probabilities or diagnoses."])

    return app


app = create_app(os.environ.get("GOLDEN_RETINA_CHECKPOINT", DEFAULT_CHECKPOINT),
                 os.environ.get("GOLDEN_RETINA_BASE_CHECKPOINT", DEFAULT_BASE))
