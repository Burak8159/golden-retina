"""Versioned, research-only image inference API for the frozen Block 21 model.

The application does not retain images; the multipart framework may briefly
spool an upload to the service's isolated temporary directory. This API is not
a clinical device and blocks clinical intended-use requests.
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


MODEL_VERSION = "golden-retina-convnext-tiny-paired-block21-2026-11-08"
MODEL_SHA256 = "61f2e891a8d71e186876fa2dec437afa73b0970f33e92412850af7c24d19c926"
RISK_THRESHOLD = 0.3177376389503479
QUALITY_THRESHOLD = 0.21102282404899597
MAX_IMAGE_BYTES = 25 * 1024 * 1024
DEFAULT_CHECKPOINT = "/data/golden-retina/workspace/models/block20_paired.pt"


class InferenceResult(BaseModel):
    request_id: str
    model_version: str
    model_sha256: str
    status: str = "RESEARCH_ONLY_NOT_CLINICALLY_VALIDATED"
    image_count: int
    grade_probabilities: list[float]
    referable_dr_risk_score: float
    ungradable_risk_score: float
    research_routing_flag: str
    clinically_actionable: bool = False
    limitations: list[str]


def check_bearer(auth_header: str | None, expected: str) -> bool:
    if not auth_header or not auth_header.startswith("Bearer "):
        return False
    return hmac.compare_digest(auth_header[7:].strip(), expected)


def research_routing_flag(risk: float, ungradable_risk: float) -> str:
    if ungradable_risk >= QUALITY_THRESHOLD:
        return "RESEARCH_QUALITY_REVIEW_FLAG"
    if risk >= RISK_THRESHOLD:
        return "RESEARCH_REFER_FLAG"
    return "RESEARCH_BELOW_THRESHOLD_FLAG"


async def decode_upload(upload: UploadFile):
    import torch

    payload = await upload.read(MAX_IMAGE_BYTES + 1)
    if not payload or len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(413, "Image is empty or exceeds the 25 MiB limit")
    try:
        with Image.open(io.BytesIO(payload)) as im:
            width, height = im.size
            if width <= 0 or height <= 0 or width * height > 50_000_000:
                raise HTTPException(413, "Image dimensions exceed the 50-megapixel limit")
            im = ImageOps.exif_transpose(im).convert("RGB")
            im.load()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(415, "Upload is not a supported, readable image")
    return im


def create_app(checkpoint: str | Path = DEFAULT_CHECKPOINT) -> FastAPI:
    state: dict[str, Any] = {"model": None, "lock": threading.Lock()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        import torch
        from golden_retina.train_mixed_view_bag_autonomy import MixedViewModel

        checkpoint_path = Path(checkpoint)
        if not checkpoint_path.is_file():
            raise RuntimeError(f"Frozen model checkpoint not found: {checkpoint_path}")
        actual_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        if not hmac.compare_digest(actual_hash, MODEL_SHA256):
            raise RuntimeError("Checkpoint hash differs from the locked Block 21 artifact")
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        args = payload["args"]
        model = MixedViewModel(
            model_name=payload["model_name"], pretrained=False,
            image_size=int(args["image_size"]),
            preserve_pretrained_risk=bool(args.get("preserve_pretrained_risk", False)),
            risk_pooling=args.get("risk_pooling", "fused"),
            risk_pool_temperature=float(args.get("risk_pool_temperature", 0.25)),
            lesion_presence_classes=0, max_views=int(args.get("max_views", 2)),
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device).eval()
        state.update(model=model, device=device, torch=torch, checkpoint_path=str(checkpoint_path))
        yield
        state.update(model=None)

    app = FastAPI(title="Golden Retina Research Inference", version="1.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def protect_upload_routes(request: Request, call_next):
        if request.url.path in {"/v1/infer", "/v1/model-card"}:
            token = os.environ.get("GOLDEN_RETINA_API_TOKEN", "")
            if len(token) < 32:
                return JSONResponse(status_code=503, content={"detail": "Service authorization is not configured"})
            if not check_bearer(request.headers.get("authorization"), token):
                return JSONResponse(status_code=401, content={"detail": "Valid bearer token required"})
        if request.url.path == "/v1/infer":
            length = request.headers.get("content-length")
            if length:
                try:
                    too_large = int(length) > 2 * MAX_IMAGE_BYTES + 256 * 1024
                except ValueError:
                    return JSONResponse(status_code=400, content={"detail": "Invalid Content-Length"})
                if too_large:
                    return JSONResponse(status_code=413, content={"detail": "Request exceeds the two-image size limit"})
        return await call_next(request)

    def authorize(header: str | None) -> None:
        token = os.environ.get("GOLDEN_RETINA_API_TOKEN", "")
        if len(token) < 32:
            raise HTTPException(503, "Service authorization is not configured")
        if not check_bearer(header, token):
            raise HTTPException(401, "Valid bearer token required")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "model_loaded": state["model"] is not None,
                "model_version": MODEL_VERSION, "clinically_actionable": False}

    @app.get("/v1/model-card")
    def model_card(authorization: str | None = Header(default=None)):
        authorize(authorization)
        return {
            "model_version": MODEL_VERSION,
            "checkpoint_sha256": MODEL_SHA256,
            "backbone": "ConvNeXt-Tiny, fine-tuned; paired-view fusion model",
            "outputs": ["five-grade DR probabilities", "referable-DR risk score", "DDR ungradable-risk score"],
            "score_semantics": {"grade_probabilities": "grades 0 through 4",
                                "ungradable_risk_score": "higher values indicate greater predicted ungradable risk"},
            "thresholds": {"research_risk": RISK_THRESHOLD, "research_quality": QUALITY_THRESHOLD},
            "status": "RESEARCH_ONLY_NOT_CLINICALLY_VALIDATED",
            "training_and_threshold_selection": "internal development sources; validation-selected",
            "external_evaluation": "Zhejiang UWF ranking only; label and domain mismatch",
            "clinically_actionable": False,
        }

    @app.post("/v1/infer", response_model=InferenceResult)
    async def infer(
        request_id: str = Form(...),
        intended_use: str = Form(default="research"),
        first_view: UploadFile = File(...),
        second_view: UploadFile | None = File(default=None),
        authorization: str | None = Header(default=None),
    ):
        authorize(authorization)
        if intended_use != "research":
            raise HTTPException(403, "This frozen model is restricted to research use")
        if not request_id or len(request_id) > 100:
            raise HTTPException(422, "request_id must contain 1–100 characters")
        model = state["model"]
        if model is None:
            raise HTTPException(503, "Model is not loaded")
        images = [await decode_upload(first_view)]
        if second_view is not None:
            images.append(await decode_upload(second_view))

        import torch
        tensors = [model.eval_transform(im) for im in images]
        tensors += [torch.zeros_like(tensors[0])] * (2 - len(tensors))
        batch = torch.stack(tensors).unsqueeze(0).to(state["device"])
        mask = torch.tensor([[True, len(images) == 2]], device=state["device"])
        with state["lock"], torch.inference_mode():
            grade_logits, risk_logit, quality_logit = model(batch, mask)
        probs = torch.softmax(grade_logits.float(), dim=-1)[0].cpu().tolist()
        risk = float(torch.sigmoid(risk_logit.float())[0].cpu())
        quality = float(torch.sigmoid(quality_logit.float())[0].cpu())
        # These flags only reproduce validation-selected research thresholds;
        # they are not diagnostic results or a certified clinical decision.
        flag = research_routing_flag(risk, quality)
        return InferenceResult(
            request_id=request_id, model_version=MODEL_VERSION, model_sha256=MODEL_SHA256,
            image_count=len(images), grade_probabilities=probs,
            referable_dr_risk_score=risk, ungradable_risk_score=quality,
            research_routing_flag=flag,
            limitations=[
                "Not clinically validated or autonomous diagnosis.",
                "Risk and quality thresholds were selected on exposed internal validation data.",
                "Single-view inference is outside the paired-view model's primary evaluation setup.",
                "Do not use to guide patient care.",
            ],
        )

    return app


app = create_app(os.environ.get("GOLDEN_RETINA_CHECKPOINT", DEFAULT_CHECKPOINT))
