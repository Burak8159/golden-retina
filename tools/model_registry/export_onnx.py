#!/usr/bin/env python3
"""Export a Golden Retina research checkpoint to an ONNX mobile artifact.

Exports are packaging only. This script does not establish camera or clinical
validity. Requires the project training dependencies plus `onnx`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def export_classifier(checkpoint: Path, output: Path, opset: int) -> dict:
    from train_rfmid_disease_expert import make_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = str(payload["architecture"])
    model = make_model(arch, pretrained=(arch == "dinov2_vits14"))
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    size = int(payload.get("size", 320))
    sample = torch.zeros(1, 3, size, size, dtype=torch.float32)
    torch.onnx.export(
        model, sample, output, input_names=["rgb_normalized"],
        output_names=["disease_logit"], opset_version=opset,
        dynamic_axes={"rgb_normalized": {0: "batch"}, "disease_logit": {0: "batch"}},
        dynamo=False,
    )
    return {
        "model_id": f"retinal_classifier_{payload['target'].lower()}",
        "task": str(payload["target"]), "architecture": arch,
        "input_size": [size, size], "layout": "NCHW", "color_space": "RGB",
        "scale": "0_to_1_then_imagenet_normalize",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "output": "one logit; sigmoid yields research score",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evidence": "RFMiD image-level test only; not clinical validation",
    }


def export_tracker(checkpoint: Path, output: Path, opset: int) -> dict:
    eye_root = ROOT / "eye_finding_ai" / "realistic_3d_v3"
    sys.path.insert(0, str(eye_root))
    from model import FaceEyeUNet

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    classes = payload.get("classes")
    if classes != ["background", "iris", "pupil"]:
        raise ValueError("Expected the frozen shared-eye three-class checkpoint")
    size = int(payload["size"])
    model = FaceEyeUNet(base=int(payload.get("base", 16)), classes=3)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    sample = torch.zeros(1, 3, size, size, dtype=torch.float32)
    torch.onnx.export(
        model, sample, output, input_names=["rgb_0_to_1"], output_names=["class_logits"],
        opset_version=opset, dynamic_axes={"rgb_0_to_1": {0: "batch"}, "class_logits": {0: "batch"}},
        dynamo=False,
    )
    return {
        "model_id": "ir_eye_tracker", "input_size": [size, size],
        "layout": "NCHW", "color_space": "RGB", "range": "0_to_1",
        "classes": classes, "postprocessing": "argmax class 2; pupil center from mask centroid",
        "source_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "evidence": "Synthetic rendered eye imagery only; not validated on IR camera frames",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("classifier", "tracker"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    meta = (export_classifier if args.kind == "classifier" else export_tracker)(
        args.checkpoint, args.output, args.opset
    )
    meta["artifact"] = args.output.name
    meta["artifact_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
