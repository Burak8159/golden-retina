"""Export selected Golden Retina research checkpoints to mobile ONNX bundles.

The produced bundles are for app/runtime integration and research inference.
They do not imply clinical validation or rights to redistribute the source data.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import models


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EYE_CHECKPOINT = ROOT / "models/eye_finding/charmorph_shared_eye_v2b.pt"
EXPERT_DIR = ROOT / "results/expert_sweep/checkpoints"
EXPERT_REGISTRY = ROOT / "results/expert_sweep/research_model_registry.json"
VERSION = "20261114-r4"
CONSUMER_INFERENCE_IDS = ["rfmid_dr", "rfmid_armd", "rfmid_mya", "rfmid_brvo"]
CONSUMER_INFERENCE_RULE = (
    "Post hoc research shortlist from the RFMiD image-split summary; 0.50 is a model cutoff, "
    "not a calibrated disease probability or clinical decision threshold. Results are exploratory."
)
MODEL_RAW_BASE = "https://raw.githubusercontent.com/Burak8159/golden-retina/main/mobile_models/onnx"
TARGET_NAMES = {
    "DR": "Diabetic retinopathy",
    "ARMD": "Age-related macular degeneration",
    "DN": "Drusen",
    "MYA": "Myopia",
    "BRVO": "Branch retinal vein occlusion",
    "ODC": "Optic-disc cupping finding",
    "ODP": "Optic-disc pallor finding",
    "ODE": "Optic-disc edema finding",
    "TSLN": "Tessellated fundus finding",
    "LS": "Laser-scar finding",
    "CSR": "Central serous retinopathy",
    "RS": "Retinal scar finding",
    "CRS": "Chorioretinitis",
    "CRVO": "Central retinal vein occlusion",
    "ERM": "Epiretinal membrane appearance",
    "MS": "Macular-scar finding",
    "RT": "Retinal traction finding",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EyeProbabilityModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image).softmax(dim=1)


class DiseaseProbabilityModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model(image).sigmoid()


def export(model: nn.Module, size: int, output: Path, input_name: str, output_name: str) -> None:
    model.eval()
    sample = torch.zeros((1, 3, size, size), dtype=torch.float32)
    torch.onnx.export(
        model, (sample,), str(output), input_names=[input_name], output_names=[output_name],
        opset_version=17, do_constant_folding=True, dynamo=False,
    )


def export_compact_disease_model(
    model: nn.Module, size: int, output: Path, target: str
) -> float:
    """Export FP16-weight ONNX while retaining float32 app inputs and outputs."""
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16

    fp32_path = output.with_suffix(".fp32.onnx")
    try:
        export(DiseaseProbabilityModel(model), size, fp32_path, "image", "score")
        graph = onnx.load(str(fp32_path))
        compact = convert_float_to_float16(
            graph,
            keep_io_types=True,
            # Android's CPU execution provider currently exposes Gelu only for
            # float32; leave that op in float32 and cast at its boundaries.
            op_block_list=["Gelu"],
        )
        # The ORT converter appends some inserted Cast nodes after their users.
        # Restore a stable topological order before validating and saving.
        nodes = list(compact.graph.node)
        producer = {name: index for index, node in enumerate(nodes) for name in node.output}
        indegree = [0] * len(nodes)
        followers: list[list[int]] = [[] for _ in nodes]
        for index, node in enumerate(nodes):
            dependencies = {producer[value] for value in node.input if value in producer}
            indegree[index] = len(dependencies)
            for dependency in dependencies:
                followers[dependency].append(index)
        ready = [index for index, degree in enumerate(indegree) if degree == 0]
        heapq.heapify(ready)
        ordered = []
        while ready:
            index = heapq.heappop(ready)
            ordered.append(nodes[index])
            for follower in followers[index]:
                indegree[follower] -= 1
                if indegree[follower] == 0:
                    heapq.heappush(ready, follower)
        if len(ordered) != len(nodes):
            raise RuntimeError(f"Could not topologically sort converted graph for {target}")
        del compact.graph.node[:]
        compact.graph.node.extend(ordered)
        onnx.checker.check_model(compact)
        onnx.save(compact, str(output))

        probe = torch.randn((1, 3, size, size), generator=torch.Generator().manual_seed(4815))
        reference = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
        candidate = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        expected = reference.run(["score"], {"image": probe.numpy()})[0]
        observed = candidate.run(["score"], {"image": probe.numpy()})[0]
        max_abs_error = float(np.max(np.abs(expected - observed)))
        if not np.all(np.isfinite(observed)) or not np.allclose(
            expected, observed, rtol=2e-3, atol=5e-4
        ):
            raise RuntimeError(
                f"FP16 ONNX parity failed for {target}: max_abs_error={max_abs_error}"
            )
        return max_abs_error
    finally:
        fp32_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "mobile_models/onnx")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    entries: dict[str, dict] = {}

    eye = torch.load(EYE_CHECKPOINT, map_location="cpu", weights_only=False)
    if eye.get("classes") != ["background", "iris", "pupil"]:
        raise ValueError("Unexpected eye checkpoint class contract")
    from eye_finding_ai.realistic_3d_v3.model import FaceEyeUNet

    eye_model = FaceEyeUNet(base=int(eye["base"]), classes=len(eye["classes"]))
    eye_model.load_state_dict(eye["model"], strict=True)
    eye_name = "ir_eye_tracker"
    eye_file = out / f"{eye_name}.onnx"
    export(EyeProbabilityModel(eye_model), int(eye["size"]), eye_file, "image", "class_probabilities")
    entries[eye_name] = {
        "display_name": "Eye localization · synthetic research model",
        "status": "ready", "version": VERSION, "format": "onnx", "file": eye_file.name,
        "artifact_url": f"{MODEL_RAW_BASE}/{eye_file.name}", "bytes": eye_file.stat().st_size,
        "sha256": sha256(eye_file), "input": {"name": "image", "shape": [1, 3, 384, 384],
        "color": "RGB", "range": "0..1", "resize": "square bilinear"},
        "output": {"name": "class_probabilities", "shape": [1, 3, 384, 384],
        "classes": eye["classes"]},
        "evidence": "synthetic renderer only; not validated on people, IR cameras, or goggles",
        "source_checkpoint_sha256": sha256(EYE_CHECKPOINT),
    }

    registry = json.loads(EXPERT_REGISTRY.read_text(encoding="utf-8"))
    for target_info in registry["targets"]:
        target = target_info["label"]
        architecture = target_info["selected_architecture"]
        checkpoint_path = EXPERT_DIR / f"{target}_{architecture}.pt"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint.get("target") != target or checkpoint.get("architecture") != architecture:
            raise ValueError(f"Checkpoint contract mismatch: {checkpoint_path}")
        if architecture == "convnext_tiny":
            model = models.convnext_tiny(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        elif architecture == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 1)
        elif architecture == "efficientnet_b0":
            model = models.efficientnet_b0(weights=None)
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        else:
            raise ValueError(f"Unsupported mobile architecture {architecture!r} for {target}")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        model_id = f"rfmid_{target.lower()}"
        model_file = out / f"{model_id}.onnx"
        size = int(checkpoint["size"])
        max_abs_error = export_compact_disease_model(model, size, model_file, target)
        tier = target_info.get("tier", "unknown")
        support = target_info.get("positive_image_support_train_valid_test", [])
        test = target_info.get("test_evaluation", {})
        entries[model_id] = {
            "display_name": f"{TARGET_NAMES.get(target, target + ' · RFMiD label')} · RFMiD",
            "status": "ready", "version": VERSION, "format": "onnx", "file": model_file.name,
            "artifact_url": f"{MODEL_RAW_BASE}/{model_file.name}", "bytes": model_file.stat().st_size,
            "sha256": sha256(model_file), "input": {"name": "image", "shape": [1, 3, 320, 320],
            "color": "RGB", "range": "ImageNet normalized", "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225], "resize": "square bilinear"},
            "output": {"name": "score", "shape": [1, 1], "threshold": 0.5,
            "interpretation": "uncalibrated model score; not a diagnosis"},
            "architecture": architecture,
            "weights_precision": "float16; Gelu and input/output float32",
            "evidence_tier": tier,
            "positive_images_train_validation_test": support,
            "export_parity_max_abs_error": max_abs_error,
            "development_test_accuracy_at_0_5": test.get("metrics", {}).get("accuracy_at_0_5"),
            "evidence": "RFMiD image-split research; patient independence not verified; not clinical validation",
            "source_checkpoint_sha256": sha256(checkpoint_path),
        }

    manifest = {
        "schema_version": 1, "project": "Golden Retina", "bundle_version": VERSION,
        "inference": "on-device only; the app does not upload selected images",
        "models": entries,
        "consumer_inference_ids": CONSUMER_INFERENCE_IDS,
        "consumer_inference_rule": CONSUMER_INFERENCE_RULE,
        "not_supported": {
            "glaucoma": "No glaucoma checkpoint is included in this mobile bundle. RFMiD optic-disc cupping is a finding and must not be presented as glaucoma."
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "models": {
        key: {"file": value["file"], "bytes": (out / value["file"]).stat().st_size,
              "sha256": value["sha256"]} for key, value in entries.items()
    }}, indent=2))


if __name__ == "__main__":
    main()
