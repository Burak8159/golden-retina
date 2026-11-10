"""Run render -> subject-split audit -> training -> locked synthetic test -> inference."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd):
    print("RUN", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), check=True)


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); p.add_argument("--blender", default="blender")
    p.add_argument("--train", type=int, default=1000); p.add_argument("--validation", type=int, default=150); p.add_argument("--test", type=int, default=200)
    p.add_argument("--samples", type=int, default=8); p.add_argument("--epochs", type=int, default=20); p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=6); p.add_argument("--size", type=int, default=384); p.add_argument("--seed", type=int, default=815911)
    p.add_argument("--sequences", type=int, default=6); p.add_argument("--sequence-frames", type=int, default=24)
    a = p.parse_args(); out = Path(a.output).resolve(); data = out/"data"; model = out/"model"
    root = Path(__file__).resolve().parents[2]
    render = root/"eye_finding_ai/realistic_3d_v3/render_dataset.py"
    train = root/"eye_finding_ai/realistic_3d_v3/train.py"
    infer = root/"eye_finding_ai/realistic_3d_v3/infer.py"
    render_sequences = root/"eye_finding_ai/realistic_3d_v3/render_sequences.py"
    evaluate_sequences = root/"eye_finding_ai/realistic_3d_v3/evaluate_sequences.py"
    audit = root/"eye_finding_ai/realistic_3d_v3/audit_dataset.py"
    run([a.blender, "-b", "--python", render, "--", "--output", data, "--train", a.train, "--validation", a.validation,
         "--test", a.test, "--samples", a.samples, "--seed", 815900])
    run([sys.executable, audit, "--manifest", data/"manifest.csv", "--output", out/"data_audit.json"])
    run([sys.executable, train, "--manifest", data/"manifest.csv", "--output", model, "--epochs", a.epochs,
         "--batch-size", a.batch_size, "--workers", a.workers, "--size", a.size, "--seed", a.seed, "--test-after-freeze"])
    rows = list(__import__("csv").DictReader((data/"manifest.csv").open()))
    test_image = next(r["image"] for r in rows if r["split"] == "test")
    run([sys.executable, infer, "--checkpoint", model/"best_checkpoint.pt", "--image", test_image, "--output-dir", out/"inference_smoke"])
    seqdir = out/"sequences"
    run([a.blender, "-b", "--python", render_sequences, "--", "--output", seqdir, "--sequences", a.sequences,
         "--frames", a.sequence_frames, "--samples", a.samples])
    run([sys.executable, evaluate_sequences, "--manifest", seqdir/"sequence_manifest.csv", "--checkpoint", model/"best_checkpoint.pt",
         "--output", out/"sequence_test_metrics.json"])
    report = {"status": "SYNTHETIC_PIPELINE_COMPLETE", "renderer": str(render), "data": str(data), "data_audit": str(out/"data_audit.json"), "model": str(model),
              "training_summary": str(model/"training_summary.json"), "locked_test_metrics": str(model/"locked_test_metrics.json"),
              "inference_smoke": str(out/"inference_smoke/observation.json"), "sequence_metrics": str(out/"sequence_test_metrics.json")}
    (out/"pipeline_report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
