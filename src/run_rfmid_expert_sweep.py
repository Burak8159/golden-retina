"""Run validation-only architecture comparisons and one test pass per selected target."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path


def positives(path: Path) -> dict[str, int]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return {name: sum(int(row[name]) for row in rows) for name in rows[0] if name.lower() != "id"}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--max-total-seconds", type=int, default=None,
                   help="Optional wall-time cap; omit for budget-controlled continuation.")
    p.add_argument("--resume", action="store_true",
                   help="Reuse completed target tests and candidate validation summaries in outdir.")
    p.add_argument("--per-candidate-seconds", type=int, default=900)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--min-positive", type=int, default=30)
    p.add_argument("--min-negative", type=int, default=30)
    p.add_argument("--seed", type=int, default=20261114)
    a = p.parse_args()
    cfg = json.loads(a.config.read_text())
    root = a.data_root.resolve()
    train_csv = root / "Training_Set/Training_Set/RFMiD_Training_Labels.csv"
    valid_csv = root / "Evaluation_Set/Evaluation_Set/RFMiD_Validation_Labels.csv"
    test_csv = root / "Test_Set/Test_Set/RFMiD_Testing_Labels.csv"
    train_pos, valid_pos, test_pos = positives(train_csv), positives(valid_csv), positives(test_csv)
    train_n = sum(1 for _ in csv.DictReader(train_csv.open(newline="", encoding="utf-8-sig")))
    valid_n = sum(1 for _ in csv.DictReader(valid_csv.open(newline="", encoding="utf-8-sig")))
    test_n = sum(1 for _ in csv.DictReader(test_csv.open(newline="", encoding="utf-8-sig")))
    a.outdir.mkdir(parents=True, exist_ok=True)
    trainer = Path(__file__).with_name("train_rfmid_disease_expert.py")
    started = time.monotonic()
    summary_path = a.outdir / "sweep_summary.json"
    previous = {}
    if a.resume and summary_path.is_file():
        try:
            previous = {x["target"]: x for x in json.loads(summary_path.read_text()).get("targets", [])}
        except (ValueError, KeyError, TypeError):
            previous = {}
    records = []

    def save_progress() -> None:
        summary_path.write_text(json.dumps({"config": str(a.config),
            "elapsed_seconds": time.monotonic()-started,
            "selection": "validation AUROC only", "targets": records,
            "limitation": "RFMiD official image-level split; patient independence is not established; findings only, not clinical validation."}, indent=2)+"\n")

    for target, architectures in cfg["candidate_architectures"].items():
        prior = previous.get(target)
        if a.resume and prior and prior.get("status") in {"FINAL_TEST_COMPLETE", "SKIPPED_SUPPORT_GATE"}:
            records.append(prior)
            continue
        support = {"train_positive": train_pos.get(target, 0), "train_negative": train_n-train_pos.get(target, 0),
                   "valid_positive": valid_pos.get(target, 0), "valid_negative": valid_n-valid_pos.get(target, 0),
                   "test_positive": test_pos.get(target, 0), "test_negative": test_n-test_pos.get(target, 0)}
        gate = cfg.get("support_gate_by_target", {}).get(target, {})
        min_positive = int(gate.get("minimum_train_positive", a.min_positive))
        min_negative = int(gate.get("minimum_train_negative", a.min_negative))
        min_valid_positive = int(gate.get("minimum_validation_positive", 10))
        min_test_positive = int(gate.get("minimum_test_positive", 10))
        tier = cfg.get("target_tier", {}).get(target, "primary_support")
        if (support["train_positive"] < min_positive or support["train_negative"] < min_negative
                or support["valid_positive"] < min_valid_positive or support["test_positive"] < min_test_positive):
            records.append({"target": target, "status": "SKIPPED_SUPPORT_GATE", "support": support})
            save_progress()
            continue
        candidates = []
        for architecture in architectures:
            remaining = (None if a.max_total_seconds is None else
                         int(a.max_total_seconds - (time.monotonic()-started)))
            if remaining is not None and remaining < 120:
                break
            seconds = a.per_candidate_seconds if remaining is None else min(a.per_candidate_seconds, remaining)
            size = 224 if architecture == "dinov2_vits14" else 320
            run_dir = a.outdir / "candidates" / target / architecture
            artifact = a.outdir / "checkpoints" / f"{target}_{architecture}.pt"
            run_dir.mkdir(parents=True, exist_ok=True); artifact.parent.mkdir(parents=True, exist_ok=True)
            candidate_summary = run_dir / f"{target}_{architecture}_summary.json"
            if a.resume and artifact.is_file() and candidate_summary.is_file():
                try:
                    summary = json.loads(candidate_summary.read_text())
                    if summary.get("target") == target and summary.get("architecture") == architecture:
                        candidates.append({"architecture": architecture, "artifact": str(artifact),
                                           "validation_auroc": summary.get("best_validation_auroc"),
                                           "summary": str(candidate_summary)})
                        continue
                except (ValueError, OSError):
                    pass
            cmd = [sys.executable, str(trainer), "--data-root", str(root), "--target", target,
                   "--architecture", architecture, "--outdir", str(run_dir), "--artifact", str(artifact),
                   "--size", str(size), "--epochs", "25", "--batch-size", str(a.batch_size),
                   "--workers", str(a.workers), "--max-seconds", str(seconds),
                   "--min-positive", str(min_positive), "--min-negative", str(min_negative),
                   "--seed", str(a.seed), "--evidence-tier", tier]
            proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (run_dir / "console.log").write_text(proc.stdout)
            candidate_summary = run_dir / f"{target}_{architecture}_summary.json"
            if proc.returncode == 0 and candidate_summary.is_file() and artifact.is_file():
                summary = json.loads(candidate_summary.read_text())
                candidates.append({"architecture": architecture, "artifact": str(artifact),
                                   "validation_auroc": summary.get("best_validation_auroc"),
                                   "summary": str(candidate_summary)})
            else:
                candidates.append({"architecture": architecture, "status": "FAILED_OR_NO_EPOCH",
                                   "returncode": proc.returncode, "log": str(run_dir / "console.log")})
            save_progress()
        eligible = [x for x in candidates if isinstance(x.get("validation_auroc"), (float, int))]
        if not eligible:
            status = "WAITING_FOR_CONTINUATION" if a.max_total_seconds is not None and time.monotonic()-started >= a.max_total_seconds else "NO_VALIDATION_CANDIDATE"
            records.append({"target": target, "status": status, "tier": tier, "support": support,
                            "candidates": candidates})
            save_progress()
            continue
        winner = max(eligible, key=lambda x: x["validation_auroc"])
        remaining = (None if a.max_total_seconds is None else
                     int(a.max_total_seconds - (time.monotonic()-started)))
        if remaining is not None and remaining < 120:
            records.append({"target": target, "status": "SELECTED_VALIDATION_ONLY_TEST_NOT_RUN", "tier": tier,
                            "support": support, "selected": winner, "candidates": candidates})
            save_progress()
            continue
        final_dir = a.outdir / "final_test" / target
        final_dir.mkdir(parents=True, exist_ok=True)
        final = subprocess.run([sys.executable, str(trainer), "--data-root", str(root), "--target", target,
                                "--architecture", winner["architecture"], "--outdir", str(final_dir),
                                "--artifact", winner["artifact"], "--size", str(224 if winner["architecture"] == "dinov2_vits14" else 320),
                                "--batch-size", str(a.batch_size), "--workers", str(a.workers),
                                "--min-positive", str(min_positive), "--min-negative", str(min_negative),
                                "--evidence-tier", tier,
                                "--evaluate-checkpoint", winner["artifact"]], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (final_dir / "console.log").write_text(final.stdout)
        test_summary = final_dir / f"{target}_{winner['architecture']}_FINAL_TEST.json"
        records.append({"target": target, "status": "FINAL_TEST_COMPLETE" if final.returncode == 0 and test_summary.exists() else "FINAL_TEST_FAILED", "tier": tier,
                        "support": support, "selected_by_validation": winner, "test_summary": str(test_summary),
                        "candidates": candidates})
        save_progress()
    final_report = {"config": str(a.config), "elapsed_seconds": time.monotonic()-started,
                    "selection": "validation AUROC only", "targets": records,
                    "limitation": "RFMiD official image-level split; prior exposure and patient overlap are not resolved; findings only, not clinical validation. Low-support tier metrics are exploratory and unstable."}
    (a.outdir / "sweep_summary.json").write_text(json.dumps(final_report, indent=2)+"\n")
    print(json.dumps({"summary": str(a.outdir / "sweep_summary.json"), "elapsed_seconds": final_report["elapsed_seconds"],
                      "targets": [(x["target"],x["status"]) for x in records]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
