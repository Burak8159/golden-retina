#!/usr/bin/env python3
"""Synchronize the disease sweep's aggregate progress into the local tracker."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATUS_PATH = ROOT / "results/disease_expansion_20261114/status.json"
CONFIG_PATH = ROOT / "configs/rfmid_expert_architecture_sweep_20261114.json"
SNAPSHOT_PATH = ROOT / "results/expert_sweep/live_remote_snapshot.json"
METRICS_CSV_PATH = ROOT / "results/expert_sweep/per_disease_metrics.csv"
REGISTRY_PATH = ROOT / "results/expert_sweep/research_model_registry.json"
PROJECT = "project-41fe08b0-9696-4f8a-928"
INSTANCE = "golden-retina-disease-l4-20261114"
ZONE = "us-west1-a"

REMOTE_CODE = r'''
import glob,json,os,shlex,subprocess
root=os.path.expanduser('~/golden_retina_run/results/expert_sweep')
def read(p):
 try:
  with open(p) as f:return json.load(f)
 except (OSError,ValueError):return None
summary=read(root+'/sweep_summary.json') or {'targets':[]}
tests={}
for p in glob.glob(root+'/final_test/*/*_FINAL_TEST.json'):
 d=read(p)
 if d:tests[d.get('target')]=d
candidates={}
for p in glob.glob(root+'/candidates/*/*/*_summary.json'):
 d=read(p)
 if d:candidates.setdefault(d.get('target'),{})[d.get('architecture')]=d
ps=subprocess.check_output(['ps','-eo','pid=,ppid=,etimes=,args='],text=True)
active_candidates=[];sweep_alive=False
for line in ps.splitlines():
 try:pid,ppid,elapsed,args=line.strip().split(None,3)
 except ValueError:continue
 if 'src/run_rfmid_expert_sweep.py' in args and 'python -c' not in args:sweep_alive=True
 if 'src/train_rfmid_disease_expert.py' in args:
  try:argv=shlex.split(args)
  except ValueError:continue
  def val(flag):
   try:return argv[argv.index(flag)+1]
   except (ValueError,IndexError):return None
  target=val('--target');architecture=val('--architecture')
  # Ignore DataLoader worker command lines that inherit the script name but
  # do not carry the trainer's target/architecture arguments.
  if target and architecture:
   active_candidates.append({'target':target,'architecture':architecture,
           'elapsed_seconds':int(elapsed),'is_test':'--evaluate-checkpoint' in argv,
           'outdir':val('--outdir')})
active=max(active_candidates,key=lambda item:item['elapsed_seconds'],default=None)
print(json.dumps({'summary':summary,'tests':tests,'candidates':candidates,
                  'active':active,'sweep_alive':sweep_alive}))
'''


def pull_snapshot() -> dict:
    remote = "cd ~/golden_retina_run && venv/bin/python -c " + shlex.quote(REMOTE_CODE)
    result = subprocess.run(
        ["gcloud", "compute", "ssh", INSTANCE, f"--project={PROJECT}",
         f"--zone={ZONE}", "--tunnel-through-iap", f"--command={remote}"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[-1200:])
    for line in reversed(result.stdout.splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise RuntimeError("Remote snapshot did not return JSON")


def update_status(snapshot: dict) -> tuple[int, int, str]:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    blocks = {b["id"]: b for b in status["blocks"]}
    target_to_block = {target: i + 6 for i, target in enumerate(cfg["candidate_architectures"])}
    records = {r.get("target"): r for r in snapshot["summary"].get("targets", [])}
    candidates = snapshot.get("candidates", {})
    tests = snapshot.get("tests", {})
    active = snapshot.get("active")
    specialists = status.setdefault("per_disease_specialist_status", {})

    for target, block_id in target_to_block.items():
        block = blocks[block_id]
        rec = records.get(target)
        target_candidates = candidates.get(target, {})
        if rec and rec.get("status") == "FINAL_TEST_COMPLETE":
            selected = rec.get("selected_by_validation", {})
            arch = selected.get("architecture")
            report = tests.get(target, {})
            metrics = report.get("test_metrics", {})
            confusion = metrics.get("confusion_at_0_5", {})
            missed, positives = confusion.get("fn"), metrics.get("positive")
            fields = {
                "accuracy": metrics.get("accuracy_at_0_5"),
                "sensitivity": metrics.get("sensitivity_at_0_5"),
                "specificity": metrics.get("specificity_at_0_5"),
                "auroc": metrics.get("auroc"),
                "average_precision": metrics.get("average_precision"),
            }
            numbers = "; ".join(f"{name} {value:.4f}" for name, value in fields.items()
                                 if isinstance(value, (float, int)))
            evidence = (f"Validation selected {arch}; one RFMiD image-split test at threshold 0.5: "
                        f"{numbers}; missed positive images {missed}/{positives}. "
                        "Image-level result; patient independence and clinical validity are not established.")
            concise = (f"RFMiD {target} image-split test at threshold 0.5: "
                       + ", ".join(f"{name}={value:.4f}" for name, value in fields.items()
                                    if isinstance(value, (float, int)))
                       + f"; missed positives={missed}/{positives}; validation-selected {arch}; "
                       "patient independence and clinical validity not established.")
            if concise not in status.setdefault("evidence", []):
                status["evidence"].append(concise)
            specialists[target] = {"status": "test_complete",
                                   "status_label": f"Test complete · {arch} selected on validation",
                                   "evidence": evidence}
            block.update(status="complete", status_label=f"Complete · {arch} selected; one image-split test",
                         evidence=evidence)
        elif rec and rec.get("status") in {
            "SELECTED_VALIDATION_ONLY_TEST_NOT_RUN", "NO_VALIDATION_CANDIDATE", "SKIPPED_SUPPORT_GATE"
        }:
            note = f"{rec['status']}; no final test result was produced."
            specialists[target] = {"status": rec["status"], "status_label": rec["status"], "evidence": note}
            block.update(status="complete", status_label=f"Complete · {rec['status']}", evidence=note)
        elif active and active.get("target") == target and snapshot.get("sweep_alive"):
            val_text = "; ".join(
                f"{arch} validation AUROC {d['best_validation_auroc']:.4f}"
                for arch, d in target_candidates.items()
                if isinstance(d.get("best_validation_auroc"), (float, int))
            )
            phase = "selected test pass" if active.get("is_test") else "candidate training"
            label = f"{active.get('architecture')} · {phase}"
            evidence = (val_text + ". " if val_text else "") + f"Current: {label}. Test is not used for model selection."
            specialists[target] = {"status": "test_running" if active.get("is_test") else "active",
                                   "status_label": label, "evidence": evidence}
            block.update(status="active", status_label=f"In progress · {label}", evidence=evidence)
        elif target_candidates:
            names = "; ".join(
                f"{arch} validation AUROC {d['best_validation_auroc']:.4f}"
                for arch, d in target_candidates.items()
                if isinstance(d.get("best_validation_auroc"), (float, int))
            )
            specialists[target] = {"status": "candidate_complete", "status_label": "Candidate validation recorded",
                                   "evidence": (names + "; selection and the one test pass are pending." if names else
                                                "Candidate run recorded; no usable validation score.")}

    if active and active.get("target") in target_to_block and snapshot.get("sweep_alive"):
        target = active["target"]
        current = target_to_block[target]
        candidate_steps = []
        for arch in cfg["candidate_architectures"][target]:
            data = candidates.get(target, {}).get(arch, {})
            done = isinstance(data.get("best_validation_auroc"), (float, int))
            is_active = arch == active.get("architecture") and not active.get("is_test")
            candidate_steps.append({
                "title": f"Train {target} candidate: {arch}",
                "status": "active" if is_active else "complete" if done else "planned",
                "status_label": "Training on L4" if is_active else "Validation summary saved" if done else "Queued",
                "evidence": (f"Validation AUROC {data['best_validation_auroc']:.4f}." if done else
                             "Train/validation only; test remains closed during selection."),
            })
        selected = max(
            ((a, d.get("best_validation_auroc")) for a, d in candidates.get(target, {}).items()
             if isinstance(d.get("best_validation_auroc"), (float, int))),
            key=lambda x: x[1], default=(None, None),
        )
        candidate_steps.append({
            "title": f"Select {target} model on validation",
            "status": "complete" if active.get("is_test") else "planned",
            "status_label": f"{selected[0]} selected" if active.get("is_test") and selected[0] else "Queued",
            "evidence": f"Validation AUROC only; selected {selected[0]} ({selected[1]:.4f})." if selected[0] else "No test-set tuning.",
        })
        candidate_steps.append({
            "title": f"Evaluate selected {target} model once",
            "status": "active" if active.get("is_test") else "planned",
            "status_label": "One image-split test pass running" if active.get("is_test") else "Queued",
            "evidence": "No threshold tuning on test; report misses with specificity and accuracy.",
        })
        status["current_block"] = current
        status["current_stage_steps"] = candidate_steps
        elapsed = active.get("elapsed_seconds", 0)
        remain = max(0, 420 - elapsed)
        status["current_block_estimate"] = (
            f"{target} {active.get('architecture')} is running; nominal candidate budget has "
            f"about {remain // 60} minutes remaining before epoch-boundary completion."
        )
        status["gpu_status"] = (
            f"Golden Retina L4 running · {target} {active.get('architecture')} · "
            "one GPU, sequential per-disease sweep · RTX unused"
        )
    elif snapshot.get("sweep_alive"):
        # The runner may be between subprocesses while it writes summaries or
        # advances to the next target. Reflect that transition instead of
        # leaving the previous disease marked as current.
        unfinished = [t for t in cfg["candidate_architectures"] if t not in records]
        if unfinished:
            target = unfinished[0]
            current = target_to_block[target]
            block = blocks[current]
            status["current_block"] = current
            target_candidates = candidates.get(target, {})
            candidate_steps = []
            for arch in cfg["candidate_architectures"][target]:
                data = target_candidates.get(arch, {})
                done = isinstance(data.get("best_validation_auroc"), (float, int))
                candidate_steps.append({
                    "title": f"Train {target} candidate: {arch}",
                    "status": "complete" if done else "planned",
                    "status_label": (f"Validation saved · AUROC {data['best_validation_auroc']:.4f}"
                                     if done else "Queued in sequential sweep"),
                    "evidence": "Train/validation only; test remains closed during selection.",
                })
            detail = "; ".join(
                f"{arch} validation AUROC {d['best_validation_auroc']:.4f}"
                for arch, d in target_candidates.items()
                if isinstance(d.get("best_validation_auroc"), (float, int))
            )
            if detail:
                block.update(status="active", status_label="In progress · candidate handoff",
                             evidence=f"{detail}. Next candidate or selection is queued; no test tuning.")
            else:
                block.update(status="active", status_label="In progress · sweep advancing to next target")
            candidate_steps.extend([
                {"title": f"Select {target} architecture on validation", "status": "planned",
                 "status_label": "Queued", "evidence": "Validation AUROC only; no test-set selection."},
                {"title": f"Evaluate selected {target} model once", "status": "planned",
                 "status_label": "Queued", "evidence": "One image-split test pass after selection."},
            ])
            status["current_stage_steps"] = candidate_steps
            status["current_block_estimate"] = (
                f"Sweep process is alive and between candidate subprocesses for {target}. "
                "Each candidate has a 7-minute epoch-boundary runtime cap; sweep continuation is budget-controlled."
            )
            status["gpu_status"] = (
                f"Golden Retina L4 running · {target} candidate handoff · "
                "one GPU, sequential per-disease sweep · RTX unused"
            )
        else:
            status["current_block_estimate"] = "All configured RFMiD targets have summaries; runner is completing final bookkeeping."
            status["gpu_status"] = "Golden Retina L4 running · final sweep bookkeeping · RTX unused"
    elif active and active.get("outdir", "").endswith("dr_dinov2_validation_recovery"):
        status["current_block"] = 25
        status["current_stage_steps"] = [{
            "title": "Train and compare DR DINOv2 recovery candidate using train/validation only",
            "status": "active",
            "status_label": "Validation-only candidate training on L4",
            "evidence": "The DR test split was already used; this recovery does not evaluate or tune on it.",
        }]
        elapsed = active.get("elapsed_seconds", 0)
        remain = max(0, 420 - elapsed)
        status["current_block_estimate"] = (
            f"DR DINOv2 recovery is running on train/validation only; about {remain // 60} minutes "
            "remain in the candidate runtime cap."
        )
        status["gpu_status"] = "Golden Retina L4 running · DR DINOv2 validation-only recovery · RTX unused"
    else:
        status["gpu_status"] = "Golden Retina L4 running · disease sweep process not detected; inspect before closeout · RTX unused"

    status["completed_blocks"] = sum(b.get("status") == "complete" for b in status["blocks"])
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    stamp = now.isoformat().replace("+00:00", "Z")
    budget = status["budget"]
    previous = dt.datetime.fromisoformat(budget["as_of_utc"].replace("Z", "+00:00"))
    budget["estimated_spent_try"] = round(
        budget.get("estimated_spent_try", 0) +
        max(0, (now - previous).total_seconds()) * budget.get("accrual_rate_try_per_hour", 35.6) / 3600, 2
    )
    budget["as_of_utc"] = stamp
    status["last_update_utc"] = status["updated_at"] = stamp
    temp = STATUS_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(STATUS_PATH)
    return status["current_block"], status["completed_blocks"], status["gpu_status"]


def save_snapshot(snapshot: dict) -> None:
    """Keep the latest aggregate cloud state locally for audit and recovery."""
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
    snapshot["observed_at_utc"] = now.isoformat().replace("+00:00", "Z")
    temp = SNAPSHOT_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(SNAPSHOT_PATH)


def save_per_disease_metrics(snapshot: dict) -> None:
    """Write a compact aggregate table; omit image IDs and row-level scores."""
    def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[str, str]:
        if total <= 0:
            return "", ""
        p = successes / total
        denom = 1 + z * z / total
        center = (p + z * z / (2 * total)) / denom
        margin = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5) / denom
        return round(max(0.0, center - margin), 8), round(min(1.0, center + margin), 8)

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    records = {r.get("target"): r for r in snapshot.get("summary", {}).get("targets", [])}
    candidates = snapshot.get("candidates", {})
    tests = snapshot.get("tests", {})
    positive_support = config.get("positive_image_support", {})
    total_by_split = {"train": 1920, "valid": 640, "test": 640}
    columns = [
        "target", "tier", "status", "best_validation_architecture_so_far", "best_validation_auroc_so_far",
        "train_positive", "train_negative", "valid_positive", "valid_negative",
        "test_positive", "test_negative", "accuracy_at_0_5", "accuracy_wilson95_low",
        "accuracy_wilson95_high", "sensitivity_at_0_5", "sensitivity_wilson95_low",
        "sensitivity_wilson95_high", "specificity_at_0_5", "specificity_wilson95_low",
        "specificity_wilson95_high", "auroc", "average_precision", "missed_positive_images",
        "test_status", "evidence_boundary",
    ]
    rows = []
    for target, architectures in config.get("candidate_architectures", {}).items():
        record = records.get(target, {})
        selected = record.get("selected_by_validation", {})
        architecture = selected.get("architecture")
        candidate_map = candidates.get(target, {})
        selected_candidate = candidate_map.get(architecture, {}) if architecture else {}
        if not architecture:
            scored = [(name, item.get("best_validation_auroc"))
                      for name, item in candidate_map.items()
                      if isinstance(item.get("best_validation_auroc"), (int, float))]
            if scored:
                architecture, _ = max(scored, key=lambda item: item[1])
                selected_candidate = candidate_map[architecture]
        support = positive_support.get(target, [None, None, None])
        row = {
            "target": target,
            "tier": config.get("target_tier", {}).get(target, "unrecorded"),
            "status": record.get("status", "incomplete"),
            "best_validation_architecture_so_far": architecture or "",
            "best_validation_auroc_so_far": selected.get("validation_auroc",
                selected_candidate.get("best_validation_auroc", "")),
            "test_status": "evaluated_once" if target in tests else "not_evaluated",
            "evidence_boundary": "RFMiD image split; patient independence and clinical validity unverified",
        }
        for split, positive in zip(("train", "valid", "test"), support):
            row[f"{split}_positive"] = positive if positive is not None else ""
            row[f"{split}_negative"] = total_by_split[split] - positive if positive is not None else ""
        metrics = tests.get(target, {}).get("test_metrics", {})
        confusion = metrics.get("confusion_at_0_5", {})
        accuracy_ci = wilson(confusion.get("tp", 0) + confusion.get("tn", 0), metrics.get("n", 0))
        sensitivity_ci = wilson(confusion.get("tp", 0), metrics.get("positive", 0))
        specificity_ci = wilson(confusion.get("tn", 0), metrics.get("negative", 0))
        row.update({
            "accuracy_at_0_5": metrics.get("accuracy_at_0_5", ""),
            "accuracy_wilson95_low": accuracy_ci[0],
            "accuracy_wilson95_high": accuracy_ci[1],
            "sensitivity_at_0_5": metrics.get("sensitivity_at_0_5", ""),
            "sensitivity_wilson95_low": sensitivity_ci[0],
            "sensitivity_wilson95_high": sensitivity_ci[1],
            "specificity_at_0_5": metrics.get("specificity_at_0_5", ""),
            "specificity_wilson95_low": specificity_ci[0],
            "specificity_wilson95_high": specificity_ci[1],
            "auroc": metrics.get("auroc", ""),
            "average_precision": metrics.get("average_precision", ""),
            "missed_positive_images": metrics.get("confusion_at_0_5", {}).get("fn", ""),
        })
        rows.append(row)
    METRICS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = METRICS_CSV_PATH.with_suffix(".csv.tmp")
    with temp.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(METRICS_CSV_PATH)


def save_research_registry(snapshot: dict) -> None:
    """Materialize a versioned, non-clinical inventory from the live run."""
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    records = {r.get("target"): r for r in snapshot.get("summary", {}).get("targets", [])}
    candidates = snapshot.get("candidates", {})
    tests = snapshot.get("tests", {})
    positive_support = config.get("positive_image_support", {})
    targets = []
    for target, architectures in config.get("candidate_architectures", {}).items():
        record = records.get(target, {})
        selected = record.get("selected_by_validation", {})
        selected_architecture = selected.get("architecture")
        checkpoint_path = (ROOT / "results/expert_sweep/checkpoints"
                           / f"{target}_{selected_architecture}.pt") if selected_architecture else None
        test = tests.get(target, {})
        candidate_rows = []
        for architecture in architectures:
            candidate = candidates.get(target, {}).get(architecture, {})
            candidate_rows.append({
                "architecture": architecture,
                "best_validation_auroc": candidate.get("best_validation_auroc"),
                "candidate_checkpoint_sha256": candidate.get("checkpoint_sha256"),
                "status": candidate.get("status", "not_started_or_no_summary"),
            })
        support = positive_support.get(target, [None, None, None])
        metrics = test.get("test_metrics")
        targets.append({
            "label": target,
            "label_description": config.get("target_display", {}).get(target, target),
            "tier": config.get("target_tier", {}).get(target, "unrecorded"),
            "input_modality": "single color fundus image",
            "candidate_models": candidate_rows,
            "selection_rule": "validation AUROC only",
            "selected_architecture": selected_architecture,
            "selected_validation_auroc": selected.get("validation_auroc"),
            "test_evaluation": ({
                "split": "official RFMiD image-level test split",
                "threshold": 0.5,
                "n_images": metrics.get("n"),
                "positive_images": metrics.get("positive"),
                "metrics": {
                    key: metrics.get(key) for key in (
                        "accuracy_at_0_5", "sensitivity_at_0_5", "specificity_at_0_5",
                        "auroc", "average_precision", "confusion_at_0_5",
                    )
                },
                "checkpoint_sha256": test.get("checkpoint_sha256"),
            } if metrics else None),
            "positive_image_support_train_valid_test": support,
            "patient_ids_available": False,
            "patient_independence_verified": False,
            "intended_camera_validated": False,
            "clinical_use_eligible": False,
            "checkpoint_local": bool(checkpoint_path and checkpoint_path.is_file()),
            "status": record.get("status", "incomplete"),
        })
    registry = {
        "registry_version": "1.0",
        "project": "Golden Retina",
        "generated_at_utc": snapshot.get("observed_at_utc"),
        "dataset": config.get("dataset"),
        "dataset_revision": config.get("revision"),
        "evidence_scope": "retrospective image-split model development; not patient-level or clinical validation",
        "targets": targets,
        "separate_evidence": {
            "glaucoma_gon": "Historical internal-validation evidence only; not part of the RFMiD sweep.",
            "media_haze": "Image-visibility/quality signal, not a disease classifier.",
        },
    }
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = REGISTRY_PATH.with_suffix(".json.tmp")
    temp.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(REGISTRY_PATH)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        try:
            snapshot = pull_snapshot()
            save_snapshot(snapshot)
            save_per_disease_metrics(snapshot)
            save_research_registry(snapshot)
            current, done, gpu = update_status(snapshot)
            print(json.dumps({"current_block": current, "completed": done, "gpu": gpu}), flush=True)
        except Exception as exc:
            print(f"tracker sync deferred: {type(exc).__name__}: {exc}", flush=True)
        if args.once:
            return
        time.sleep(max(15, args.interval))


if __name__ == "__main__":
    main()
