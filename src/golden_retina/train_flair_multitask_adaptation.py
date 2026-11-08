#!/usr/bin/env python3
"""Partial fine-tuning of official FLAIR visual encoder on fixed Golden Retina splits.

Only train/validation rows from the supplied frozen manifest are consumed.
Raw row-level outputs and checkpoints are local/private artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r.get("split") in {"train", "valid"}]
    for row in rows:
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(row["image_path"])
    return rows


class FundusDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], size: int, train: bool):
        self.rows, self.size, self.train = rows, size, train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows[i]
        with Image.open(row["image_path"]) as im:
            im = im.convert("RGB")
            w, h = im.size
        scale = max(h, w) / self.size
        new_size = (max(1, int(w / scale)), max(1, int(h / scale)))
        im = im.resize(new_size, resample=Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), (0, 0, 0))
        canvas.paste(im, (0, 0))
        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1)
        if self.train:
            if random.random() < 0.5:
                x = x.flip(-1)
            # Small photometric perturbations; preserve FLAIR's [0,1] input convention.
            if random.random() < 0.5:
                gain = random.uniform(0.94, 1.06)
                bias = random.uniform(-0.025, 0.025)
                x = (x * gain + bias).clamp_(0, 1)
        return (x, int(row["dr_grade"]), int(row["ungradable"]), int(row["gon"]), i)


class DROrdinalHead(nn.Module):
    """Proportional-odds DR head with monotonically ordered grade cut points."""
    def __init__(self, width: int):
        super().__init__()
        self.score = nn.Linear(width, 1)
        self.first_threshold = nn.Parameter(torch.zeros(()))
        self.threshold_gaps = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        thresholds = torch.cat([
            self.first_threshold.view(1),
            self.first_threshold + torch.cumsum(nn.functional.softplus(self.threshold_gaps), dim=0),
        ])
        # Each column estimates P(grade > k), k = 0,...,3.
        return self.score(x) - thresholds.view(1, 4)


class MultiTask(nn.Module):
    def __init__(self, visual: nn.Module, width: int, ordinal_dr: bool = False):
        super().__init__()
        self.visual = visual
        self.dr = DROrdinalHead(width) if ordinal_dr else nn.Linear(width, 5)
        self.ordinal_dr = ordinal_dr
        self.quality = nn.Linear(width, 1)
        self.gon = nn.Linear(width, 1)

    def forward(self, x):
        z = self.visual(x)
        return self.dr(z), self.quality(z).squeeze(-1), self.gon(z).squeeze(-1)


def summarize(rows, logits, ordinal_dr: bool = False):
    from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
    dr, quality, gon = logits
    report = {}
    by_source = sorted({r["source"] for r in rows})
    for source in by_source:
        ids = np.array([i for i, r in enumerate(rows) if r["source"] == source])
        graded = ids[np.array([int(rows[i]["dr_grade"]) >= 0 for i in ids])]
        if len(graded):
            y = np.array([int(rows[i]["dr_grade"]) for i in graded])
            referable = y >= 2
            if ordinal_dr:
                cumulative = torch.sigmoid(torch.tensor(dr[graded])).numpy()
                pred = (dr[graded] >= 0).sum(axis=1)
                score = cumulative[:, 1]
            else:
                p = torch.softmax(torch.tensor(dr[graded]), dim=1).numpy()
                pred = p.argmax(1)
                score = p[:, 2:].sum(1)
            item = {"n": int(len(y)), "dr_grade_accuracy": float(accuracy_score(y, pred)),
                    "referable_accuracy_at_0_5": float(accuracy_score(referable, score >= .5))}
            if len(np.unique(referable)) == 2:
                item.update(referable_auroc=float(roc_auc_score(referable, score)),
                            referable_auprc=float(average_precision_score(referable, score)),
                            sensitivity_at_0_5=float(((score >= .5) & referable).sum() / referable.sum()),
                            specificity_at_0_5=float(((score < .5) & ~referable).sum() / (~referable).sum()))
            report[source] = item
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--flair-source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--max-seconds", type=int, default=7200)
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--resume-checkpoint", type=Path)
    ap.add_argument("--init-visual-from", type=Path,
                    help="Initialize only the visual encoder from another run; task heads start fresh.")
    ap.add_argument("--ordinal-dr-head", action="store_true",
                    help="Use a proportional-odds DR head with four ordered grade cut points.")
    ap.add_argument("--heads-only", action="store_true",
                    help="Freeze the complete FLAIR vision tower and train only the task heads.")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    sys_path = str(args.flair_source.resolve())
    import sys
    sys.path.insert(0, sys_path)
    from flair import FLAIRModel

    rows = read_rows(args.manifest)
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "valid"]
    if not tr or not va or any(r["split"] == "test" for r in rows):
        raise ValueError("Expected train/validation-only manifest rows")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model0 = FLAIRModel.from_pretrained("jusiro2/FLAIR")
    visual = model0.vision_model
    width = int(visual.out_dim)
    if args.init_visual_from:
        init_state = torch.load(args.init_visual_from, map_location="cpu", weights_only=False)["state_dict"]
        visual_state = {key.removeprefix("visual."): value
                        for key, value in init_state.items() if key.startswith("visual.")}
        if not visual_state:
            raise ValueError("No visual.* parameters were found in --init-visual-from checkpoint")
        missing, unexpected = visual.load_state_dict(visual_state, strict=False)
        if unexpected or any(key.endswith("weight") or key.endswith("bias") for key in missing):
            raise ValueError(f"Visual checkpoint mismatch; missing={missing}, unexpected={unexpected}")
    # Default: adapt layer4 and released visual projection. The heads-only ablation
    # freezes the complete foundation tower to reduce capacity after overfitting.
    for p in visual.parameters():
        p.requires_grad_(False)
    if not args.heads_only:
        for p in visual.model.layer4.parameters():
            p.requires_grad_(True)
        for p in visual.projection_head_vision.parameters():
            p.requires_grad_(True)
    model = MultiTask(visual, width, ordinal_dr=args.ordinal_dr_head).to(device)
    del model0
    start_epoch = 1
    resume_state = None
    if args.resume_checkpoint:
        resume_state = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(resume_state["state_dict"])
        start_epoch = int(resume_state.get("epoch", 0)) + 1
    # Preserve frozen BatchNorm statistics; trainable layer4 convolution weights still update.
    def set_frozen_bn_eval(m):
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()
    train_ds = FundusDataset(tr, args.size, True)
    val_ds = FundusDataset(va, args.size, False)
    counts = {}
    for r in tr:
        counts[r["source"]] = counts.get(r["source"], 0) + 1
    weights = [1.0 / counts[r["source"]] for r in tr]
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                              pin_memory=True, persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True, persistent_workers=args.workers > 0)
    optimizer_groups = []
    if not args.heads_only:
        optimizer_groups.extend([
            {"params": model.visual.model.layer4.parameters(), "lr": 1e-5},
            {"params": model.visual.projection_head_vision.parameters(), "lr": 2e-5},
        ])
    optimizer_groups.append({"params": list(model.dr.parameters()) + list(model.quality.parameters()) +
                             list(model.gon.parameters()), "lr": 5e-4})
    optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    grade_weights = torch.tensor([1., 1., 1., 1., 1.], device=device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prior_progress = {}
    if args.resume_checkpoint and (args.output_dir / "progress.json").is_file():
        prior_progress = json.loads((args.output_dir / "progress.json").read_text())
    best = float(resume_state.get("validation_loss", float("inf"))) if resume_state else float("inf")
    stale = 0
    history = list(prior_progress.get("history", []))
    started = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train(); model.apply(set_frozen_bn_eval)
        if args.heads_only:
            model.visual.eval()
        losses = []
        for x, grade, quality, gon, _ in train_loader:
            if time.time() - started >= args.max_seconds:
                break
            x=x.to(device, non_blocking=True); grade=grade.to(device); quality=quality.to(device); gon=gon.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                d, q, g = model(x)
                pieces=[]
                m=grade>=0
                if m.any():
                    if args.ordinal_dr_head:
                        cutpoints = torch.arange(4, device=device).view(1, 4)
                        ordinal_targets = (grade[m].view(-1, 1) > cutpoints).float()
                        pieces.append(nn.functional.binary_cross_entropy_with_logits(d[m], ordinal_targets))
                    else:
                        pieces.append(nn.functional.cross_entropy(d[m], grade[m], weight=grade_weights))
                m=quality>=0
                if m.any(): pieces.append(nn.functional.binary_cross_entropy_with_logits(q[m], quality[m].float()))
                m=gon>=0
                if m.any(): pieces.append(nn.functional.binary_cross_entropy_with_logits(g[m], gon[m].float()))
                loss=torch.stack(pieces).mean()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            scaler.step(optimizer); scaler.update(); losses.append(float(loss.detach().cpu()))
        model.eval(); val_losses=[]; collected=[[],[],[]]; valid_rows=[]
        with torch.inference_mode():
            for x, grade, quality, gon, indexes in val_loader:
                x=x.to(device, non_blocking=True); grade=grade.to(device); quality=quality.to(device); gon=gon.to(device)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                    d,q,g=model(x)
                parts=[]; m=grade>=0
                if m.any():
                    if args.ordinal_dr_head:
                        cutpoints = torch.arange(4, device=device).view(1, 4)
                        ordinal_targets = (grade[m].view(-1, 1) > cutpoints).float()
                        parts.append(nn.functional.binary_cross_entropy_with_logits(d[m], ordinal_targets))
                    else:
                        parts.append(nn.functional.cross_entropy(d[m],grade[m]))
                m=quality>=0
                if m.any(): parts.append(nn.functional.binary_cross_entropy_with_logits(q[m],quality[m].float()))
                m=gon>=0
                if m.any(): parts.append(nn.functional.binary_cross_entropy_with_logits(g[m],gon[m].float()))
                val_losses.append(float(torch.stack(parts).mean().cpu()))
                for dest,t in zip(collected,(d,q,g)): dest.extend(t.float().cpu().numpy())
                valid_rows.extend([va[int(i)] for i in indexes.numpy()])
        val_loss=float(np.mean(val_losses)); item={"epoch":epoch,"train_loss":float(np.mean(losses)) if losses else None,
            "validation_loss":val_loss,"validation_metrics_by_source":summarize(valid_rows,tuple(np.asarray(x) for x in collected), args.ordinal_dr_head),
            "elapsed_seconds":round(time.time()-started,1)}
        history.append(item); print(json.dumps(item),flush=True)
        if val_loss < best:
            best=val_loss;stale=0
            torch.save({"state_dict":model.state_dict(),"epoch":epoch,"validation_loss":best,"seed":args.seed,
                        "manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                        "config":vars(args)},args.output_dir/"best_checkpoint.pt")
        else: stale+=1
        (args.output_dir/"progress.json").write_text(json.dumps({"history":history,"best_epoch":epoch-stale,
            "elapsed_seconds":time.time()-started,"max_seconds":args.max_seconds},indent=2,default=str))
        if stale>=5 or time.time()-started>=args.max_seconds: break
    report={"status":"internal development train/validation only; no test or Zhejiang rows used",
        "encoder":("FLAIR ResNet-50 visual tower fully frozen; only task heads trainable" if args.heads_only
                   else "FLAIR ResNet-50 visual tower; layer4 and projection trainable; earlier backbone frozen"),
        "heads":["ordinal DR cutpoints for grades 0-4" if args.ordinal_dr_head else "DR grade 0-4",
                 "DDR ungradable where labeled","HYGD GON where labeled"],
        "loss":"masked multi-task cross-entropy/BCE; source-balanced sampling; AdamW; gradient clipping",
        "history":history,"best_epoch":min(history,key=lambda x:x['validation_loss'])['epoch'] if history else None,
        "elapsed_seconds":time.time()-started,"manifest_sha256":hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "clinical_claim":"none; exposed development splits only"}
    (args.output_dir/"run_summary.json").write_text(json.dumps(report,indent=2,default=str)+"\n")
    print(json.dumps({"status":report["status"],"best_epoch":report["best_epoch"],"elapsed_seconds":report["elapsed_seconds"]}),flush=True)


if __name__ == "__main__":
    main()
