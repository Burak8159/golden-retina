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
from torch import nn
from torch.autograd import Function
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from train_scoped_retina_heads import (
    LESIONS,
    RetinaRows,
    ScopedRetinaNet,
    build_rows,
    deduplicate_exact_images,
    infer,
    read_csv,
    summarize,
    validation_loss,
)


class ReverseGradient(Function):
    @staticmethod
    def forward(ctx, features, coefficient):
        ctx.coefficient = coefficient
        return features.view_as(features)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.coefficient * grad_output, None


class SourceRows(Dataset):
    def __init__(self, base: RetinaRows, rows: list[dict], source_ids: dict[str, int]):
        self.base = base
        self.source_ids = [source_ids[row["source"]] for row in rows]

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        return (*self.base[index], self.source_ids[index])


def forward_with_domain(model, source_head, x, grl_coefficient: float):
    z = torch.flatten(model.encoder(x), 1)
    disease = (
        model.dr_grade(z),
        model.quality(z).squeeze(1),
        model.mmrdr_lesions(z),
        model.glaucoma(z).squeeze(1),
    )
    source_logits = source_head(ReverseGradient.apply(z, grl_coefficient))
    return disease, source_logits


def source_accuracy(model, source_head, loader, device):
    model.eval(); source_head.eval(); correct = total = 0
    with torch.inference_mode():
        for x, *_rest, source in loader:
            x = x.to(device, non_blocking=True)
            z = torch.flatten(model.encoder(x), 1)
            pred = source_head(z).argmax(dim=1).cpu()
            source = source.long().cpu()
            correct += int((pred == source).sum())
            total += len(source)
    return correct / total if total else None


def main():
    parser = argparse.ArgumentParser(description="Source-invariance ablation using gradient reversal.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--mmrdr-manifest", type=Path, default=Path("results/mmrdr_uwf_manifest_20261013.csv"))
    parser.add_argument("--mmrdr-labels", type=Path, default=Path("external/MMRDR/MMRDR-UWF/UWF.csv"))
    parser.add_argument("--hygd", type=Path, default=Path("external/HYGD_v1_1_0"))
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--domain-weight", type=float, default=0.1)
    parser.add_argument("--grl-coefficient", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--max-seconds", type=int, default=7000)
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    root = args.root.resolve()
    rows = build_rows(root, args.manifest, args.mmrdr_manifest, args.mmrdr_labels,
                      args.hygd, args.seed, include_mmrdr=False)
    rows, duplicate_audit = deduplicate_exact_images(rows)
    split_rows = {split: [row for row in rows if row["split"] == split]
                  for split in ("train", "valid", "test")}
    sources = sorted({row["source"] for row in rows})
    source_ids = {source: index for index, source in enumerate(sources)}

    for split, selected in split_rows.items():
        print(json.dumps({"split": split, "n": len(selected),
                          "sources": {source: sum(row["source"] == source for row in selected)
                                      for source in sources}}), flush=True)
    outdir = root / args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "split_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    split_hash = hashlib.sha256((outdir / "split_manifest.csv").read_bytes()).hexdigest()
    (outdir / "duplicate_audit.json").write_text(json.dumps(duplicate_audit, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The source-invariance ablation requires the L4 CUDA device")
    model = ScopedRetinaNet().to(device)
    initial = torch.load(root / args.init_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(initial["state_dict"])
    width = model.dr_grade.in_features
    source_head = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Dropout(0.1),
                                nn.Linear(256, len(sources))).to(device)
    optimizer = torch.optim.AdamW([*model.parameters(), *source_head.parameters()],
                                  lr=args.lr, weight_decay=2e-4)
    train_rows = split_rows["train"]
    counts = {source: sum(row["source"] == source for row in train_rows) for source in sources}
    sampler = WeightedRandomSampler([1.0 / counts[row["source"]] for row in train_rows],
                                    len(train_rows), replacement=True)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.workers,
              "pin_memory": True, "persistent_workers": args.workers > 0}
    train_base = RetinaRows(train_rows, args.size, True)
    train_loader = DataLoader(SourceRows(train_base, train_rows, source_ids), sampler=sampler, **kwargs)
    valid_base = RetinaRows(split_rows["valid"], args.size, False)
    valid_loader = DataLoader(valid_base, shuffle=False, **kwargs)
    valid_domain_loader = DataLoader(SourceRows(valid_base, split_rows["valid"], source_ids),
                                     shuffle=False, **kwargs)

    counts_grade = np.bincount([row["dr_grade"] for row in train_rows if row["dr_grade"] >= 0], minlength=5)
    class_weights = np.sqrt(counts_grade.sum() / np.maximum(counts_grade, 1)); class_weights /= class_weights.mean()
    dr_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, device=device, dtype=torch.float32))
    quality_loss = nn.BCEWithLogitsLoss(); lesion_loss = nn.BCEWithLogitsLoss(); gon_loss = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda")
    history = []; best = float("inf"); stale = 0; started = time.time(); timed_out = False

    for epoch in range(1, args.epochs + 1):
        model.train(); source_head.train(); epoch_losses=[]
        for x, grade, ungradable, lesions, gon, source in train_loader:
            if time.time() - started >= args.max_seconds:
                timed_out = True; break
            x=x.to(device, non_blocking=True); grade=grade.to(device); ungradable=ungradable.to(device)
            lesions=lesions.to(device); gon=gon.to(device); source=source.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                disease, source_logits = forward_with_domain(model, source_head, x, args.grl_coefficient)
                dr_logits, q_logits, lesion_logits, gon_logits = disease
                pieces=[]
                known=grade >= 0
                if known.any(): pieces.append(dr_loss(dr_logits[known], grade[known]))
                known=ungradable >= 0
                if known.any(): pieces.append(quality_loss(q_logits[known], ungradable[known].float()))
                known=(lesions >= 0).all(dim=1)
                if known.any(): pieces.append(lesion_loss(lesion_logits[known], lesions[known]))
                known=gon >= 0
                if known.any(): pieces.append(gon_loss(gon_logits[known], gon[known].float()))
                if not pieces: continue
                task_loss=torch.stack(pieces).mean()
                loss=task_loss + args.domain_weight * nn.functional.cross_entropy(source_logits, source)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            epoch_losses.append(float(loss.detach().cpu()))
        if not epoch_losses: break
        metrics=summarize(split_rows["valid"], infer(model, valid_loader, device))
        val_loss=validation_loss(model, valid_loader, device,
                                 (dr_loss, quality_loss, lesion_loss, gon_loss, True))
        item={"epoch":epoch,"train_loss":float(np.mean(epoch_losses)),
              "validation_loss":val_loss,"validation_metrics":metrics,
              "validation_source_classifier_accuracy":source_accuracy(model, source_head, valid_domain_loader, device)}
        history.append(item); print(json.dumps(item),flush=True)
        if val_loss < best:
            best=val_loss;stale=0
            torch.save({"state_dict":model.state_dict(),"source_head_state_dict":source_head.state_dict(),
                        "source_map":source_ids,"epoch":epoch,"args":vars(args),
                        "split_sha256":split_hash},root/args.outdir/"best_checkpoint.pt")
        else:
            stale+=1
            if stale>=5: break
        if timed_out or time.time()-started>=args.max_seconds:
            timed_out=True;break
    if not (root/args.outdir/"best_checkpoint.pt").is_file():
        raise RuntimeError("No source-invariant checkpoint was saved")
    report={"status":"SCOPED_DEVELOPMENT_ONLY_NOT_CLINICAL_VALIDATION",
            "internal_test_status":"not loaded or scored","test_metrics":None,
            "method":"ConvNeXt-Tiny baseline initialized; gradient-reversal source classifier; source-balanced masked multi-task loss",
            "sources":sources,"domain_weight":args.domain_weight,"grl_coefficient":args.grl_coefficient,
            "split_manifest_sha256":split_hash,"duplicate_audit":duplicate_audit,
            "elapsed_seconds":time.time()-started,"time_limit_reached":timed_out,
            "history":history,"clinical_claim":"none; exposed development validation only"}
    report["best_epoch"]=torch.load(root/args.outdir/"best_checkpoint.pt",map_location="cpu",weights_only=False)["epoch"]
    (root/args.outdir/"run_summary.json").write_text(json.dumps(report,indent=2,default=str))
    print(json.dumps({k:v for k,v in report.items() if k!="history"},indent=2,default=str),flush=True)


if __name__ == "__main__":
    main()
