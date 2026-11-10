"""Train a compact pupil/iris segmentation model on generated eye crops."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from synthetic_eye import HEIGHT, WIDTH, N_CLASSES, render_eye


class EyeDataset(Dataset):
    def __init__(self, count: int, base_seed: int, training: bool = False):
        self.count, self.base_seed, self.training = count, base_seed, training

    def __len__(self): return self.count

    def __getitem__(self, i):
        seed = self.base_seed + i
        gaze = random.Random(seed).uniform(-0.3, 0.3) if self.training else None
        image, mask, _ = render_eye(seed, sequence_offset=gaze)
        x = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        y = torch.from_numpy(mask.astype(np.int64))
        if self.training and random.Random(seed ^ 0x51A).random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.e1, self.e2, self.e3 = block(1, base), block(base, base*2), block(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.mid = block(base*4, base*8)
        self.u3, self.d3 = nn.ConvTranspose2d(base*8, base*4, 2, 2), block(base*8, base*4)
        self.u2, self.d2 = nn.ConvTranspose2d(base*4, base*2, 2, 2), block(base*4, base*2)
        self.u1, self.d1 = nn.ConvTranspose2d(base*2, base, 2, 2), block(base*2, base)
        self.out = nn.Conv2d(base, N_CLASSES, 1)

    def forward(self, x):
        a = self.e1(x); b = self.e2(self.pool(a)); c = self.e3(self.pool(b)); z = self.mid(self.pool(c))
        z = self.d3(torch.cat((self.u3(z), c), 1)); z = self.d2(torch.cat((self.u2(z), b), 1))
        z = self.d1(torch.cat((self.u1(z), a), 1))
        return self.out(z)


def metrics(model, loader, device):
    model.eval(); inter = np.zeros(N_CLASSES); union = np.zeros(N_CLASSES); dice_i = np.zeros(N_CLASSES); dice_d = np.zeros(N_CLASSES)
    center_errors, detected, n_positive = [], 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(1)
            for cls in range(1, N_CLASSES):
                p, t = pred == cls, y == cls
                inter[cls] += (p & t).sum().item(); union[cls] += (p | t).sum().item()
                dice_i[cls] += 2*(p & t).sum().item(); dice_d[cls] += p.sum().item()+t.sum().item()
            for j in range(len(y)):
                truth = torch.nonzero(y[j] == 3)
                guess = torch.nonzero(pred[j] == 3)
                if truth.numel():
                    n_positive += 1
                    if guess.numel():
                        detected += 1
                        center_errors.append(torch.linalg.vector_norm(truth.float().mean(0)-guess.float().mean(0)).item())
    return {"class_iou": {"sclera": float(inter[1]/max(union[1],1)), "iris": float(inter[2]/max(union[2],1)), "pupil": float(inter[3]/max(union[3],1))},
            "macro_dice_foreground": float(np.mean(dice_i[1:]/np.maximum(dice_d[1:],1))),
            "pupil_detection_rate": float(detected/max(n_positive,1)),
            "pupil_center_error_px_mean": float(np.mean(center_errors)) if center_errors else None,
            "pupil_center_error_px_p95": float(np.percentile(center_errors,95)) if center_errors else None,
            "n_eval_images": int(len(loader.dataset))}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',default='artifacts/eye_pupil_unet.pt'); ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--train-count',type=int,default=3000); ap.add_argument('--valid-count',type=int,default=400); ap.add_argument('--test-count',type=int,default=600)
    ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--workers',type=int,default=4); ap.add_argument('--seed',type=int,default=8159)
    args=ap.parse_args(); random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.backends.cudnn.benchmark=(device.type=='cuda')
    train=DataLoader(EyeDataset(args.train_count,args.seed,True),batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=device.type=='cuda',persistent_workers=args.workers>0)
    valid=DataLoader(EyeDataset(args.valid_count,args.seed+1_000_000),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
    test=DataLoader(EyeDataset(args.test_count,args.seed+2_000_000),batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=device.type=='cuda')
    model=TinyUNet().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=7e-4,weight_decay=1e-4)
    weights=torch.tensor([0.25,0.65,1.0,1.7],device=device); best=float('inf'); best_state=None; history=[]; started=time.time()
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True)
    for epoch in range(1,args.epochs+1):
        model.train(); losses=[]
        for x,y in train:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):
                logits=model(x); loss=nn.functional.cross_entropy(logits,y,weight=weights)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.0); optimizer.step(); losses.append(loss.item())
        vm=metrics(model,valid,device); score=1-vm['macro_dice_foreground']; rec={"epoch":epoch,"train_loss":float(np.mean(losses)),"valid":vm,"elapsed_min":round((time.time()-started)/60,2)}
        history.append(rec); print(json.dumps(rec),flush=True)
        if score<best:
            best=score; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save({"state_dict":best_state,"model":"TinyUNet","base_channels":16,"image_size":[HEIGHT,WIDTH],"classes":["background","sclera","iris","pupil"],"seed":args.seed,"epoch":epoch},out)
    model.load_state_dict(best_state); test_metrics=metrics(model,test,device)
    report={"status":"SYNTHETIC_HELD_OUT_ONLY","device":str(device),"train_count":args.train_count,"valid_count":args.valid_count,"test_count":args.test_count,"best_epoch":min(history,key=lambda x:1-x['valid']['macro_dice_foreground'])['epoch'],"history":history,"test":test_metrics,"elapsed_minutes":round((time.time()-started)/60,2),"checkpoint":str(out)}
    report_path=out.with_suffix('.metrics.json'); report_path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"final_report":str(report_path),"test":test_metrics,"status":report['status']},indent=2),flush=True)

if __name__=='__main__': main()
