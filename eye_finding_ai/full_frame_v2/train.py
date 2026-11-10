"""Train only on train split; use validation for selection; leave test untouched."""
from __future__ import annotations
import argparse,csv,json,random,time,os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from model import FullFrameUNet

NAMES=["background","left_sclera","left_iris","left_pupil","right_sclera","right_iris","right_pupil"]

class SegDataset(Dataset):
    def __init__(self,manifest,split):
        with open(manifest,newline="") as f:self.rows=[r for r in csv.DictReader(f) if r["split"]==split]
        if not self.rows: raise ValueError(f"no rows for split={split}")
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i];x=np.asarray(Image.open(r["image"]).convert("L"),dtype=np.float32)/255.;y=np.asarray(Image.open(r["mask"]).convert("L"),dtype=np.int64)
        return torch.from_numpy(x.copy()).unsqueeze(0),torch.from_numpy(y.copy())

def metrics(model,loader,device):
    inter=np.zeros(7);union=np.zeros(7);tp=np.zeros(7);den=np.zeros(7)
    model.eval()
    with torch.inference_mode():
        for x,y in loader:
            p=model(x.to(device)).argmax(1).cpu();
            for k in range(1,7):
                a=p==k;b=y==k;inter[k]+=(a&b).sum().item();union[k]+=(a|b).sum().item();tp[k]+=2*(a&b).sum().item();den[k]+=a.sum().item()+b.sum().item()
    dice=tp[1:]/np.maximum(den[1:],1);iou=inter[1:]/np.maximum(union[1:],1)
    return {"foreground_macro_dice":float(dice.mean()),"foreground_macro_iou":float(iou.mean()),
            "class_dice":{NAMES[k]:float(tp[k]/max(den[k],1)) for k in range(1,7)},"class_iou":{NAMES[k]:float(inter[k]/max(union[k],1)) for k in range(1,7)},"n_frames":len(loader.dataset)}

def dice_loss(logits,y):
    prob=logits.softmax(1); one=torch.nn.functional.one_hot(y,num_classes=7).permute(0,3,1,2).float()
    dims=(0,2,3); inter=(prob*one).sum(dims);den=prob.sum(dims)+one.sum(dims)
    return 1-((2*inter[1:]+1)/(den[1:]+1)).mean()

def atomic_save(obj,path):
    path=Path(path);tmp=path.with_suffix(path.suffix+".tmp");torch.save(obj,tmp);os.replace(tmp,path)

def main():
    p=argparse.ArgumentParser();p.add_argument("--manifest",required=True);p.add_argument("--output",default="artifacts/full_frame_v2");p.add_argument("--epochs",type=int,default=24);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--workers",type=int,default=4);p.add_argument("--seed",type=int,default=815900);p.add_argument("--patience",type=int,default=5);p.add_argument("--resume",action="store_true");a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=True;device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train=DataLoader(SegDataset(a.manifest,"train"),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=device.type=="cuda",persistent_workers=a.workers>0)
    val=DataLoader(SegDataset(a.manifest,"validation"),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=device.type=="cuda",persistent_workers=a.workers>0)
    model=FullFrameUNet().to(device);opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=1e-4)
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="max",factor=.5,patience=2,min_lr=1e-5)
    weights=torch.tensor([.08,.8,1.4,2.5,.8,1.4,2.5],device=device);scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);best=-1.;best_epoch=0;stale=0;history=[];start_epoch=1;started=time.time()
    last_path=out/"last_checkpoint.pt"
    if a.resume and last_path.is_file():
        ck=torch.load(last_path,map_location=device,weights_only=False);model.load_state_dict(ck["model"]);opt.load_state_dict(ck["optimizer"]);sched.load_state_dict(ck["scheduler"]);scaler.load_state_dict(ck["scaler"]);start_epoch=int(ck["epoch"])+1;best=float(ck["best_validation_dice"]);best_epoch=int(ck.get("best_epoch",0));stale=int(ck.get("stale_epochs",0))
        hp=out/"training_history.json"
        if hp.exists():history=json.loads(hp.read_text())
        print(json.dumps({"resume_from_epoch":start_epoch-1,"best_validation_dice":best,"stale_epochs":stale}),flush=True)
    for epoch in range(start_epoch,a.epochs+1):
        model.train();losses=[]
        for x,y in train:
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=="cuda",dtype=torch.float16):
                z=model(x);loss=nn.functional.cross_entropy(z,y,weight=weights)+.35*dice_loss(z,y)
            scaler.scale(loss).backward();scaler.unscale_(opt);nn.utils.clip_grad_norm_(model.parameters(),1.);scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        vm=metrics(model,val,device);sched.step(vm["foreground_macro_dice"])
        rec={"epoch":epoch,"train_loss":float(np.mean(losses)),"validation":vm,"lr":opt.param_groups[0]["lr"],"elapsed_min":round((time.time()-started)/60,2)};history.append(rec)
        print(json.dumps(rec),flush=True)
        if vm["foreground_macro_dice"]>best:
            best=vm["foreground_macro_dice"];best_epoch=epoch;stale=0
            atomic_save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"scaler":scaler.state_dict(),"seed":a.seed,"classes":NAMES,"best_validation_dice":best,"best_epoch":best_epoch,"stale_epochs":stale},out/"best_checkpoint.pt")
        else:stale+=1
        atomic_save({"epoch":epoch,"model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":sched.state_dict(),"scaler":scaler.state_dict(),"seed":a.seed,"classes":NAMES,"best_validation_dice":best,"best_epoch":best_epoch,"stale_epochs":stale},last_path)
        with (out/"training_history.json").open("w") as f:json.dump(history,f,indent=2)
        if stale>=a.patience:break
    summary={"status":"TRAINED_ON_SYNTHETIC_FULL_FRAME_TRAIN_SPLIT","device":str(device),"seed":a.seed,"train_frames":len(train.dataset),"validation_frames":len(val.dataset),"best_epoch":best_epoch,"best_validation_macro_dice":best,"epochs_run":len(history),"elapsed_min":round((time.time()-started)/60,2),"history_file":str(out/"training_history.json"),"checkpoint":str(out/"best_checkpoint.pt"),"resume_checkpoint":str(last_path),"test_used_for_selection":False}
    (out/"training_summary.json").write_text(json.dumps(summary,indent=2)+"\n");print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
