"""Train full-frame, two-eye/pupil segmentation on deterministic synthetic faces."""
import argparse,json,random,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from torch.utils.data import DataLoader,Dataset
from synthetic_face import H,W,render_face,centers_and_boxes

class FaceDataset(Dataset):
    def __init__(self,n,seed):self.n=n;self.seed=seed
    def __len__(self):return self.n
    def __getitem__(self,i):
        gray,mask=render_face(self.seed+i)
        if (self.seed+i)%2: gray=np.ascontiguousarray(gray[:,::-1]);mask=np.ascontiguousarray(np.where(mask==0,0,np.where(mask<4,mask+3,mask-3))[:,::-1])
        return torch.from_numpy(gray.copy()).float().unsqueeze(0)/255.,torch.from_numpy(mask.astype(np.int64,copy=False))

def block(a,b):return nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(True),nn.Conv2d(b,b,3,padding=1,bias=False),nn.BatchNorm2d(b),nn.ReLU(True))
class DualEyeUNet(nn.Module):
    def __init__(self,base=12):
        super().__init__();self.e1=block(1,base);self.e2=block(base,base*2);self.e3=block(base*2,base*4);self.pool=nn.MaxPool2d(2);self.mid=block(base*4,base*8)
        self.u3=nn.ConvTranspose2d(base*8,base*4,2,2);self.d3=block(base*8,base*4);self.u2=nn.ConvTranspose2d(base*4,base*2,2,2);self.d2=block(base*4,base*2);self.u1=nn.ConvTranspose2d(base*2,base,2,2);self.d1=block(base*2,base);self.out=nn.Conv2d(base,7,1)
    def forward(self,x):
        a=self.e1(x);b=self.e2(self.pool(a));c=self.e3(self.pool(b));z=self.mid(self.pool(c));z=self.d3(torch.cat((self.u3(z),c),1));z=self.d2(torch.cat((self.u2(z),b),1));z=self.d1(torch.cat((self.u1(z),a),1));return self.out(z)

def evaluate(model,loader,device):
    model.eval();inter=np.zeros(7);union=np.zeros(7);di=np.zeros(7);dd=np.zeros(7);ce={'frame_left':[],'frame_right':[]};det={'frame_left':0,'frame_right':0};available={'frame_left':0,'frame_right':0}
    with torch.no_grad():
        for x,y in loader:
            x,y=x.to(device),y.to(device);pred=model(x).argmax(1)
            for cls in range(1,7):
                p=pred==cls;t=y==cls;v=(p&t).sum().item();inter[cls]+=v;union[cls]+=(p|t).sum().item();di[cls]+=2*v;dd[cls]+=p.sum().item()+t.sum().item()
            for k,classes in [('frame_left',(1,2,3)),('frame_right',(4,5,6))]:
                for j in range(len(y)):
                    ty=torch.nonzero(y[j]==classes[-1]);py=torch.nonzero(pred[j]==classes[-1])
                    if ty.numel():
                        available[k]+=1
                        if py.numel():det[k]+=1;ce[k].append(torch.linalg.vector_norm(ty.float().mean(0)-py.float().mean(0)).item())
    return {'macro_dice_6_foreground_classes':float(np.mean(di[1:]/np.maximum(dd[1:],1))),
      'class_iou':{str(i):float(inter[i]/max(1,union[i])) for i in range(1,7)},
      'per_eye_pupil_detection_rate':{k:det[k]/max(available[k],1) for k in ce},
      'per_eye_center_error_px_mean':{k:float(np.mean(ce[k])) if ce[k] else None for k in ce},
      'n_frames':len(loader.dataset)}

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/dual_eye_unet.pt');p.add_argument('--epochs',type=int,default=10);p.add_argument('--train-count',type=int,default=2000);p.add_argument('--valid-count',type=int,default=300);p.add_argument('--test-count',type=int,default=400);p.add_argument('--seed',type=int,default=18159);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');torch.backends.cudnn.benchmark=device.type=='cuda'
    tr=DataLoader(FaceDataset(a.train_count,a.seed),batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=device.type=='cuda',persistent_workers=a.workers>0)
    va=DataLoader(FaceDataset(a.valid_count,a.seed+1_000_000),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda')
    te=DataLoader(FaceDataset(a.test_count,a.seed+2_000_000),batch_size=a.batch_size,shuffle=False,num_workers=a.workers,pin_memory=device.type=='cuda')
    m=DualEyeUNet().to(device);opt=torch.optim.AdamW(m.parameters(),lr=4e-4,weight_decay=2e-4);weights=torch.tensor([.15,.8,1.4,2.2,.8,1.4,2.2],device=device)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);best=-1;best_state=None;hist=[];since_improve=0;started=time.time()
    for ep in range(1,a.epochs+1):
        m.train();losses=[]
        for x,y in tr:
            x,y=x.to(device,non_blocking=True),y.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda',dtype=torch.float16):z=m(x);loss=nn.functional.cross_entropy(z,y,weight=weights)
            loss.backward();nn.utils.clip_grad_norm_(m.parameters(),2.);opt.step();losses.append(loss.item())
        vm=evaluate(m,va,device);record={'epoch':ep,'train_loss':float(np.mean(losses)),'validation':vm,'elapsed_min':round((time.time()-started)/60,2)};hist.append(record);print(json.dumps(record),flush=True)
        if vm['macro_dice_6_foreground_classes']>best:
            best=vm['macro_dice_6_foreground_classes'];best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()};since_improve=0
            torch.save({'state_dict':best_state,'base_channels':12,'classes':['background','frame_left_sclera','frame_left_iris','frame_left_pupil','frame_right_sclera','frame_right_iris','frame_right_pupil'],'seed':a.seed,'best_epoch':ep,'frame_size_wh':[W,H]},out)
        else:since_improve+=1
        if since_improve>=3:break
    m.load_state_dict(best_state);tm=evaluate(m,te,device);report={'status':'SYNTHETIC_FULL_FRAME_HELD_OUT_ONLY','device':str(device),'seed':a.seed,'train_count':a.train_count,'validation_count':a.valid_count,'test_count':a.test_count,'best_epoch':min(hist,key=lambda r:-r['validation']['macro_dice_6_foreground_classes'])['epoch'],'history':hist,'test':tm,'elapsed_min':round((time.time()-started)/60,2),'checkpoint':str(out)}
    report_path=out.with_suffix('.metrics.json');report_path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'final_report':str(report_path),'test':tm,'status':report['status']},indent=2),flush=True)
if __name__=='__main__':main()
