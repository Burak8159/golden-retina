"""Measure batch-one model latency on the configured device."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from train_segmenter import TinyUNet

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',default='artifacts/inference_profile.json');p.add_argument('--warmup',type=int,default=100);p.add_argument('--runs',type=int,default=500);a=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=TinyUNet().to(device).eval();m.load_state_dict(ck['state_dict'])
    x=torch.rand(1,1,128,192,device=device)
    with torch.no_grad():
        for _ in range(a.warmup):m(x)
        if device.type=='cuda':torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(a.runs):
            t=time.perf_counter();m(x)
            if device.type=='cuda':torch.cuda.synchronize()
            times.append((time.perf_counter()-t)*1000)
    report={'status':'SYNTHETIC_INPUT_RUNTIME_PROFILE','device':str(device),'warmup_runs':a.warmup,'measured_runs':a.runs,'batch_size':1,
      'latency_ms_median':float(np.median(times)),'latency_ms_p95':float(np.percentile(times,95)),'throughput_fps_median_inverse':float(1000/np.median(times)),
      'peak_allocated_gpu_mib':float(torch.cuda.max_memory_allocated()/2**20) if device.type=='cuda' else None}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
