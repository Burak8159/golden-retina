"""Write the seed-disjoint synthetic split manifest used by the trainer."""
import argparse, csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/split_manifest.csv');p.add_argument('--seed',type=int,default=8159);p.add_argument('--train-count',type=int,default=3000);p.add_argument('--valid-count',type=int,default=400);p.add_argument('--test-count',type=int,default=600);a=p.parse_args()
    rows=[]
    for name,offset,count in [('train',0,a.train_count),('validation',1_000_000,a.valid_count),('test',2_000_000,a.test_count)]:
        for i in range(count):
            seed=a.seed+offset+i
            rows.append({'split':name,'synthetic_subject_id':f'syn-{seed}','scene_seed':seed,'source':'golden_retina_synthetic_renderer_v0.1','width':192,'height':128,'labels':'background,sclera,iris,pupil'})
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    print(f'Wrote {len(rows)} deterministic samples to {out}; seeds are disjoint by split.')
if __name__=='__main__': main()
