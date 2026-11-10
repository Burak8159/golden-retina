"""Reproducible end-to-end synthetic full-frame pipeline from an empty run folder."""
import argparse,json,subprocess,sys,time
from pathlib import Path

def run(cmd):
    print("RUN", " ".join(map(str,cmd)),flush=True);subprocess.run(list(map(str,cmd)),check=True)

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",default="results/full_frame_v2");p.add_argument("--train",type=int,default=1800);p.add_argument("--validation",type=int,default=300);p.add_argument("--test",type=int,default=400);p.add_argument("--epochs",type=int,default=24);p.add_argument("--batch-size",type=int,default=16);p.add_argument("--seed",type=int,default=815900);a=p.parse_args()
    root=Path(__file__).resolve().parent;out=Path(a.run_dir);data=out/"data";art=out/"artifacts";out.mkdir(parents=True,exist_ok=True)
    state={"pipeline":"full_frame_v2","status":"STARTED_FROM_EMPTY_RUN_DIRECTORY","started_unix":time.time(),"generator":"pending","dataset_audit":"pending","training":"pending","locked_test":"pending","inference_smoke":"pending"}
    (out/"pipeline_state.json").write_text(json.dumps(state,indent=2)+"\n")
    try:
        run([sys.executable,root/"generate.py","--output",data,"--train",a.train,"--validation",a.validation,"--test",a.test,"--seed",a.seed]);state["generator"]="complete";state["manifest"]=str(data/"manifest.csv");(out/"pipeline_state.json").write_text(json.dumps(state,indent=2)+"\n")
        run([sys.executable,root/"audit_dataset.py","--manifest",data/"manifest.csv","--output",data/"dataset_audit.json"]);state["dataset_audit"]="complete";(out/"pipeline_state.json").write_text(json.dumps(state,indent=2)+"\n")
        run([sys.executable,root/"train.py","--manifest",data/"manifest.csv","--output",art,"--epochs",a.epochs,"--batch-size",a.batch_size,"--seed",a.seed]);state["training"]="complete";(out/"pipeline_state.json").write_text(json.dumps(state,indent=2)+"\n")
        run([sys.executable,root/"evaluate.py","--manifest",data/"manifest.csv","--checkpoint",art/"best_checkpoint.pt","--output",art/"locked_test_metrics.json","--batch-size",a.batch_size]);state["locked_test"]="complete";(out/"pipeline_state.json").write_text(json.dumps(state,indent=2)+"\n")
        import csv
        with (data/"manifest.csv").open(newline="") as f:sample=next((r for r in csv.DictReader(f) if r["split"]=="test"),None)
        if sample:
            run([sys.executable,root/"predict.py","--image",sample["image"],"--checkpoint",art/"best_checkpoint.pt","--output-dir",art/"inference_smoke"]);state["inference_smoke"]="complete"
        state["status"]="END_TO_END_PIPELINE_COMPLETE_SYNTHETIC_ONLY";state["elapsed_min"]=round((time.time()-state["started_unix"])/60,2)
    except Exception as e:
        state["status"]="FAILED_WITH_ERROR";state["error"]=repr(e);state["elapsed_min"]=round((time.time()-state["started_unix"])/60,2);raise
    finally:(out/"pipeline_state.json").write_text(json.dumps(state,indent=2)+"\n")
    print(json.dumps(state,indent=2))
if __name__=="__main__":main()
