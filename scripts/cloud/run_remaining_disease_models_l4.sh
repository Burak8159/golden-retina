#!/usr/bin/env bash
# Sequential L4 queue for already-staged, publicly accessible disease cohorts.
# Images, split manifests, checkpoints, and per-image predictions remain on VM.
set -u

ROOT=/data/golden-retina/workspace
STOP_TRAINING_UTC=2026-11-22T06:45:00Z
STATUS="$ROOT/results/disease_expansion_queue_20261121/queue_status.log"
mkdir -p "$(dirname "$STATUS")"
cd "$ROOT" || exit 2

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
note() { printf '%s %s\n' "$(stamp)" "$*" | tee -a "$STATUS"; }
remaining() {
  local end now
  end=$(date -u -d "$STOP_TRAINING_UTC" +%s) || return 1
  now=$(date -u +%s)
  echo $((end-now))
}
run_step() {
  local name="$1" seconds rc
  shift
  seconds=$(remaining) || return 2
  if (( seconds <= 0 )); then note "SKIP $name: protected closeout window reached"; return 125; fi
  note "START $name; max remaining training seconds=$seconds"
  timeout --signal=INT --kill-after=90 "$seconds" "$@"
  rc=$?
  note "END $name exit=$rc"
  return 0
}

note "Queue started; waiting for active CLARUS PID 15961; training cutoff=$STOP_TRAINING_UTC"
while kill -0 15961 2>/dev/null; do sleep 20; done
note "CLARUS process ended; continuing only from local staged data"

run_step uwf_slo_dr_peng_2026_grouped_development \
  /data/golden-retina/venv/bin/python src/golden_retina/train_uwf_dr_peng_grouped.py \
  --root external/uwf_dr_peng_20261122/unpacked \
  --out results/uwf_dr_peng_20261122 \
  --epochs 12 --seconds-per-model 900 --image-size 320 --batch-size 12 --workers 4

run_step acrima_glaucoma \
  /data/golden-retina/venv/bin/python src/golden_retina/train_acrima_glaucoma.py \
  --data external/acrima_20261121/data/Database/Images \
  --out results/acrima_glaucoma_candidates_20261122 \
  --epochs 12 --batch-size 16 --workers 4

run_step fives_amd_dr_glaucoma \
  /data/golden-retina/venv/bin/python src/golden_retina/train_fives_fixed_split.py \
  --root 'external/expanded_20261121/fives/FIVES A Fundus Image Dataset for AI-based Vessel Segmentation' \
  --out results/fives_disease_transfer_20261122 \
  --exclude-root external/mendeley_expansion_20261121 \
  --exclude-root external/expanded_20261121/clarus700 \
  --image-size 384 --epochs 10 --batch-size 8 --workers 4 --seconds-per-model 1800

note "STAGE MuReD images and labels on VM"
python3 - <<'PY'
import csv, json, shutil, zipfile
from pathlib import Path, PurePosixPath

source = Path('external/mendeley_expansion_20261121/pc4mb3h8hz_v2')
target = Path('external/expanded_20261121/mured')
archive = source / 'images.zip'
target.mkdir(parents=True, exist_ok=True)
free = shutil.disk_usage(target).free
with zipfile.ZipFile(archive) as z:
    infos = z.infolist()
    required = sum(x.file_size for x in infos)
    names = [PurePosixPath(x.filename) for x in infos]
    if any(p.is_absolute() or '..' in p.parts or (p.parts and p.parts[0] != 'images') for p in names):
        raise SystemExit('MuReD archive path safety check failed')
    if free < required * 1.25:
        raise SystemExit(f'Insufficient VM disk headroom: free={free}, required={required}')
    z.extractall(target)
shutil.copy2(source/'train_data.csv', target/'train_data.csv')
shutil.copy2(source/'val_data.csv', target/'val_data.csv')
images = [p for p in (target/'images').rglob('*') if p.is_file()]
by_stem = {}
for p in images:
    if p.stem in by_stem:
        raise SystemExit('MuReD duplicate image stems prevent unambiguous CSV mapping')
    by_stem[p.stem] = p
summary = {'source':'MuReD v2 Mendeley CC BY 4.0','images_extracted':len(images),'train_rows':0,'validation_rows':0,'labels':[],'unmatched_keys':0,'patient_ids_available':False}
for filename, key in [('train_data.csv','train_rows'),('val_data.csv','validation_rows')]:
    with (target/filename).open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f'Empty MuReD table: {filename}')
    fields=list(rows[0]); names=fields[1:]
    if len(names)!=20:
        raise SystemExit(f'Expected 20 MuReD label columns, found {len(names)}')
    summary[key]=len(rows); summary['labels']=names
    summary['unmatched_keys'] += sum(Path(str(r[fields[0]]).strip()).stem not in by_stem for r in rows)
if summary['unmatched_keys']:
    raise SystemExit(f"MuReD label-to-image mapping failed: unmatched={summary['unmatched_keys']}")
Path('results/disease_expansion_queue_20261121/mured_stage_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary))
PY
stage_rc=$?
if (( stage_rc == 0 )); then
  run_step mured_20_label_development \
    /data/golden-retina/venv/bin/python src/golden_retina/train_mured_multilabel_development.py \
    --data-root external/expanded_20261121/mured \
    --out results/mured_20_label_development_20261122 \
    --epochs 10 --seconds 5400 --batch-size 12 --workers 4
else
  note "SKIP mured_20_label_development: VM-side staging/mapping preflight exit=$stage_rc"
fi

run_step uwf_tumor_six_class \
  /data/golden-retina/venv/bin/python src/golden_retina/train_uwf_tumor_exploratory.py \
  --data 'external/expanded_20261121/uwf_tumor/UWF Fundus Images of Intraocular Tumors' \
  --out results/uwf_tumor_exploratory_20261122 \
  --image-size 384 --epochs 10 --batch-size 8 --workers 4 --seconds-per-model 1800

note "Queue reached end; verify each aggregate_report.json and preserve the 63-minute closeout window."
