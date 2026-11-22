import json
from datetime import datetime

p = "tools/progress_tracker/status.json"
d = json.load(open(p))
now = "2026-11-22T02:15:00Z"
by = {b["id"]: b for b in d["blocks"]}
by[89]["evidence"] = (
    "BMRD exact SHA-256 scan covered 1,821 image paths / 1,640 unique hashes and six existing source roots "
    "(RFMiD1, FDR26, PAPILA, FIVES, Bangladesh multi-disease, ACRIMA); zero cross-source exact matches. "
    "This does not detect recompressed/cropped copies. Rajbari ZIP, in-progress Mendeley13 and JustRAIGS "
    "are still pending the complete overlap gate."
)
by[90]["evidence"] += (
    " Frozen DINOv2 ViT-B/14 feature probe completed (training-hospital validation selected C; BMRD test sites "
    "had already been opened by CNN, so secondary results are exploratory). DR DINO: n=291, accuracy 67.01%, "
    "balanced accuracy 67.30%, macro-F1 57.88%, macro-AUROC .871; lower accuracy/AUROC but higher balanced "
    "accuracy/F1 than ConvNeXt. AMD DINO: n=43, accuracy 34.88%, balanced accuracy 44.36%, macro-F1 36.65%, "
    "macro-AUROC .668 (worse than ConvNeXt). Glaucoma DINO: n=43, accuracy 58.14%, balanced accuracy 53.77%, "
    "macro-F1 49.17%, macro-AUROC .798; very small class support. No model selected from test comparisons."
)
for x in d.get("disease_program", {}).get("diseases", []):
    nm = x.get("name", "").lower()
    if "diabetic retinopathy" in nm:
        x["campaign_status"] = "BMRD held-out-site result provisional; cross-source exact overlap passed for six known roots; patient IDs absent"
        x["data_status"] = (
            "BMRD provisional Khulna-site test n=291. ConvNeXt accuracy 69.76%, bAcc 64.66%, macro-F1 51.76%, "
            "macro-AUROC .921; Level 3 sensitivity 2.1% (47/48 missed). DINOv2 accuracy 67.01%, bAcc 67.30%, "
            "macro-F1 57.88%, macro-AUROC .871; Level 3 sensitivity 37.5% (30/48 missed). Site-separated, no patient IDs; "
            "test was opened by CNN before the secondary DINO probe. Exploratory only."
        )
        x["next_action"] = "Do not choose using this exposed test; seek larger, patient-linked and label-harmonized DR cohorts."
    elif "age-related macular degeneration" in nm:
        x["campaign_status"] = "BMRD AMD held-out-site result low-support exploratory; no patient IDs"
        x["data_status"] = (
            "BMRD Khulna-site test n=43. ConvNeXt accuracy 53.49%, bAcc 50.59%, macro-F1 49.71%, macro-AUROC .746. "
            "DINOv2 accuracy 34.88%, bAcc 44.36%, macro-F1 36.65%, macro-AUROC .668. Only 43 images; exploratory."
        )
        x["next_action"] = "Keep ConvNeXt as the better of the two exploratory candidates; acquire more independent AMD evidence."
    elif "glaucoma-suspect" in nm:
        x["campaign_status"] = "BMRD glaucoma severity held-out-site result low-support exploratory; JustRAIGS remains active candidate"
        x["data_status"] = (
            "BMRD Anwar Khan site test n=43; no patient IDs. ConvNeXt accuracy 60.47%, bAcc 40.95%, macro-F1 41.27%, "
            "macro-AUROC .772. DINOv2 accuracy 58.14%, bAcc 53.77%, macro-F1 49.17%, macro-AUROC .798. Per-grade supports "
            "6/3/8/26. Do not interpret as confirmed clinical severity grading."
        )
        x["next_action"] = "Complete JustRAIGS RG-vs-NRG model using reconstructed fellow-eye groups; report that separate binary task."
roster = d.setdefault("candidate_roster", [])
new_candidates = [
    {"name": "MCFI multi-session colored fundus images (Mendeley 2026)", "family": "longitudinal / repeatability, not disease classifier", "availability_status": "screened · excluded from disease training", "terms_status": "CC BY 4.0 per source record; verify at use", "decision": "442 images from 37 participants with bilateral/repeated sessions; health context is participant-reported, not a disease-grade reference standard. Potential repeatability study only."},
    {"name": "CAVRI-H5 VMA/VMT OCT cohort (Mendeley 2026)", "family": "OCT-only retinal interface pathology", "availability_status": "screened · excluded for current fundus camera scope", "terms_status": "source-specific; not downloaded", "decision": "OCT B-scans are required; cannot train current fundus-photo system."},
]
for item in new_candidates:
    if not any(r.get("name") == item["name"] for r in roster):
        roster.append(item)
datasets = d.setdefault("datasets", [])
new_datasets = [
    {"name": "MCFI: Multi-Session Colored Fundus Image Dataset", "access_class": "public record; CC BY 4.0 reported on Mendeley", "local_root": "not downloaded", "local_status": {"present": False}, "golden_retina_role": "longitudinal/repeatability research, not disease-supervised classification", "notes": "442 images from 37 participants with bilateral repeated sessions; participant-reported health, no verified disease-grade classifier reference. Screened and excluded from disease training."},
    {"name": "CAVRI-H5 VMA/VMT", "access_class": "public Mendeley record; no download attempted", "local_root": "not downloaded", "local_status": {"present": False}, "golden_retina_role": "none for current fundus-only camera", "notes": "OCT B-scan pathology; excluded because current imaging pipeline is fundus-only."},
]
for item in new_datasets:
    if not any(x.get("name") == item["name"] for x in datasets):
        datasets.append(item)
d["activity_feed"].insert(0, {"timestamp": now, "kind": "model_comparison", "title": "DINOv2 frozen probe compared on BMRD; results mixed", "text": "With identical site splits and validation-only C selection, frozen DINOv2 features improved DR balanced accuracy/macro-F1 (67.30%/57.88% vs ConvNeXt 64.66%/51.76%) but reduced raw accuracy (67.01% vs 69.76%) and macro-AUROC (.871 vs .921); AMD was worse (34.88% accuracy); glaucoma balanced accuracy was higher but only n=43 with tiny class support. Secondary scores reuse an already-open test and remain exploratory; no test-driven model selection."})
d["activity_feed"] = d["activity_feed"][:100]
d["current_block"] = 90
d["current_phase"] = 90
d["updated_at"] = now
elapsed = (datetime.fromisoformat(now.replace("Z", "+00:00")) - datetime.fromisoformat("2026-11-21T18:48:11+00:00")).total_seconds() / 3600
b = d["budget"]
b["gpu_hours_used"] = round(elapsed, 2)
b["gpu_hours_remaining"] = round(max(0, 13 - elapsed), 2)
b["estimated_spent_try"] = round(436.79 + elapsed * 45.8 + 7, 2)
b["spend_verified"] = False
b["display_label"] = (f"Estimate only: prior ₺436.79 + {elapsed:.2f} elapsed campaign hours at ₺45.80/h + provisional ₺7 network "
                      f"= ₺{b['estimated_spent_try']:.2f}; live invoice, tax and retained-storage charges unverified. "
                      "₺80 reserve protected; hard ceiling ₺1,980.")
b["planning_estimate_try"] = round(436.79 + 13 * 45.8 + 7, 2)
with open(p, "w") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
    f.write("\n")
print("tracker refreshed", round(elapsed, 2), "hours; estimate ₺", b["estimated_spent_try"])
