# Golden Retina mobile result and disease-label content contract

## Label taxonomy

The app follows `mobile_models/onnx/manifest.json` `output_kind` values. It contains **7 RFMiD dataset condition labels** and **10 retinal image findings**, across 17 optional research heads. This classification is dataset label taxonomy, not a claim that a model clinically detects each condition.

### Dataset condition labels

- Diabetic retinopathy (`rfmid_dr`)
- Age-related macular degeneration (`rfmid_armd`)
- Myopia (`rfmid_mya`); do not rename it pathological myopia
- Branch retinal vein occlusion (`rfmid_brvo`)
- Central retinal vein occlusion (`rfmid_crvo`)
- Central serous retinopathy (`rfmid_csr`)
- Chorioretinitis (`rfmid_crs`)

### Retinal image findings

- Drusen (`rfmid_dn`)
- Optic-disc cupping (`rfmid_odc`)
- Optic-disc pallor (`rfmid_odp`)
- Optic-disc edema (`rfmid_ode`)
- Tessellation (`rfmid_tsln`)
- Laser scar (`rfmid_ls`)
- Retinitis image finding (`rfmid_rs`); preserve the manifest's finding kind rather than upgrading it to a confirmed diagnosis
- Macular scar (`rfmid_ms`)
- Retinal traction (`rfmid_rt`)
- Epiretinal membrane (`rfmid_erm`); RFMiD manifest explicitly marks this head as a retinal image finding

The UI groups related findings for navigation, but does not mix them into disease-condition groups. Optic-disc cupping is not glaucoma; there is no glaucoma classifier. There is no quality head, no supported “any retinal disease” head, and no calibrated clinical-risk output. Uninstalled heads are not assessed, not zero.

## Score wording and result layout

- Keep the user's explicit eye assignment. Show Left first on the physical left position and Right second; each eye tab shows that eye's own highest available raw score or an em dash before analysis.
- Label displayed numbers “raw model score” or “highest raw score.” Percentage formatting does not make them a probability or clinical risk.
- Show RFMiD provenance, installed model version, and core-shortlist versus exploratory status for each model result.
- Explain that the research cutoff `0.50` is not a clinical decision threshold. Do not use positive/negative, high/low risk, “all clear,” or referral wording.
- Separate the guided sample demonstration from image-based local inference. Fixed values are demo-only and must remain visibly marked as sample; no image or model generated them.
- Show unsupported/absent outputs as “not assessed” or explain that only installed heads are listed. Never imply that omitted conditions scored zero.

## Evidence boundary for app copy

The ONNX manifest records RFMiD image-split development/test performance for these weights, but patient independence is not verified. The four post-hoc core shortlist heads and exploratory heads are research demonstrations; they are not independently validated for screening or diagnosis. The app therefore exposes model output with provenance and limitations, not clinical recommendations. The synthetic eye-localization model is separate and has no validation on people, IR camera frames, or goggles.

## Locked and future functions

Goggle capture, Bluetooth pairing, quality/gradable assessment, examination history, report export and functional-vision testing remain unavailable until required hardware/API, data, security and validation contracts exist. Do not add permissions, fake telemetry, medical records or clinical claims to fill those gaps.
