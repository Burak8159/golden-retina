# Golden Retina dataset candidate screen

**As of:** 2026-11-21
**Purpose:** choose fundus sources for model development without placing image archives on this Mac.
**Data-handling rule:** dataset images, archives, and extracted files go directly to the authorized Golden Retina VM only, after the billing and VM readiness gates pass. This document contains source metadata and access notes only.

## Screening results

| Candidate | Image-supported task | Evidence / split caveat | Access and rights gate | Current decision |
|---|---|---|---|---|
| **RFMiD / RIADD** | Multi-label retinal disease/finding recognition and normal/abnormal screening; 3,200 fundus images and 46 categories (normal plus abnormal categories) are described by the challenge/data paper. | Rare labels have small support; confirm image-to-patient and image-to-eye linkage, exact label map, and challenge split provenance before any claim. | Official RIADD challenge pages are the acquisition route. Recheck the current participant rules and exact terms before VM retrieval; do not rely on third-party mirrors or assume the paper's license covers every use. | **Top multi-disease candidate; terms and metadata verification first.** |
| **JustRAIGS** | Referable-glaucoma screening from color fundus images, with additional glaucoma-feature labels. Official source describes 101,442 gradable training images; the closed test set is not downloadable. | Strong scale; verify whether subject identifiers and eye linkage permit a patient-level split in the training release. No independent closed test can be locally scored. | Official challenge page says the training release is public on Zenodo under CC BY-NC-SA. Honor attribution/share-alike/noncommercial conditions; inspect the actual Zenodo record/license and intended-use fit before use. | **High-priority glaucoma development source; not a substitute for independent external testing.** |
| **AIROGS** | Referable glaucoma, nonreferable glaucoma, and ungradable-image classification; source describes 113,893 images from 60,357 subjects and roughly 500 sites. | Excellent source/domain diversity potential. Keep subject groups together; test set is closed and evaluated through the challenge. | Official page says training access requests on Zenodo are manually accepted. Do not automate around the approval gate. Exact training terms must be checked before use. | **High-value, access-request gated.** |
| **ODIR-5K** | Patient-level multi-label recognition covering diabetes, glaucoma, cataract, AMD, hypertension, myopia, other, and normal labels, using bilateral fundus photos plus age. | Labels are assigned at the **patient** level using both eyes and age; they cannot be silently copied as independent eye-level ground truth. Split by patient and report this as a limitation. | Official challenge page provides a download route but the precise current reuse terms must be checked before downloading, training, or publishing derived outputs. | **Useful broad-screening research source after label/task and rights audit.** |
| **PAPILA** | Glaucoma assessment with paired left/right fundus images, clinical labels, and optic-disc/cup segmentations. | Small cohort; keep both eyes of a patient in one partition. It supports glaucoma/structure work, not broad disease screening. | The data descriptor identifies the Figshare DOI and content. Verify the current item's exact license/version and reuse conditions from the repository record before VM staging; a third-party catalog is not sufficient. | **Promising paired-eye glaucoma/segmentation source; item-level terms pending.** |
| **ADAM / iChallenge-AMD** | AMD/non-AMD fundus classification, optic-disc segmentation, fovea localization, and lesion detection/segmentation. | Official challenge article describes a 1,200-image dataset. Confirm currently downloadable subset, patient identifiers, and class counts before choosing tasks. | Official Grand Challenge says participation entails registration and agreement to challenge rules. Complete only accurate authorized registration; review the terms before retrieving. | **High-priority AMD/lesion candidate; challenge terms gate.** |
| **PALM / iChallenge-PM** | Pathological-myopia fundus classification and lesion segmentation. | Task-specific source; do not generalize to ordinary myopia or other disease. Verify usable patient counts and split fields. | Official challenge requires team registration and agreement to rules; exact current access and publication conditions must be read before retrieval. | **Retain as a separate pathological-myopia task; access gated.** |
| **REFUGE / REFUGE2** | Glaucoma classification, image quality, and optic-disc/cup or related structure tasks, depending on release. | Device/source split can be useful; confirm patient linkage, release version, and actual labels before setting an evaluation design. | Official challenge rules apply to registered teams and downloaded data. Verify REFUGE2-specific access, permitted use, and test-data handling; do not infer a license from mirrors. | **Candidate for glaucoma evaluation after exact release/terms review.** |

## Selection and VM plan

1. Keep all downloaded data off the Mac. Use the Mac only for source pages, code, metadata, and documentation.
2. First clear the cloud cost gate and inspect the existing Golden Retina VM/disk without deleting it. The VM is currently recorded as terminated; its 150 GB persistent disk is preserved.
3. On the VM, record each selected source's official version, terms, download URL, file count, archive checksum, label schema, and patient/eye identifiers before extraction or training.
4. Prefer a small, task-specific first acquisition under the budget: RFMiD for multi-label screening and one glaucoma source with verified patient IDs; add AMD/lesion or pathological-myopia sources only when access and data-volume checks pass.
5. Freeze patient-level splits before model selection. Keep challenge test sets closed and use their official evaluation route when applicable.
6. Do not publish images, row-level predictions, or derived records unless the source terms explicitly permit it. Publish code and suitably aggregated results only after a terms review.

## Source references

- [RIADD / RFMiD official challenge](https://riadd.grand-challenge.org/) and [RFMiD data descriptor](https://www.mdpi.com/2306-5729/6/2/14)
- [JustRAIGS official dataset page](https://justraigs.grand-challenge.org/dataset/) and [official Zenodo training record](https://zenodo.org/records/10035093)
- [AIROGS official data and challenge page](https://airogs.grand-challenge.org/data-and-challenge/)
- [ODIR-2019 official dataset page](https://odir2019.grand-challenge.org/dataset/)
- [PAPILA data descriptor](https://www.nature.com/articles/s41597-022-01388-1) and [Figshare DOI](https://doi.org/10.6084/m9.figshare.14798004)
- [ADAM / iChallenge-AMD official challenge](https://amd.grand-challenge.org/)
- [PALM official challenge and rules](https://palm.grand-challenge.org/Home/)
- [REFUGE official rules](https://refuge.grand-challenge.org/Rules/)

**Boundary:** a public landing page is evidence that a dataset is described or discoverable, not by itself proof that a particular training, derivative, or redistribution use is permitted. Where the exact version terms or identifiers were not directly verified, the candidate remains gated.
