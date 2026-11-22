# Disease Dataset Expansion Audit — 2026-11-21

## Goal and boundary

This audit broadens Golden Retina's disease-specific training candidates using
datasets whose image modality is plausibly available from a fundus-photo
workflow. It does not make one model capable of detecting every eye disease.
An OCT-only or anterior-segment condition is out of scope for the current
fundus-camera pipeline. Every source is tracked separately because label
definitions, patient linkage, devices, populations, and access terms differ.

The full machine-readable register is
[`configs/disease_dataset_candidates_20261121.json`](../configs/disease_dataset_candidates_20261121.json).
Dataset archives, image-level manifests, row-level predictions, and checkpoints
are to remain on the authorized Google Cloud VM. Nothing from a dataset archive
is to be downloaded to this Mac or pushed to GitHub.

## New candidates with the clearest immediate path

| Candidate | Main use | Why it is useful | Limit that controls use |
|---|---|---|---|
| HYGD | Glaucomatous optic neuropathy | 747 fundus images from 288 patients, patient IDs, quality scores, and reference labels supported by comprehensive examinations including OCT and visual fields; openly downloadable under ODbL-1.0 | One site and one 45-degree camera; its patient split is a development estimate, not external validation of a different camera |
| Eye Disease Image Dataset (Mendeley) | DR, glaucoma, CSCR, RD, RP, myopia, disc edema, macular scar | 5,318 fundus originals with ten classes and a stated CC BY 4.0 license | No patient linkage established; 16,242 augmentations are also described, so only original images may be considered and image lineage must be checked |
| MuReD | Multi-label training across 19 disease labels plus normal/other categories | 2,208 fundus images; CC BY 4.0; a broad label vocabulary | Built from ARIA, STARE, and RFMiD; overlap with existing RFMiD/ARIA/STARE data is expected, so it is not an independent test |
| CSC fundus segmentation dataset | CSC classification and SRF localization | 194 CSC eyes plus 93 healthy photos; manual segmentation; CC BY 4.0 | Patient IDs and source partitions are not established; use only for exploratory development after duplicate checks |

HYGD is the strongest new candidate for a *disease-specific* model because it
combines patient IDs, an image-quality score, and a clinically grounded GON
reference. This is a rationale for running a properly grouped experiment, not a
claim that a model trained on HYGD will generalize to Golden Retina hardware.

## Candidate gates

- **HYAMD (AMD):** very good modality and label fit (patient-linked fundus
  photos with examination/OCT-supported labels), but PhysioNet requires a
  registered-user approval and dataset-specific DUA. Keep it gated until the
  user's own access grant and agreement are verified.
- **PAPILA (glaucoma):** valuable paired-eye patient-linked cohort; the primary
  Figshare record's data license must be verified before use. A secondary
  catalogue reports GPL-3.0-or-later; that is not sufficient to clear the
  rights gate by itself.
- **G1020 (glaucoma):** reports 1,020 images from 432 patients and clinical
  labels, but the original download route is unreliable and a standard dataset
  license was not verified. Do not substitute a third-party bundle as proof of
  permission.
- **REFUGE / REFUGE2 (glaucoma):** useful multi-device challenge cohorts, but
  registration and challenge-specific research/educational terms apply. Use
  only through the official route after terms are clear.
- **RFMiD 2.0:** 860 images and 49 labels are attractive for rare-label probes,
  but its official cloud endpoint previously returned HTTP 403. Do not bypass
  the official access path or use unknown-license mirrors.
- **TOP (multi-disease):** a 13,047-image, 5,389-patient Optos UWF cohort is
  described in the literature. It is a separate ultra-widefield pseudocolor
  domain; exact release/access terms and compatibility with the deployed camera
  remain unresolved.
- **OCTDL and other OCT datasets:** excluded from this fundus-only pipeline.
  They may become candidates only if the product acquires OCT hardware.
- **ROP datasets:** excluded from the current adult fundus workflow because
  infant cohorts and RetCam/widefield acquisition create a distinct population
  and device task. Do not fold ROP into an adult model as if it were the same
  endpoint.

## Per-disease model plan

The established disease targets remain separate experts. The first new training
priority is glaucoma on HYGD, with a patient-grouped split and a frozen
ConvNeXt-Tiny baseline compared against available retinal foundation encoders
only if the weights can be accessed through their documented official path.
Next, train source-specific binary experts for CSC and the Mendeley fundus
classes that have adequate positive and negative patient/duplicate support.
MuReD is a train-only multi-label augmentation of coverage after source overlap
checks; its original validation partition is not independent evidence.

For each disease, establish in this order: exact label definition; acquisition
modality fit; original-file and patient/eye structure; source terms; duplicates
and overlap; patient split; baseline; disease-specific adaptation; validation
threshold selection; one frozen evaluation; per-disease uncertainty and error
report. Do not tune on a cohort after scoring its held-out portion.

Rare labels should not receive a standalone model merely because their names
appear in a dataset. Require adequate patient-level positive and negative
support. Otherwise retain them as exploratory labels or exclude them from the
app's disease-risk interface. Do not relabel a finding (e.g. drusen, disc
cupping, or tessellation) as a confirmed disease diagnosis.

## Cloud and budget gate

All three named L4 instances are currently stopped; three 150-GB persistent
disks and one snapshot remain. The latest stored billing report is dated
2026-11-20 and shows TRY 0 net after credits alongside TRY 3,932.09 gross usage;
same-day charges and retained-storage cost are not verified. The configured
monthly billing alert is TRY 1,400. The project campaign ceiling remains
TRY 1,900 planned plus TRY 80 protected reserve (TRY 1,980 hard), but that is a
project guard, not proof of the current bill.

At the recorded conservative rate of TRY 45.80 per L4-hour, a 13-hour run is
estimated at TRY 595.40 of compute; adding the prior campaign's TRY 436.79
compute estimate gives TRY 1,032.19 before retained storage, tax, network, and
delayed charges. This is under the planned ceiling on the estimate alone. The
live invoice is still unverified, so the cloud run must have an automatic
13-hour STOP cap and spend tracking, and no unrelated Cloud project/resource
may be modified. If the live charge estimate would exceed the hard ceiling,
compute must stop and only no-cost source audit work may continue.

## Sources

- [HYGD official PhysioNet record](https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.1.0/)
- [Eye Disease Image Dataset, Mendeley](https://data.mendeley.com/datasets/s9bfhswzjb/1)
- [MuReD, Mendeley](https://data.mendeley.com/datasets/pc4mb3h8hz/2)
- [CSC fundus segmentation, Mendeley](https://data.mendeley.com/datasets/4k64fwnp4k/5)
- [HYAMD official PhysioNet record](https://physionet.org/content/hillel-yaffe-fundus-amd/1.0.0/)
- [PAPILA source paper and Figshare DOI](https://www.nature.com/articles/s41597-022-01388-1)
- [G1020 source paper](https://arxiv.org/abs/2006.09158)
- [REFUGE official challenge](https://refuge.grand-challenge.org/)
- [RFMiD 2.0 dataset descriptor](https://www.mdpi.com/2306-5729/8/2/29)
- [TOP supplementary data description](https://bjo.bmj.com/content/bjophthalmol/early/2021/08/02/bjophthalmol-2021-319030/DC1)
- [OCTDL paper](https://arxiv.org/abs/2312.08255)
- [Google Cloud Spot pricing](https://cloud.google.com/spot-vms/pricing)


## Additional source screening during the 13-hour run

- **AMDNet23** is listed as CC BY 4.0 and describes 2,000 enhanced images from six other public sources. The official Mendeley API exposes 1,994 flat JPEG file records but no verified label-to-file mapping. It is therefore not trained or counted as new AMD evidence; source duplication is also expected. [Official Mendeley record](https://data.mendeley.com/datasets/yj35kjgrv3/1)
- **Composite retinal fundus and OCT dataset v4** is listed as CC BY 4.0 and describes fundus plus OCT images for AMD, CSR, DME and glaucoma. Its 1.54-GB archive is being staged on the L4 for modality and label audit. OCT scans will remain outside the fundus classifier. [Official Mendeley v4 record](https://data.mendeley.com/datasets/trghs22fpg/4)
- **Bajwa Hospital Multi Eye Disease Dataset** is listed as CC BY 4.0 but its 601 images include fused modalities and an unspecified “retinal disease” class; it is excluded from disease-specific training. [Official Mendeley record](https://data.mendeley.com/datasets/rgwpd4m785/3)
- **Bangladesh Multi-Disease Retinal Dataset (BMRD)** is listed as CC BY 4.0 and claims 1,786 color fundus images from three hospitals with DR, AMD, and glaucoma severity labels. The official archive contains 1,821 JPEGs but no sidecar label table; only a small subset of filenames carries recognizable disease tokens and most are generic image names. It is staged on the VM, but the label map cannot currently be reconstructed safely, so it is not trainable until the authors provide a mapping. [Official Mendeley record](https://data.mendeley.com/datasets/959whsbc6v/1)
- **A Multi-Class Retinal Fundus Image Dataset for Deep Learning-Based Ocular Disease Diagnosis** (Rajbari clinic, 2026) is CC BY 4.0 and reports 1,222 originals across 11 DR/AMD/glaucoma severity labels plus 5,500 augmented derivatives. The official archive downloaded to the VM contains exactly 5,500 images, 500 per class; the reported original images are absent. The archive therefore needs an augmentation-parent/near-duplicate audit before any split or test. Do not treat its balanced folder counts as independent patients. [Official Mendeley record](https://data.mendeley.com/datasets/z227d626vm/1)
- **Diabetic Retinopathy and Retinal Diseases Fundus Image Dataset** (2025) is CC BY 4.0 and advertises 22,454 images from three public sources. The official API exposes 22,453 individual files (2.88 GB) rather than a labeled archive. Filename audit identifies only a subset with unambiguous disease-name prefixes; the rest are numeric/unmapped, and source overlap is expected. Keep it screened until file-to-class and source/duplicate lineage can be established. [Official Mendeley record](https://data.mendeley.com/datasets/6ry9n79gb4/1)
- **FIVES** is a directly downloadable CC BY 4.0 fundus cohort with 800 images across AMD, DR, glaucoma, and normal categories. It reports 573 patients but provides an image-level 600/200 train/test organization rather than a patient-linked partition; both eyes may be present. This is being added as an untouched source-level transfer check for the already-trained Mendeley model, with no FIVES-based model selection. [Official Figshare record](https://figshare.com/articles/figure/FIVES_A_Fundus_Image_Dataset_for_AI-based_Vessel_Segmentation/19688169)
- **CLARUS700** is a newly released (21 November 2026) CC0 cohort: 1,051 indexed UWF fundus images from 674 patients, with official patient-level train/validation/test partitions and four binary outputs (DR, RVO, high myopia, and heterogeneous other disease). The five image archives total about 11.9 GB and their MD5 values match the official Figshare API. They were fetched and extracted only on the L4. The public splits resolve to 732/150/163 model-eligible images and 471/101/102 patients; test positives are DR 23, RVO 11, high myopia 29 and other disease 28. The “other disease” label is not AMD-specific. The camera is a single-centre approximately 133-degree ZEISS CLARUS 700 UWF system, so even a patient-level test is only single-site/device evidence, not a standard-camera clinical validation. [Official Figshare record](https://doi.org/10.6084/m9.figshare.32712513.v1), [dataset descriptor](https://www.nature.com/articles/s41597-026-08875-9)

### CLARUS700 experiment protocol

The new patient-disjoint source permits a stronger held-out evaluation than the image-only sources. The L4 run compares ImageNet ConvNeXt-Tiny, Mendeley-initialized ConvNeXt-Tiny, ImageNet ResNet-50, and ImageNet EfficientNet-B0. Candidate/epoch selection uses validation macro-AUPRC (AUROC tie-break); per-label thresholds are selected on validation only. Only the selected candidate is scored on the official test, with patient-cluster bootstrap intervals. Images marked ineligible are excluded. All row-level identifiers, predictions, and checkpoints remain on the VM. This 133-degree UWF domain differs from the current ordinary fundus-camera workflow; transfer to that hardware remains unproven.

### Follow-up official-source screening (22 November 2026)

- **HYAMD, Hillel Yaffe AMD**: the current PhysioNet page describes 1,560 images from 325 patients with early and intermediate/late AMD labels supported by comprehensive exams and OCT. The page is access-restricted: credentialed access, required CITI training, and a signed DUA are required. It is not present on the L4 and was not downloaded. The cohort shares the Hillel Yaffe site/camera context with HYGD, so it is not an independent site test of HYGD. [Official PhysioNet record](https://physionet.org/content/hillel-yaffe-fundus-amd/1.0.0/)
- **BRSET v1.0.2**: the official record describes 16,266 fundus images from 8,524 patients and image labels including DR, diabetic macular edema, scar, nevus, AMD, vascular occlusion, hypertensive retinopathy, drusen, hemorrhage, detachment, myopic fundus, increased cup-disc ratio, and other. It is access-restricted and requires credentialing, CITI training, and a signed DUA. It is not present on the L4 and was not downloaded. These exact source labels need separate clinical-semantic review before treating every label as a disease endpoint. [Official PhysioNet record](https://physionet.org/content/brazilian-ophthalmological/1.0.2/)
- **Mendeley 22,454-image retinal compilation** remains a screened-only source. Its CC BY 4.0 metadata does not solve source lineage: the record says it merges three public sources, while filenames leave 11,356 entries without a clear disease prefix. It is not counted as a fresh cohort or independent test. [Official Mendeley record](https://data.mendeley.com/datasets/6ry9n79gb4/1)

### Newly staged UWF diabetic-retinopathy source (22 November 2026)

- **Ultra-wide-field (SLO) fundus image dataset for intelligent diabetic retinopathy system**: the official Figshare API reports 1,630 images in a 401,670,342-byte archive, licensed CC BY 4.0. The associated descriptor reports three classes (normal, NPDR, PDR), 809 patients, two Shenzhen hospitals, expert annotation, and Optomap Daytona 220°–240° UWF acquisition. This is a promising disease-specific DR source, but its UWF modality is not interchangeable with ordinary 45° fundus photographs or the product camera. [Official Figshare record](https://doi.org/10.6084/m9.figshare.31259494.v1), [dataset descriptor](https://www.nature.com/articles/s41597-026-07093-7)
- The archive was downloaded and extracted **on the L4 VM only**. The Figshare-supplied MD5 matched the remote metadata (`8abeda2c1a91eca0df53cd317db64961`); the archive SHA-256 is `4d60410559620a14dee62bca67472c38cd27c4e1b88cd4e4191c21f3986ff26b`. The extracted archive contains 1,630 PNGs and train/validation/test CSVs (1,141/163/326 rows).
- The source paper does not define the patient-ID semantics of filename tokens. A provisional grouping key made from the first two filename tokens occurs in all three official partitions (3 train/validation, 4 train/test, and 1 validation/test shared keys). Therefore the official image split is **not accepted as patient-independent** on present evidence. Before model selection, rebuild a disjoint development/evaluation split using a verified identity key if available; otherwise use the filename key only as a conservative grouping proxy and label the resulting estimate accordingly. Exact-content duplicates and overlap with existing training cohorts must also be audited before scoring.
