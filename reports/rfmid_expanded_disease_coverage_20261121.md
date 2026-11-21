# RFMiD disease coverage extension — 2026-11-21

## Scope and state

This is a bounded extension of the Golden Retina development campaign, using the RFMiD 1.0 image-level train/validation/test partitions already present on the L4 VM. No new dataset was downloaded. The run warm-starts from the saved epoch-14 shared ConvNeXt-Tiny checkpoint and expands loss support from the previous 14-label 30-positive/30-negative floor to every label with at least 10 positives and 10 negatives in training.

The training run is in progress. The model is one shared multi-label image classifier, not 24 independently trained specialist models. RFMiD does not supply patient identifiers in the staged label tables; independence by patient cannot be checked. This RFMiD test split has been seen in earlier project work, so any new test result is descriptive development evidence, not a new independent validation.

## Label support and interpretation

The official RFMiD data description enumerates source-defined disease and finding labels. Their names below preserve the dataset semantics; they are not a claim that every output is a standalone diagnosis. In particular, `MH` means **media haze**, while `MHL` means **macular hole**. `ODC` is optic-disc cupping and is not a glaucoma diagnosis. `MYA` is an image appearance label, not a refraction measurement. Source label counts are aggregate image counts in the official partitions.

| Code | Source label | Interpretation for this project | Train positive | Validation positive | Test positive | Planned handling |
|---|---|---|---:|---:|---:|---|
| DR | Diabetic retinopathy | Disease | 376 | 132 | 124 | Supported label |
| ARMD | Age-related macular degeneration | Disease | 100 | 38 | 31 | Supported label |
| MH | Media haze | Image quality / visibility | 317 | 102 | 104 | Quality finding, not disease |
| DN | Drusen | Retinal image finding | 138 | 46 | 46 | Finding, not AMD diagnosis |
| MYA | Myopia | Source-defined appearance / phenotype | 101 | 34 | 32 | Not a refractive diagnosis |
| BRVO | Branch retinal vein occlusion | Disease | 73 | 23 | 23 | Supported label |
| TSLN | Tessellation | Fundus appearance finding | 186 | 65 | 53 | Finding, not a disease label |
| ERM | Epiretinal membrane | Disease | 14 | 7 | 5 | Low support; exploratory |
| LS | Laser scars | Prior-treatment finding | 47 | 17 | 15 | Finding, not active disease |
| MS | Macular scars | Retinal image finding | 15 | 5 | 7 | Low support; exploratory |
| CSR | Central serous retinopathy | Disease label | 37 | 11 | 13 | Low validation support |
| ODC | Optic-disc cupping | Optic-disc morphology | 282 | 72 | 91 | Not a glaucoma diagnosis |
| CRVO | Central retinal vein occlusion | Disease | 28 | 8 | 9 | Low validation/test support |
| AH | Asteroid hyalosis | Ocular finding | 16 | 4 | 5 | Low support; exploratory |
| ODP | Optic-disc pallor | Optic-disc finding | 65 | 26 | 24 | Finding, not an etiologic diagnosis |
| ODE | Optic-disc edema | Optic-disc finding | 58 | 21 | 17 | Finding, not an etiologic diagnosis |
| AION | Anterior ischemic optic neuropathy | Disease | 17 | 5 | 4 | Very low support; exploratory |
| PT | Parafoveal telangiectasia | Disease label | 11 | 2 | 4 | Insufficient validation/test support |
| RT | Retinal traction | Retinal finding / pathology label | 14 | 6 | 5 | Low support; exploratory |
| RS | Retinitis | Disease label | 43 | 14 | 14 | Supported for exploratory analysis |
| CRS | Chorioretinitis | Disease label | 32 | 11 | 11 | Low validation support |
| EDN | Exudation | Retinal finding | 15 | 5 | 4 | Low support; exploratory |
| RPEC | Retinal pigment epithelium changes | Retinal finding | 22 | 6 | 4 | Low support; exploratory |
| MHL | Macular hole | Disease label | 11 | 3 | 3 | Very low support; exploratory |

Twenty-four of 45 source label columns meet the 10/10 training floor. The other 21 are not trained as supported outputs in this run. Meeting a training support floor does not guarantee adequate validation/test support or a usable classifier.

## Training and threshold policy

- Architecture: shared torchvision ConvNeXt-Tiny, 320-pixel input, initialized from the epoch-14 RFMiD checkpoint (SHA-256 `f98c7dcaf67409805a4564057a205bef8d65b884effd750d569597bcb70a35f8`).
- Training: AdamW, learning rate `2e-5`, weight decay `2e-4`, batch size 32, two data-loader workers, source-prescribed mild geometric/color augmentation, positive weighting `sqrt(negative/positive)` clipped to `[1,10]`.
- Loss mask: at least 10 positive and 10 negative training images per label.
- Early stopping: validation binary cross-entropy, patience 5, minimum improvement `1e-5`; maximum training time 10,800 seconds under the original cumulative campaign guard.
- Operating point: only labels with at least 10 positive and 10 negative validation images may receive a validation-selected maximum-sensitivity threshold subject to at least 90% validation specificity. Lower-support validation labels remain at a fixed 0.5 threshold and are explicitly marked as not threshold-selected. The test set is not used for threshold selection.
- Run outputs: model checkpoint, aggregate JSON and guarded log remain on the VM until checksum-verified closeout. Checkpoint and row-level outputs are not for GitHub.

## Broader condition coverage boundary

Existing complementary results remain separate: the Zhejiang UWF campaign evaluated six mutually exclusive source diagnoses (AMD, DR, pathological myopia, retinal detachment, RVO and uveitis) with one shared ConvNeXt-Tiny model; the HYGD work compared fundus encoders for its GON+/GON− glaucoma-suspect label on development validation. These results do not combine into a validated universal retinal-disease classifier. RFMiD findings such as disc cupping, drusen, pallor and edema must not be re-labeled as glaucoma, AMD, or other diagnoses.

Conditions requiring OCT, angiography, slit lamp, corneal topography, or other unavailable acquisition remain outside this fundus-only scope. No evidence in this campaign establishes clinical benefit, safety, calibration for real-world risk, or performance on Golden Retina hardware.

## References

- RFMiD dataset description: Pachade et al., *Retinal Fundus Multi-Disease Image Dataset (RFMiD): A Dataset for Multi-Disease Detection Research*, [Data 6(2), 14 (2021)](https://doi.org/10.3390/data6020014).
- RFMiD multi-disease challenge report: [Medical Image Analysis (2024)](https://www.sciencedirect.com/science/article/pii/S1361841524002901).
- Staged mirror and declared terms: [ctmedtech/RFMID dataset card](https://huggingface.co/datasets/ctmedtech/RFMID), revision `16ee0cf528e9cb6ec38a6d7b1f3e4580172bc1da`; the mirror declares CC BY 4.0. The run report records the exact mirror and license statement.
