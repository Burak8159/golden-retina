# RFMiD expanded shared-model run: descriptive summary

Run `rfmid1_expanded10_v2_20261121`; one ConvNeXt-Tiny multi-label model initialized from the earlier shared checkpoint. RFMiD 1.0 official image partitions: 1,920 train / 640 validation / 640 test; 320-pixel input; 24 labels met the training floor of at least 10 positives and 10 negatives; validation-BCE early stopping selected epoch 10.

## Aggregate image-split metrics

| Measure | Result | Meaning / caveat |
|---|---:|---|
| Macro AUROC | 94.2% | 24 train-supported labels with evaluable positive and negative examples |
| Macro average precision | 57.0% | Prevalence-sensitive; across supported labels with test positives |
| Element-wise multilabel accuracy | 75.3% | Over all image-label decisions; includes many negative labels |
| Mean per-label accuracy | 95.1% | Average accuracy across the 24 train-supported labels |
| Exact-set accuracy | 0.0% | Fraction of images with all 45 source-label decisions exact; very strict for sparse multilabel output |

These are image-level descriptive results on RFMiD's supplied partition. Patient IDs are absent in the staged label tables, so patient-level disjointness is unverified. This official image-test partition has previously been exposed in this project; these numbers are development evidence, not independent external validation or clinical performance. Interpret the high mean per-label accuracy with each label's sensitivity, specificity, prevalence, and false-negative count.

## Disease label results

Thresholds were selected on validation only to maximize sensitivity subject to at least 90% validation specificity, and only when validation had at least 10 positives and 10 negatives. Otherwise threshold 0.5 was retained. Test metrics were not used for selection.

| RFMiD label | Train + | Val + | Test + | Threshold | Sensitivity | Specificity | Accuracy | AUROC | AP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Anterior ischemic optic neuropathy (AION) | 17 | 5 | 4 | 0.500 * | 25.0% | 98.4% | 98.0% | 90.8% | 18.5% | 3 |
| Age-related macular degeneration (ARMD) | 100 | 38 | 31 | 0.147 | 87.1% | 92.1% | 91.9% | 97.0% | 68.3% | 4 |
| Branch retinal vein occlusion (BRVO) | 73 | 23 | 23 | 0.079 | 87.0% | 90.4% | 90.3% | 93.4% | 78.6% | 3 |
| Chorioretinitis (CRS) | 32 | 11 | 11 | 0.076 | 90.9% | 91.9% | 91.9% | 96.3% | 42.7% | 1 |
| Central retinal vein occlusion (CRVO) | 28 | 8 | 9 | 0.500 * | 55.6% | 99.8% | 99.2% | 93.7% | 79.0% | 4 |
| Central serous retinopathy (CSR) | 37 | 11 | 13 | 0.051 | 84.6% | 96.0% | 95.8% | 98.4% | 70.4% | 2 |
| Diabetic retinopathy (DR) | 376 | 132 | 124 | 0.310 | 95.2% | 94.8% | 94.8% | 98.3% | 93.9% | 6 |
| Epiretinal membrane (ERM) | 14 | 7 | 5 | 0.500 * | 0.0% | 100.0% | 99.2% | 79.8% | 14.3% | 5 |
| Macular hole (MHL) | 11 | 3 | 3 | 0.500 * | 0.0% | 100.0% | 99.5% | 96.7% | 46.1% | 3 |
| Parafoveal telangiectasia (PT) | 11 | 2 | 4 | 0.500 * | 0.0% | 99.8% | 99.2% | 97.2% | 14.9% | 4 |
| Retinitis (RS) | 43 | 14 | 14 | 0.088 | 85.7% | 97.1% | 96.9% | 98.0% | 81.3% | 2 |

* Threshold was fixed at 0.5 because validation positive support was below the minimum. Small test positive counts make sensitivity unstable; scores are not calibrated clinical risk probabilities.

## Important interpretation

- RFMiD source label **MH means media haze**, not macular hole; **MHL means macular hole**. Several channels (drusen, tessellation, optic-disc cupping, media haze, edema, exudation, scars, vessel appearance) are source-defined findings or markers, not standalone diagnoses.
- RFMiD optic-disc cupping is not a glaucoma diagnosis: the dataset paper notes cupping may arise from glaucoma or other conditions. No glaucoma disease classifier was trained in this run.
- Rare labels below the train/validation support floors were not threshold-tuned; they remain exploratory.
- The dataset paper describes expert ophthalmologist labels based on medical records and visual fields with adjudication. Most subjects were dilated before acquisition. This does not establish equivalence to the planned device or deployment population.

## Reproduction provenance

- Architecture: `torchvision.convnext_tiny`; input 320; seed 20261021; best epoch 10.
- Dataset mirror revision: `16ee0cf528e9cb6ec38a6d7b1f3e4580172bc1da`; mirror declares CC BY 4.0. Upstream description: [Pachade et al., Data 2021, 6(2), 14](https://doi.org/10.3390/data6020014). Mirror card: [ctmedtech/RFMID](https://huggingface.co/datasets/ctmedtech/RFMID).
- Initialization checkpoint SHA-256: `f98c7dcaf67409805a4564057a205bef8d65b884effd750d569597bcb70a35f8`.
- Selected checkpoint SHA-256: `fd094aca13d459dae4155f184a955a5d41fefb3db9c8ce785d458439741a2cdc`.
- Full 45-label aggregate table: `rfmid_expanded10_v2_test_metrics.csv`. Raw images and row-level predictions are excluded.
