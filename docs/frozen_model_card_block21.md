# Frozen paired-view model card — Block 21

## Model identity

- Product/project: Golden Retina research system.
- Version: `golden-retina-convnext-tiny-paired-block21-2026-11-08`.
- Backbone: ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k`), fine-tuned with
  paired-view fusion and task heads.
- Checkpoint SHA-256:
  `61f2e891a8d71e186876fa2dec437afa73b0970f33e92412850af7c24d19c926`.
- Input: one or two fundus images per eye. The main development evaluation used
  paired views; single-view output is a compatibility mode, not an independently
  validated use condition.
- Outputs: five DR-grade probabilities (grades 0–4), a referable-DR risk score,
  and an ungradable-risk score. Higher ungradable-risk values indicate greater
  predicted ungradability.

## Intended use and limits

This is a retrospective research artifact for model-transfer and workflow
experiments. It is **not clinically validated**, is not autonomous diagnosis,
and must not guide patient care. The API blocks a clinical intended-use value
and marks all outputs non-actionable. The quality head was trained on the
available DDR grade-5 ungradable labels; it is not a universal image-quality
standard.

## Internal development operating point

The validation-selected policy was assessed on 938 internal development cases:
436 referable DR, 369 gradable nonreferable, and 133 labeled ungradable cases.
Observed results were 78.04% accuracy, 89.68% referable sensitivity, 92.41%
specificity among gradable nonreferable cases, and 100% case-action coverage
because ungradable cases were routed to review/referral. It missed **45 of 436
referable cases**. These are exposed, validation-selected development results;
they do not support a zero-miss, autonomous, or clinical-safety claim.

## Reserved Zhejiang source-transfer analysis

The frozen checkpoint was run once on the reserved Zhejiang UWF source (700
images, 602 patients, 689 eyes). The source labels distinguish DR diagnosis
from healthy control, not referable-DR grades. Image counts differ by class.
For DR-versus-healthy ranking, patient-mean aggregation gave AUROC 0.705 (95%
patient-bootstrap interval 0.609–0.795); one deterministic eye gave 0.793
(0.705–0.871); and maximum-over-eyes gave 0.881 (0.818–0.937). The max result
can be inflated by unequal image counts: 53 DR patients contribute 100 images
(47 bilateral), while 99 healthy patients contribute 100 images. These are
exploratory score-transfer estimates, not referable-DR external validation,
binary diagnostic performance, calibration, or disease-specific models.

Other category-versus-healthy AUROCs from the same DR-trained risk score are
exploratory only: AMD 0.334, pathologic myopia 0.683, retinal detachment 0.917,
retinal vein occlusion 0.908, and uveitis 0.695 under patient-mean scoring.
They do not mean the software has validated heads for those diseases.

Overall-quality transfer was also mismatched: the Zhejiang poor/good label is
not identical to DDR grade-5 ungradability. The frozen quality score had AUROC
0.688 and, at the DDR-derived threshold, 28.2% sensitivity for poor quality and
91.9% specificity for good quality. Treat this as exploratory only.

## Data and reproduction notes

The raw images, row-level manifests/predictions, and checkpoint remain in
private, access-controlled storage and are not included in this public release.
Reproduction requires separately obtaining each source under its own terms.
The run cards, aggregate tables, evaluation code, and reported hashes are
provided where redistribution permits. No patient-level certainty or
prospective benefit is inferred from this retrospective analysis.

The Zhejiang dataset is described in the [source dataset paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11579006/).
