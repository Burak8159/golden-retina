# Abnormality, lesion and image-quality output gate

## Decision

Do not ship a new lesion, image-quality, or generic “abnormal” head from this campaign. The current checked, patient-grouped Zhejiang classifier is trained on six mutually exclusive disease labels and produces disease-specific outputs. It is not a lesion detector, quality grader, or calibrated global disease-risk model.

## Available annotation evidence

- **Zhejiang UWF:** disease-category labels are staged and used; source data descriptor includes image-quality assessments, but a verified quality-label join and a quality-specific frozen train/validation evaluation were not part of this campaign's model manifest. Do not infer a quality score from disease probabilities.
- **DDR:** 3,002 DR images include gradability labels, but this archive is training-only here and lacks verified patient grouping. There is no patient-independent validation for a new quality head.
- **MMRDR-UWF:** seven lesion annotations are described in the project’s prior dataset plan, but the corresponding image/label archive is not staged in this active workspace. No MMRDR lesion training was run.
- **RFMiD legacy outputs:** sparse finding labels include severe miss patterns (for example, all five ERM positives and 45/46 drusen positives were missed in the recorded historical image-level run). The split did not establish patient independence and those findings are not interchangeable with disease diagnoses.

## Product and model contract

- Expose the six disease scores only as research model outputs, with disease name, model/version, and a clear non-diagnostic boundary.
- For absent or unsupported targets, return “not evaluated / unsupported,” not a fabricated percentage.
- If image quality has not been separately measured by a validated quality head, the API and mobile apps must not imply a scan is gradable. Preserve a manual/unknown state and defer clinical interpretation.
- Keep morphological findings such as drusen, cupping, pallor, scars, tessellation, and traction distinct from named diseases. Do not map them to AMD, glaucoma, or other diagnoses without appropriate labels and evaluation.

## Gate to reopen

A future quality or lesion head requires a verified annotation-to-image join, permissible source-specific use, patient/source-aware split, adequate examples per class, and a validation cohort that was not used for selection. If a clean split cannot be formed, stop at a descriptive data audit.
