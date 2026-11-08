# Golden Retina Stage 36 closeout

## Research results

The RETFound-Green source-invariance candidate was frozen as a **research comparator**, not a clinical model. On the exposed patient/source-grouped validation set (314 groups, 150 positive), seed 1 at a fixed 0.5 threshold produced 89.81% accuracy, 90.00% sensitivity, 89.63% specificity, AUROC 0.9491, and AUPRC 0.9546. Seed 2 reduced specificity to 72.56% while ranking metrics stayed similar; the apparent operating-point gain did not reproduce. The candidate-versus-control AUROC/AUPRC differences were within bootstrap uncertainty.

On 200 images from 152 patients in a **previously exposed** Zhejiang subset (53 source-labeled DR, 99 healthy), the frozen score ranked source-defined DR diagnosis versus healthy with patient AUROC 0.9897 (95% CI 0.9773–0.9981) and AUPRC 0.9826 (0.9629–0.9967). No threshold was evaluated. This is not referable-DR validation, independent external validation, clinical validation, or diagnosis certification.

Supported research labels in the staged project are DR and HYGD glaucoma, with quality supervision only where labels exist. The work does not establish AMD, RVO, “any retinal disease,” clinical benefit, or autonomous diagnosis.

## Research API

A versioned, bearer-authenticated, loopback-only API was implemented and exercised with synthetic images. Authentication, inference response, clinical-use rejection, corrupt/oversized upload rejection, interrupted-upload recovery, and restart/reload passed. It was then stopped during cloud closeout. The endpoint is not a public or clinical service. RETFound-Green terms remain limited to internal non-commercial research; weights are not redistributed.

## Release and artifacts

Commit `1d3dacf` was pushed to `https://github.com/Burak8159/golden-retina` and the remote `main` SHA was verified. It contains 64 selected source, documentation, tracker, and aggregate-result files. Raw images, row-level predictions, secrets, and model checkpoint binaries were excluded. Both frozen model weights were copied to `models/private_research/retfound_green_20261108/`; `.gitignore` excludes `.pt` and `.pth` files, and their SHA-256 hashes match the freeze record.

The release file hashes are listed in `results/technical_dev_20261108_rerun/public_artifact_manifest.json`.

## Cloud shutdown and cost

The Golden Retina inference process was stopped and `golden-retina-l4-01` is `TERMINATED`. Project inventory showed no reserved external IP, forwarding rule, or Cloud Storage bucket. The 150-GB persistent disk remains `READY` so private dataset/cache contents are preserved; it may continue to incur storage charges. Estimated compute spend through shutdown is **₺407.13** at the configured accrual rate; this excludes persistent-disk storage, tax, network charges, and any later settled invoice.

The local progress tracker remains available at `http://127.0.0.1:8769/`. The cloud inference endpoint is off with the VM.

## Remaining scope limits

At the user’s direction, the four gated or unavailable candidate stages (original stages 16–19: EyeCLIP, RET-CLIP, EyeFound, and RFM) were removed from the active plan. No gated weights were accessed. The tracker now presents a 34-stage active plan; completed work after the removed slots is renumbered for the live view, while original stage IDs are preserved in `removed_blocks` for audit history. The local checkout still contains unrelated pre-existing changes that were deliberately left unstaged.
