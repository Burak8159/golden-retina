# Golden Retina VM recovery and cost gate

> **Historical preflight record.** This document preserves the decisions and failed capacity attempts made earlier on 2026-11-21. Its “Current state,” recovery sequence, and budget paragraphs describe that earlier checkpoint, not the final campaign state. Final campaign results and cloud shutdown are in `reports/cloud_compute_shutdown_20261121.md`; the campaign limits were later updated to ₺1,900 planned + ₺80 protected margin (₺1,980 absolute). The training run finished, artifacts were pulled and hashed, and all three named VMs are now `TERMINATED`. The attached disks and snapshot were retained. Do not use the old “start/recovery” instructions as the current next action.

**Observed:** 2026-11-21
**Project:** `golden-retina-dev-20261107`
**VM:** `golden-retina-l4-01`, `g2-standard-4`, `us-west1-a`

## Current state

- The original VM `golden-retina-l4-01` remains `TERMINATED`; its start attempt in `us-west1-a` failed with `ZONE_RESOURCE_POOL_EXHAUSTED` for an NVIDIA L4 stockout.
- Its attached 150 GB `pd-standard` boot disk is `READY`. Preserve it because it may hold prior datasets, environments, or artifacts.
- A fresh read-only inventory found exactly one Compute Engine VM, the stopped Golden Retina L4, and one attached 150 GB disk; the project bucket listing returned empty. The legacy `gcloud storage` command itself is broken in this local SDK (`ssl.PROTOCOL_TLSv1`), so the bucket check used `gsutil` instead.
- The original instance metadata has a maximum runtime of 27,000 seconds (7.5 hours) and a STOP termination action. L4 quota is 0/1 used in `us-west1`; L4 hardware is advertised in `us-west1-b` and `us-west1-c`, but quota and advertisement do not guarantee capacity.
- The boot disk is marked `autoDelete` on **instance deletion**; it remains present while the instance is merely stopped. The source disk is untouched. The same-project snapshot `golden-retina-l4-01-zone-copy-20261121` is now `READY`.
- The active Google identity can read the Golden Retina project and billing account. Cloud Billing and Resource Manager APIs are now enabled for project metadata reads; project-to-billing linkage is confirmed. The existing ₺1,400 monthly budget is an alert budget, not a hard cap. The Billing Budgets API returns alert thresholds but not current accrued charges. No BigQuery dataset is listed in the Golden Retina project, so no project-local billing-export table is available to query.
- Authenticated Cloud Console Billing report scoped to `golden-retina-dev-20261107` shows **TRY 0.00 net total** and **TRY 0.00 forecast** for Nov 1–20, 2026. Gross Compute Engine usage of TRY 3,925.66 and Networking usage of TRY 6.43 were fully offset by TRY 3,932.09 in savings. Nov 21 same-day billing may post later; do not count the savings as guaranteed for future usage. No project BigQuery datasets or export table exist. L4 requests failed in `us-west1-a/b/c`, `us-central1-a/b/c`, `us-east4-a/c`, `us-east1-b/c`, and standard/Spot requests in `us-east1-d`. The unattached `us-east1-d` disk clone was deleted after confirming no VM used it. A 150 GB snapshot-derived boot disk is now `RESTORING` in `us-east1-c` for a fresh Spot attempt; no replacement VM exists. No dataset transfer or training has occurred.

## Current list-price scenario (not a bill)

Using Google's published on-demand `g2-standard-4` rate of **$0.706832276/hour**, 8 VM-hours are approximately **$5.65**. Google's standard persistent-disk rate is **$0.000054795/GiB-hour**; a conservative 150-GiB, 30-day disk scenario is approximately **$5.92**. The current snapshot metadata reports **53,906,299,264 stored bytes (50.204 GiB)**; budgeting that size for 24 hours, plus 8 hours of replacement-disk overlap and up to 8 GiB of outbound artifact transfer, yields approximately **₺652 before a 25% stress allowance (≈₺816 total)** at the 2026-11-20 CBRT USD selling rate of ₺50.4936 per dollar. This is an estimate, not an actual bill; it excludes tax, account-specific billing adjustments, and unrelated prior usage. Keep snapshot retention and outbound transfer within the stated bounds; stop before the ₺850 planned cutoff and never cross the ₺1,000 absolute ceiling. Recalculate using the selected region's published rate before training if it is higher.

Sources: [Google accelerator-optimized pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized), [Google disk pricing](https://cloud.google.com/compute/disks-image-pricing), and [CBRT 2026-11-20 indicative rates](https://www.turkiye.gov.tr/doviz-kurlari?v=1.0.20).

## Repair sequence after billing clearance

1. Use the authenticated Nov 1–20 billing report as the latest baseline, but recheck it before longer compute because same-day charges may lag. The limits remain 8 aggregate GPU-hours, ₺850 planned, and ₺1,000 absolute. Keep a 25% cost stress allowance and a bounded outbound artifact-transfer allowance; do not use the alert budget as a hard cap.
2. Start only the named Golden Retina VM, then run `scripts/cloud/vm_preflight.sh` remotely. It checks the GPU and Python environment, verifies the existing workspace, creates only missing output directories, reports disk capacity and existing top-level data directory names, and performs no download, package installation, or training.
3. If `/data/golden-retina/workspace` is absent, stop and inspect/mount the preserved disk correctly. Do not create a replacement workspace over unknown old data.
4. Repair missing output directories (including the previously missing `models/` directory) idempotently before any training command. Save all logs, checkpoints, configs, manifests, and results on the VM disk.
5. Download/extract datasets only on the VM, after per-source access and terms checks. Keep image archives and row-level outputs private.
6. Use a campaign timer and cost checkpoints in addition to the VM's 7.5-hour maximum runtime. Stop early if the planned-spend threshold is approached; never use the protected margin as a target.
7. Close out in order: stop jobs, create remote checksum inventory, pull all authorized outputs and verify hashes, push only permitted code/docs/aggregates to GitHub, then stop the VM and verify it is `TERMINATED`. Keep the persistent boot disk unless its contents have been recovered and its deletion is explicitly safe.

## Capacity-recovery event

Google returned `ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS` for NVIDIA L4 in all three `us-west1` zones, all three `us-central1` zones, both `us-east4` zones, `us-east1-b/c`, and standard/Spot requests in `us-east1-d`. Failed requests left no VM; the unattached `us-east1-d` clone was deleted. A 150 GB boot disk restored from the READY snapshot is now `RESTORING` in `us-east1-c` for a fresh Spot-capacity attempt. The original VM/disk and snapshot remain preserved. Confirm boot, GPU, and workspace before research; recalculate cost for the selected region before training.

## Fresh project-scoped inventory

- One temporary preservation snapshot is `READY`; no reserved external addresses were found. The source VM has an ephemeral external NAT configuration while running.
- The authenticated billing report is scoped to the Golden Retina project. The report covers Nov 1–20, 2026; same-day Nov 21 usage is not yet reflected. Gross usage was fully offset by savings, so future list-price estimates deliberately assume no such credits.
- The VM’s hard runtime setting is 27,000 seconds (7.5 hours) with STOP action. This bounds one boot but does not itself enforce the total ₺850/₺1,000 project budget; the campaign ledger and egress cap remain required.

## Prepared remote check

The preflight script exists at `scripts/cloud/vm_preflight.sh`. The aggregate-only, VM-enforced mBRSET schema audit is prepared at `scripts/cloud/audit_mbrset_metadata.py`; it refuses to run on macOS, restricts paths to `/data/golden-retina`, suppresses small counts, and never writes filenames or patient IDs. Neither script has been run against cloud data. They will be used only after the relevant billing/access gates clear.

## Dataset-only-on-VM decision

No dataset archives or extracted images are to be stored on this Mac. The source audit is metadata-only and is recorded in `docs/dataset_candidate_screen_20261121.md`. Dataset acquisition starts only after the cost gate and exact source-specific access checks pass.

## Budget update (2026-11-21)

The current campaign ceiling supersedes the earlier preflight values: **₺1,900 planned operating spend + ₺80 protected margin = ₺1,980 absolute cap**. The eight aggregate VM/GPU-hour cap and 7.5-hour instance STOP guard remain. The ≈₺816 full-run stress estimate is not an invoice; same-day and future charges may lag. The local tracker and remote guard config were updated to this user-set budget.
