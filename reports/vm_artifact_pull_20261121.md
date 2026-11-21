# VM artifact pull record

## Earlier L4-02 closeout

- **VM:** `golden-retina-l4-02` (`golden-retina-dev-20261107`, `us-east1-c`)
- **Remote archive:** `/tmp/golden_retina_vm_closeout_20261121.tar.gz`
- **Archive size:** 2,163,469,488 bytes
- **Archive SHA-256:** `31511d04d238b6eaf6dd2706437daebb1e798f95c060d325b07285d98c2dfef5`
- **Archive members:** 757 (701 regular files after extraction; remaining members are directory entries)
- **Verification:** local archive SHA-256 exactly matches the VM-computed digest; extraction completed; private output remains under the Git-ignored `private_vm_artifacts/20261121/` directory.
- **Included:** code snapshot, scripts, configuration, docs, data manifests, reports, result aggregates, row-level development outputs, training logs, and model/checkpoint artifacts.
- **Excluded from workstation transfer:** the external dataset/model-cache tree (8,873 files, approximately 25.4 GB) to honor the prior VM-only storage instruction. It remains on the VM's attached persistent storage until the user explicitly authorizes a different disposition.
- **Publication boundary:** the private archive and extracted records include dataset-derived row-level material and checkpoints. They are not GitHub content. The public update will be limited to source, protocols, documentation, and permitted aggregates.

This record verifies the archive transfer, not a scientific reproduction from a clean clone. See the evidence ledger for split and exposure limitations.

## Current RFMiD run — L4-03

- **VM:** `golden-retina-l4-03` (`golden-retina-dev-20261107`, `us-central1-a`)
- **Remote archive:** `/tmp/rfmid1_l4_03_closeout_20261121.tar.gz`
- **Archive size:** 103,357,549 bytes
- **Archive SHA-256 (remote and local):** `f729c4b374cd0463f9bf463f1b6aa59efb091420fb59cdca6e1cc790dce6ee60`
- **Local private destination:** `private_vm_artifacts/20261121/rfmid1_l4_03/` (Git-ignored)
- **Members:** selected ConvNeXt-Tiny checkpoint, final run summary, training log, and exact training script snapshot. No raw images or dataset archive were pulled.
- **Member verification:** checkpoint `f98c7dcaf67409805a4564057a205bef8d65b884effd750d569597bcb70a35f8`; summary `62bb9d137711fad341ea0088c2ae0850b41f58fa8b891a545a7f6995c3b3ff05`; log `7696f5fee091a2a7e66b672658ca6fa54bd77450cd2b598d4c05c3c705a32437`; exact run script `5ab6c506da8b59bf55c5316cfad4fdb787f277cb2dce4745cca36cf001212aaf`. The VM-side hashes matched each local extracted file.
- **Public trainer relationship:** the public `src/train_rfmid_multilabel.py` preserves the run's training and evaluation logic. Its post-run difference from the captured script is limited to the dataset description and declared source/license metadata in the emitted run summary; the run snapshot remains privately archived byte-for-byte.
- **Public release boundary:** only the source, protocol, and aggregate-only disease report are candidates for GitHub. Checkpoint, full raw run summary, and detailed log stay private.

The archive closes the local artifact transfer for the RFMiD run. It does not establish patient-level independence, clinical validation, or a clean-clone reproduction. Final safe GitHub publication and named-VM shutdown are recorded in `reports/cloud_compute_shutdown_20261121.md`.

## Glaucoma recovery — L4-03 final closeout

- **Safe aggregate files pulled:** RIM-ONE hospital audit and the two aggregate-only run summaries are in Git-ignored `private_vm_artifacts/20261121/rimone_aggregates/`.
- **Checksum verification:** local hashes match the corresponding L4 files: `summary_aggregate.json` `09dae748244e01862c4f2e9c47851dc0b0b4b402adf9baeb585b8e316742e193`; `scratch_summary_aggregate.json` `26d86c4c6fe0a32db6c78c68d7b583e9b2395c06f02af7e785584342728d6a9b`; audit `rimone_hospital_audit_20261121.json` `b215b2ce9bbdb3e620eb5c0f1d0c253faa2b3da8372d6e1dc079c6ef2f04b20d`.
- **Code verification:** the two glaucoma trainer files match the L4 workspace byte-for-byte. The local split-audit helper was not present in the remote tree; its source and audit logic are preserved in the repository.
- **Not transferred:** RIM-ONE image archive, row-level split manifests, per-image predictions, and both model checkpoints remain on the VM/disk because the source release prohibits redistribution. The expanded RFMiD full run summary and training logs were not copied because they contain per-image or private provenance material; the safe aggregate-only report/CSV already exists locally.
- **Scope:** this is a safe aggregate/code pull, not a claim that every remote file was copied. Restricted/raw artifacts remain on preserved project storage pending any future source-authorized handling.
