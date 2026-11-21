# Golden Retina cloud compute shutdown record

> **Interim closeout snapshot.** This record reflects the shutdown verified after the first RFMiD campaign closeout. The user then authorized a disease-coverage extension; `golden-retina-l4-03` was restarted for guarded phase 34 on 2026-11-21. Therefore the shutdown statements below are a valid historical checkpoint but are not the current live state until the extension's final closeout is appended.

**Project:** `golden-retina-dev-20261107`
**Checked:** 2026-11-21
**Scope:** Golden Retina Compute Engine inventory and related resources visible to the available Google Cloud account.

## Compute shutdown

After the RFMiD 1.0 run completed and its archive was transferred and checksum-verified, the final remote process scan found no training/evaluation process. The NVIDIA L4 reported 0% utilization and 0 MiB allocated. `golden-retina-l4-03` was stopped and its state was polled to `TERMINATED`.

The project-scoped instance listing then showed all three named Golden Retina VMs as `TERMINATED`:

| Instance | Zone | Final state |
|---|---|---|
| `golden-retina-l4-01` | `us-west1-a` | TERMINATED |
| `golden-retina-l4-02` | `us-east1-c` | TERMINATED |
| `golden-retina-l4-03` | `us-central1-a` | TERMINATED |

No reserved addresses or forwarding rules were returned by the corresponding project-scoped list commands. No unrelated project was changed.

## Storage retained

The project still contains three `READY` persistent disks and one `READY` snapshot:

- `golden-retina-l4-03-boot` in `us-central1-a` (attached to the stopped instance)
- `golden-retina-l4-02-boot` in `us-east1-c` (attached to the stopped instance)
- `golden-retina-l4-01` in `us-west1-a` (attached to the stopped instance)
- `golden-retina-l4-01-zone-copy-20261121`, a 150 GB snapshot (observed stored size in the prior inventory: 53,906,299,264 bytes, about 50.2 GiB)

These data-bearing resources were deliberately retained. Persistent storage can continue to incur charges after VM shutdown. No disk or snapshot was deleted.

## Verification limits and cost

The campaign runtime estimate is about 7.37 L4-hours and approximately ₺337.46 at the recorded ₺45.8/hour estimate. This is not a settled invoice and excludes any retained-storage charges, delayed billing, tax, or unmetered services.

Cloud Asset Inventory was disabled and the active account lacked permission to enable it. Cloud Storage listing was unavailable in the local CLI. Cloud Run, Functions, App Engine, GKE, Cloud SQL, and Scheduler APIs were not enabled, so their resources could not be listed without additional access and enabling services. No such API was enabled. This record verifies that all three known Golden Retina VM instances are stopped; it does **not** establish that every Google Cloud service is absent or that the project has zero ongoing cost.

## Final phase-43 verification — 2026-11-21 18:12 UTC

The disease-coverage extension ran after the earlier interim closeout above. The final remote scan found no active training job; `nvidia-smi` showed 0 MiB allocated and 0% utilization. The final explicit stop completed, and a fresh project instance listing verified:

| Instance | Zone | Final state |
|---|---|---|
| `golden-retina-l4-01` | `us-west1-a` | TERMINATED |
| `golden-retina-l4-02` | `us-east1-c` | TERMINATED |
| `golden-retina-l4-03` | `us-central1-a` | TERMINATED |

The temporary `golden-retina-temp-nat-20261121` and `golden-retina-temp-egress-20261121` were deleted after the Mendeley API returned HTTP 401. Project router listing returned no remaining routers. No unrelated projects were changed.

At closeout, the tracker’s cumulative runtime estimate was about 9.54 cumulative L4-hours at the conservative ₺45.80/hour rate (approximately ₺436.79 compute estimate). This is not the settled invoice and excludes retained disks/snapshot, network, tax, delayed billing, and any services outside the visible inventory. Three persistent disks and one snapshot remain READY to preserve data; they can continue to incur storage charges. Cloud Asset Inventory and some service inventories were unavailable, so this verifies the three named Compute Engine VMs and temporary router cleanup, not that every possible Google Cloud service is absent or the project has zero ongoing cost.

Phase 43 is complete only for the declared closeout scope: permitted aggregate/code transfer, safe GitHub sync, and verified stop of all three named Golden Retina VMs. Restricted source data and checkpoints remain on preserved project disks.
