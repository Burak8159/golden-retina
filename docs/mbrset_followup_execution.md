# Golden Retina mBRSET follow-up execution

## Phase order and initial-program spend limit

The current initial program has 36 blocks and no independent wall-clock deadline. Continue until block 36 closeout or until the ₺1,900 planned spend ceiling is reached, preserving the ₺80 contingency; ₺1,980 is the absolute operational maximum. This is separate from the mBRSET phase budget below. Use one Golden Retina L4 VM; RTX remains unauthorized. Cloud billing is delayed, so the tracker uses a conservative accrued-spend estimate and the VM must be stopped before the planned ceiling is crossed.

## Start condition and order

Burak will notify the project when the mBRSET files arrive. Record the arrival and stage the files in private, access-controlled project storage. Do not start the mBRSET work until the initial 36-block Golden Retina program has finished and its artifact pull, checksum, permitted-publication, and cloud shutdown checks have passed. Then begin the mBRSET follow-up immediately.

## mBRSET scope and limits

- Maximum duration: 12 hours.
- Planned compute/operations ceiling: ₺1,400.
- Contingency: up to ₺75 for justified completion work; absolute maximum ₺1,475.
- Continue the existing Golden Retina software suite and compatible heads; do not add a new model family.
- First audit patient and eye identifiers, label semantics, view pairing, duplicates, source overlap, and compatibility with existing inputs/outputs.
- Evaluate compatible frozen models first. Any adaptation makes mBRSET development data for that model version; results from it are not independent confirmation after tuning.
- Stop at 12 hours or ₺1,475, whichever comes first. Record actual runtime and billing estimate, noting billing delays and excluded charges.

## Data handling and repository contents

Keep raw images, source archives, and row-level predictions in authorized private storage. The GitHub repository may contain code, configuration, data-free manifests/schema descriptions, hashes where safe, run cards, and aggregate findings only when the relevant terms permit. Never commit the raw dataset or identifying/row-level records to the public repository.

## Closeout

Save checkpoints and logs privately; pull and checksum permitted artifacts; update the tracker and this execution record with counts, split provenance, per-task metrics, uncertainty, failure cases, compute time, and spend. Push only reviewed, permitted repository artifacts. Shut down and verify only Golden Retina cloud resources; leave unrelated projects untouched.
