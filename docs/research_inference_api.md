# Golden Retina research inference API

This service packages the frozen Block 21 paired-view ConvNeXt-Tiny checkpoint
for controlled research demonstrations. It does **not** provide a clinical
diagnosis and must not guide patient care.

## Frozen contract

- Version: `golden-retina-convnext-tiny-paired-block21-2026-11-08`.
- Checkpoint SHA-256: `61f2e891a8d71e186876fa2dec437afa73b0970f33e92412850af7c24d19c926`.
- Accepts one or two image uploads, with a 25 MiB per-image limit.
- Returns grade 0–4 probabilities, a referable-DR risk score, a DDR
  ungradable-risk score (higher means greater predicted ungradability), a
  research routing flag, and the model version/hash.
- The application does not retain uploaded images. The multipart framework may
  temporarily spool large requests under the service's isolated temporary
  directory; temporary upload files are closed and removed with each request.
- `/v1/infer` and `/v1/model-card` require a bearer token checked before body
  parsing. Bind the service to loopback and access it through an authenticated
  tunnel; do not expose it on a public interface.
- A non-research `intended_use` is rejected. Thresholds were selected on
  exposed internal validation and are shown only as research routing flags.

## Run on the dedicated L4 VM

Install `fastapi`, `uvicorn`, and `python-multipart` in the project virtual
environment. Store `GOLDEN_RETINA_API_TOKEN` (at least 32 random characters)
and `GOLDEN_RETINA_CHECKPOINT` in the root-readable
`/etc/golden-retina/api.env`, then install
`deploy/golden-retina-research-api.service` as a systemd unit. The unit runs as
the project account, binds only to `127.0.0.1:8769`, disables request access
logs, isolates temporary upload files, and verifies the locked checkpoint hash
at startup. Never put the token in Git, the tracker, request logs, or terminal
output. This deployment was used for private IAP-tunnel smoke testing; it is
not an Internet-facing or clinical service.

For a private client session, create the SSH port forward with:

```sh
gcloud compute ssh golden-retina-l4-01 \
  --project=golden-retina-dev-20261107 --zone=us-west1-a \
  --tunnel-through-iap -- \
  -N -L 127.0.0.1:18769:127.0.0.1:8769
```

Then send one or two views to `http://127.0.0.1:18769/v1/infer` as multipart
fields `first_view` and optional `second_view`, with `request_id`,
`intended_use=research`, and `Authorization: Bearer $GOLDEN_RETINA_API_TOKEN`.
The service echoes `request_id`; clients should accept a response only when its
ID matches the currently active request. The secret is held in a root-readable
VM environment file and must never be committed or printed.

## Evidence boundary

The model's internal validation operating point had 45 missed referable cases
among 436; it is neither zero-miss nor autonomous-certified. The Zhejiang
external-source pass has diagnosis-vs-healthy labels, not referable-grade
labels, so its ranking results cannot validate these thresholds. Keep the
research-only status in any user interface or integration.
