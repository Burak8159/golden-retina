# Disease-specialist research API

This is the versioned inference interface for separately trained Golden Retina
disease specialists. It does not combine unlike disease labels into a single
"any eye disease" output. Each enabled model has its own source label,
architecture, preprocessing rule, checkpoint hash, and version.

## Research-only contract

- `POST /v1/screen` accepts one fundus image, a request ID, and
  `intended_use=research`.
- The response returns a raw sigmoid model score for each configured source
  label. `calibrated_probability` and `clinical_diagnosis` are always false.
- No model output is a patient-specific risk percentage. Calibration,
  representative external validation, and clinical utility have not been
  established by this development campaign.
- `/v1/model-card` returns the enabled checkpoint hashes and preprocessing.
- `/healthz` returns service status without exposing model paths or scores.
- Images are decoded in memory and are not written to application-managed
  storage. The service does not retain request IDs or images.
- `/v1/screen` and `/v1/model-card` require a bearer token of at least 32
  characters. Bind to loopback or a private authenticated network only.
- The model registry and checkpoints remain on the RTX. The tracked example
  registry contains no live checkpoint path and enables no model.

## RTX setup after model selection

Copy `configs/rtx/disease_specialists.local.example.json` to a private local
registry on the RTX, then fill it only from verified aggregate run reports and
checkpoint hashes. Never commit the local registry, access token, images,
patient rows, or checkpoints. Set:

```text
GOLDEN_RETINA_SPECIALIST_REGISTRY=<RTX project root>\configs\rtx\disease_specialists.local.json
GOLDEN_RETINA_API_TOKEN=<secret stored in the RTX user's protected environment>
PYTHONPATH=<RTX project root>\src
```

Launch with the Golden Retina virtual environment and bind the service to
`127.0.0.1` or a private Tailscale address. Do not expose it to the public
Internet. The checkpoint SHA-256 is checked at service startup; any mismatch
prevents the model from loading.

## Client rule

Clients must retain the request ID and discard a response whose ID does not
match the active scan. Display scores as experimental model outputs, together
with the research-only status and clinician-review requirement. Never label
them as calibrated disease risk or use them to triage a patient.
