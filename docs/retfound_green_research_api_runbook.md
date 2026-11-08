# RETFound-Green research inference endpoint

This is a private, loopback-only research service for the frozen candidate in `results/technical_dev_20261108_rerun/retfound_green_research_model_freeze.json`.

- It binds only to `127.0.0.1` on the research VM; use an authenticated SSH tunnel for a local client.
- `/v1/infer` accepts one image and a request ID, and returns five-grade DR scores plus referable-DR and HYGD glaucoma scores.
- `/v1/model-card` describes checkpoint provenance and restrictions. `/healthz` reports process/model readiness without inference details.
- Both inference and model-card routes require a bearer token of at least 32 characters. Keep the token in a mode-0600 file outside the repository.
- Upload size is capped at 25 MiB and image dimensions at 50 MP. Uploaded images are decoded in memory and not written by the handler.
- The checkpoint and RETFound-Green base weights are hash-pinned; startup fails closed if either file is missing or differs.
- Responses are research scores, not diagnoses or calibrated clinical probabilities. The service is not validated for patient-care use.
- RETFound-Green use is limited to the applicable non-commercial research terms. Do not redistribute its weights or expose the service as a clinical/commercial product.

Start with `PYTHONPATH=src uvicorn golden_retina.retfound_green_research_api:app --host 127.0.0.1 --port 8768`. Configure `GOLDEN_RETINA_API_TOKEN`, `GOLDEN_RETINA_CHECKPOINT`, and `GOLDEN_RETINA_BASE_CHECKPOINT` from a protected environment file. A service running on this VM is for project-internal research checks only; it is not a public deployment.
