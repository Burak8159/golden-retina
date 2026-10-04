Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

Set-Location "C:\Users\bataa\golden_retina"
$env:PYTHONPATH = "src"

.\.venv-golden-retina\Scripts\python.exe -m golden_retina.build_autonomy_route_manifest `
  --output results/autonomy_route_combined_manifest_rtx.csv `
  --summary-output results/autonomy_route_combined_manifest_rtx_summary.json

Get-Content results\autonomy_route_combined_manifest_rtx_summary.json
