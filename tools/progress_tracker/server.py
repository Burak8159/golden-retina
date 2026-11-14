#!/usr/bin/env python3
"""Local-only read-only progress dashboard for Golden Retina development."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os

ROOT = Path(__file__).resolve().parents[2]
WEB = Path(__file__).resolve().parent
STATUS = ROOT / "results" / "disease_expansion_20261114" / "status.json"
CANDIDATES = ROOT / "configs" / "rfmid_expert_architecture_sweep_20261114.json"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            try:
                status = json.loads(STATUS.read_text(encoding="utf-8"))
                config = json.loads(CANDIDATES.read_text(encoding="utf-8"))
                specialist = status.get("per_disease_specialist_status", {})
                roster = []
                for target, architectures in config.get("candidate_architectures", {}).items():
                    result = specialist.get(target, {})
                    assignment = next(
                        (x for x in status.get("disease_model_assignments", []) if x.get("target") == target),
                        {},
                    )
                    baseline = assignment.get("baseline_metrics", {})
                    metric_text = ""
                    if baseline:
                        metric_text = (
                            f"Baseline accuracy {baseline.get('accuracy_at_0_5', float('nan')):.1%}; "
                            f"AUROC {baseline.get('auroc', float('nan')):.3f}"
                        )
                    roster.append({
                        "name": assignment.get("display", target),
                        "family": " / ".join(architectures),
                        "availability_status": result.get("status_label", "Queued"),
                        "terms_status": assignment.get("tier", "support tier unrecorded"),
                        "decision": " · ".join(x for x in [result.get("evidence", ""), metric_text] if x),
                    })
                for target, label, family in [
                    ("HYGD_GON", "Glaucoma suspect / GON · historical model", "RETFound-MAE ViT-L/16 · existing internal validation"),
                ]:
                    result = specialist.get(target, {})
                    if result:
                        roster.append({"name": label, "family": family,
                                       "availability_status": result.get("status_label", "Historical evidence"),
                                       "terms_status": "Not part of the active RFMiD sweep",
                                       "decision": result.get("evidence", "")})
                mh = next((b for b in status.get("blocks", []) if b.get("id") == 23), {})
                if mh:
                    roster.append({"name": "Media haze / fundus visibility · quality signal",
                                   "family": "ConvNeXt-Tiny multi-label baseline · not a disease head",
                                   "availability_status": mh.get("status_label", "Quality signal"),
                                   "terms_status": "Separate non-disease quality evidence",
                                   "decision": mh.get("evidence", "")})
                status["candidate_roster"] = roster
                body = json.dumps(status, ensure_ascii=False).encode("utf-8")
            except OSError:
                self.send_error(503, "Progress data is not available yet")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            body = (WEB / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)

    def log_message(self, _format, *_args):
        pass


if __name__ == "__main__":
    address = ("127.0.0.1", int(os.environ.get("GOLDEN_RETINA_TRACKER_PORT", "8768")))
    print(f"Golden Retina tracker: http://{address[0]}:{address[1]}/", flush=True)
    ThreadingHTTPServer(address, Handler).serve_forever()
