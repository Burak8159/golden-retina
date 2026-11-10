"""Minimal frame-in / eye-observation-out HTTP service; keeps frames in memory only."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import io
import json
import re
import threading

import numpy as np
from PIL import Image, UnidentifiedImageError
import torch

from model import FaceEyeUNet
from preprocess import letterbox_rgb, remap_xy
from sequence_filter import AlphaBeta


class EyeService:
    def __init__(self, checkpoint: str, confidence: float, max_sessions: int = 256):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.size = int(self.ck.get("size", 384)); self.threshold = confidence; self.max_sessions = max_sessions
        self.model = FaceEyeUNet(base=self.ck.get("base", 16)).to(self.device)
        self.model.load_state_dict(self.ck["model"]); self.model.eval()
        self.checkpoint_hash = hashlib.sha256(open(checkpoint, "rb").read()).hexdigest()
        self.sessions = OrderedDict(); self.lock = threading.Lock()

    def predict(self, raw: bytes, session: str):
        try:
            with Image.open(io.BytesIO(raw)) as src:
                width,height=src.size
                if width < 64 or height < 64 or width > 8192 or height > 8192: raise ValueError("frame dimensions must be between 64 and 8192 pixels")
                image=src.convert("RGB").copy()
        except ValueError: raise
        except (UnidentifiedImageError, OSError) as exc: raise ValueError("body must contain a decodable PNG or JPEG frame") from exc
        x, transform = letterbox_rgb(image,self.size)
        with torch.inference_mode(): probs=self.model(torch.from_numpy(x).permute(2,0,1)[None].to(self.device)).softmax(1)[0].cpu().numpy()
        pred=probs.argmax(0);eyes={}
        with self.lock:
            filters=self.sessions.get(session)
            if filters is None:
                filters={s:AlphaBeta() for s in ("left","right")};self.sessions[session]=filters
            self.sessions.move_to_end(session)
            while len(self.sessions)>self.max_sessions:self.sessions.popitem(last=False)
            for side,cls in (("left",4),("right",8)):
                yy,xx=np.where(pred==cls);conf=float(probs[cls,yy,xx].mean()) if len(xx)>=5 else 0.
                detection=remap_xy(float(xx.mean()),float(yy.mean()),transform) if len(xx)>=5 and conf>=self.threshold else None
                tr=filters[side].update(detection)
                eyes[side]={"status":"visible_candidate" if detection else "uncertain_or_occluded",
                            "detection_xy_px":[round(v,2) for v in detection] if detection else None,
                            "track_xy_px":[round(v,2) for v in tr["xy"]] if tr["xy"] else None,
                            "track_state":tr["state"],"confidence":round(conf,4) if detection else None,
                            "pupil_pixels":int(len(xx))}
        return {"status":"research_eye_observation","synthetic_only_validation":True,"physical_camera_validation":False,
                "checkpoint_sha256":self.checkpoint_hash,"device":str(self.device),"frame_size_px":[width,height],"eyes":eyes}


def main():
    p=argparse.ArgumentParser();p.add_argument("--checkpoint",required=True);p.add_argument("--host",default="127.0.0.1");p.add_argument("--port",type=int,default=8780)
    p.add_argument("--minimum-confidence",type=float,default=.35);p.add_argument("--max-frame-bytes",type=int,default=10*1024*1024);a=p.parse_args()
    service=EyeService(a.checkpoint,a.minimum_confidence);body_limit=a.max_frame_bytes
    class Handler(BaseHTTPRequestHandler):
        server_version="GoldenRetinaEyeService/0.1"
        def _send(self,code,payload):
            data=json.dumps(payload).encode();self.send_response(code);self.send_header("Content-Type","application/json");self.send_header("Content-Length",str(len(data)));self.send_header("Cache-Control","no-store");self.end_headers();self.wfile.write(data)
        def do_GET(self):
            if self.path!="/health":return self._send(404,{"error":"not_found"})
            self._send(200,{"status":"ready","device":str(service.device),"checkpoint_sha256":service.checkpoint_hash,
                            "sessions_cached":len(service.sessions),"synthetic_only_validation":True})
        def do_POST(self):
            if self.path!="/v1/frame":return self._send(404,{"error":"not_found"})
            try:
                n=int(self.headers.get("Content-Length","0"))
                if n<=0 or n>body_limit:return self._send(413,{"error":"invalid_frame_size","max_bytes":body_limit})
                raw=self.rfile.read(n)
                session=self.headers.get("X-Session-ID","default")
                if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}",session):return self._send(400,{"error":"invalid_session_id"})
                self._send(200,service.predict(raw,session))
            except ValueError as e:self._send(400,{"error":"invalid_frame","detail":str(e)})
            except Exception as e:self._send(500,{"error":"inference_failed","detail":type(e).__name__})
        def log_message(self,fmt,*args):print("eye-api",self.address_string(),fmt%args,flush=True)
    print(json.dumps({"status":"starting","host":a.host,"port":a.port,"device":str(service.device),"checkpoint_sha256":service.checkpoint_hash}),flush=True)
    ThreadingHTTPServer((a.host,a.port),Handler).serve_forever()


if __name__=="__main__":main()
