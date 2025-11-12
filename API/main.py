import asyncio
import base64
import os
import threading
import time
import json
import numpy as np

from io import BytesIO
from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.websockets import WebSocketState

try:
    from ultralytics import YOLO
except Exception as exc:
    raise RuntimeError(
        "Missing dependency 'ultralytics'. Install: pip install ultralytics"
    ) from exc


# region Config

MODEL_PATH = os.getenv("MODEL_PATH", "best.pt")
CAMERA_SRC = os.getenv("CAMERA_SRC", "0")
CONF_THRES = float(os.getenv("CONF_THRES", "0.5"))
IOU_THRES = float(os.getenv("IOU_THRES", "0.45"))
TARGET_FPS = float(os.getenv("TARGET_FPS", "15"))
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")

# endregion

# region App Init

app = FastAPI(title="YOLO Real-Time Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in CORS_ORIGINS.split(",")] if CORS_ORIGINS else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# endregion

# region Data Models 

class Box(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    cls: int
    label: str
    conf: float
    box: Box

class PredictResponse(BaseModel):
    width: int
    height: int
    detections: List[Detection] = Field(default_factory=list)
    latency_ms: float

# endregion

# region Camera Worker

@dataclass
class FrameBundle:
    frame: Optional[Any]
    ts: float

class CameraWorker:
    """จับกล้องใน thread เพื่อให้ API non-blocking; เก็บแค่เฟรมล่าสุด"""

    def __init__(self, src: str | int) -> None:
        self.src = int(src) if src.isdigit() else src
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._latest: FrameBundle = FrameBundle(frame=None, ts=0.0)
        self._alive = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        try:
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap or not self.cap.isOpened():
                print(f"Warning: Cannot open camera source: {self.src}. Camera features will be disabled.")
                self.cap = None
                return
            self._alive.set()
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        except Exception as e:
            print(f"Warning: Failed to start camera: {e}. Camera features will be disabled.")
            self.cap = None

    def _loop(self) -> None:
        while self._alive.is_set():
            ok, frame = self.cap.read()
            ts = time.time()
            if ok:
                with self._lock:
                    self._latest = FrameBundle(frame=frame, ts=ts)
            else:
                time.sleep(0.01)

    def get_latest(self) -> FrameBundle:
        with self._lock:
            return FrameBundle(frame=None if self._latest.frame is None else self._latest.frame.copy(),
                               ts=self._latest.ts)

    def stop(self) -> None:
        self._alive.clear()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.cap = None

# endregion

# YOLO Runtime
class YoloRuntime:
    def __init__(self, model_path: str) -> None:
        self.model = YOLO(model_path)
        self.names: Dict[int, str] = self.model.model.names if hasattr(self.model, "model") else self.model.names

    def predict_image(self, img_bgr: Any, conf: float, iou: float):
        return self.model.predict(img_bgr, conf=conf, iou=iou, verbose=False)[0]

# App State
class AppState:
    def __init__(self) -> None:
        self.cam: Optional[CameraWorker] = None
        self.yolo: Optional[YoloRuntime] = None

state = AppState()

# Utils 
def result_to_detections(res, names: Dict[int, str]) -> Tuple[List[Detection], int, int]:
    boxes = res.boxes
    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, "xyxy") else []
    confs = boxes.conf.cpu().numpy() if hasattr(boxes, "conf") else []
    clss = boxes.cls.cpu().numpy().astype(int) if hasattr(boxes, "cls") else []
    dets: List[Detection] = []
    for (x1, y1, x2, y2), conf, ci in zip(xyxy, confs, clss):
        dets.append(
            Detection(
                cls=int(ci),
                label=names.get(int(ci), str(ci)),
                conf=float(conf),
                box=Box(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2)),
            )
        )
    h, w = res.orig_img.shape[:2]
    return dets, w, h

def draw_boxes(img_bgr: Any, dets: List[Detection]) -> Any:
    for d in dets:
        x1, y1, x2, y2 = map(int, (d.box.x1, d.box.y1, d.box.x2, d.box.y2))
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.label} {d.conf:.2f}"
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img_bgr, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return img_bgr

def encode_jpeg(img_bgr: Any, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    return buf.tobytes()

# Lifespan 
@app.on_event("startup")
def on_startup() -> None:
    try:
        print(f"📦 Loading YOLO model from: {MODEL_PATH}")
        state.yolo = YoloRuntime(MODEL_PATH)
        print(f"✅ Model loaded successfully. Classes: {len(state.yolo.names)}")
    except Exception as e:
        print(f"❌ CRITICAL: Failed to load model: {e}")
        state.yolo = None
    
    state.cam = CameraWorker(CAMERA_SRC)
    state.cam.start()

@app.on_event("shutdown")
def on_shutdown() -> None:
    if state.cam:
        state.cam.stop()

# Dependencies
def ensure_ready() -> None:
    if not state.yolo:
        print("❌ ERROR: YOLO model not loaded!")
        raise HTTPException(status_code=503, detail="Model not ready - check MODEL_PATH")

def ensure_camera_ready() -> None:
    ensure_ready()
    if not state.cam or not state.cam.cap:
        raise HTTPException(status_code=503, detail="Camera not available")

# Endpoints
@app.get("/health")
def health() -> Dict[str, Any]:
    cam_ok = state.cam is not None and state.cam.get_latest().frame is not None
    return {"status": "ok", "camera_ready": cam_ok, "model": MODEL_PATH}

@app.get("/labels")
def labels() -> Dict[int, str]:
    ensure_ready()
    return state.yolo.names  # type: ignore[return-value]

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    ensure_ready()
    data = await file.read()
    nparr = cv2.imdecode(
        buf := cv2.UMat(bytearray(data)) if hasattr(cv2, "UMat") else None, 1
    ) if False else cv2.imdecode(
        np_from_buffer(data), 1
    )

def np_from_buffer(b: bytes):
    import numpy as np
    return np.frombuffer(b, dtype="uint8")

@app.post("/predict", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)) -> PredictResponse:
    ensure_ready()
    import numpy as np
    data = await file.read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    t0 = time.time()
    res = state.yolo.predict_image(img, conf=CONF_THRES, iou=IOU_THRES)  # type: ignore[arg-type]
    dets, w, h = result_to_detections(res, state.yolo.names)  # type: ignore[arg-type]
    latency = (time.time() - t0) * 1000.0
    return PredictResponse(width=w, height=h, detections=dets, latency_ms=latency)

@app.get("/stream.mjpg")
def stream_mjpeg():
    ensure_camera_ready()

    async def gen():
        # why: รักษา FPS เป้าหมาย & ลดโหลด
        frame_interval = 1.0 / max(TARGET_FPS, 1.0)
        while True:
            start = time.time()
            fb = state.cam.get_latest()  # type: ignore[union-attr]
            if fb.frame is None:
                await asyncio.sleep(0.01)
                continue
            res = state.yolo.predict_image(fb.frame, conf=CONF_THRES, iou=IOU_THRES)  # type: ignore[arg-type]
            dets, _, _ = result_to_detections(res, state.yolo.names)  # type: ignore[arg-type]
            annotated = draw_boxes(fb.frame, dets)
            jpg = encode_jpeg(annotated, quality=80)
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            yield boundary + jpg + b"\r\n"
            elapsed = time.time() - start
            sleep_s = max(0.0, frame_interval - elapsed)
            await asyncio.sleep(sleep_s)

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.websocket("/ws")
async def ws_detect(websocket: WebSocket):
    ensure_camera_ready()
    await websocket.accept()
    frame_interval = 1.0 / max(TARGET_FPS, 1.0)
    try:
        while True:
            start = time.time()
            fb = state.cam.get_latest()  # type: ignore[union-attr]
            if fb.frame is None:
                await asyncio.sleep(0.01)
                continue
            res = state.yolo.predict_image(fb.frame, conf=CONF_THRES, iou=IOU_THRES)  # type: ignore[arg-type]
            dets, w, h = result_to_detections(res, state.yolo.names)  # type: ignore[arg-type]
            payload = {
                "ts": fb.ts,
                "width": w,
                "height": h,
                "detections": [d.model_dump() for d in dets],
            }
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json(payload)
            elapsed = time.time() - start
            sleep_s = max(0.0, frame_interval - elapsed)
            await asyncio.sleep(sleep_s)
    except WebSocketDisconnect:
        return
    except Exception as exc:
        # why: ให้ client เห็น error ทันที
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"error": str(exc)})
        await websocket.close()
        
@app.websocket("/ws-client")
async def ws_client_detect(websocket: WebSocket):
    """WebSocket endpoint สำหรับรับ frame จาก client browser"""
    ensure_ready()
    await websocket.accept()
    print("🔌 Client connected to /ws-client")
    
    try:
        while True:
            data = await websocket.receive_text()
            try:
                payload = json.loads(data)
                if "image" not in payload:
                    await websocket.send_json({"error": "Missing image field"})
                    continue
                    
                # Decode base64 image
                try:
                    image_data = base64.b64decode(payload["image"])
                    image = Image.open(BytesIO(image_data))
                    img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                except Exception as e:
                    print(f"❌ Image decode error: {e}")
                    await websocket.send_json({"error": f"Invalid image: {str(e)}"})
                    continue
                
                # Run YOLO detection
                t0 = time.time()
                res = state.yolo.predict_image(img_bgr, conf=CONF_THRES, iou=IOU_THRES)
                dets, w, h = result_to_detections(res, state.yolo.names)
                
                # Send results back
                response_payload = {
                    "ts": time.time(),
                    "width": w,
                    "height": h,
                    "detections": [d.model_dump() for d in dets],
                    "latency_ms": (time.time() - t0) * 1000.0
                }
                
                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_json(response_payload)
                    
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                await websocket.send_json({"error": "Invalid JSON format"})
            except Exception as e:
                print(f"❌ Processing error: {e}")
                if websocket.application_state == WebSocketState.CONNECTED:
                    await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print("🔌 Client disconnected from /ws-client")
    except Exception as exc:
        print(f"❌ WebSocket exception: {exc}")
        if websocket.application_state == WebSocketState.CONNECTED:
            await websocket.send_json({"error": str(exc)})
        await websocket.close()
        
# =================== Serve Static Files ===================
STATIC_DIR = "out"
if os.path.exists(STATIC_DIR) and os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
else:
    print(f"Warning: The '{STATIC_DIR}' directory was not found. Please run 'npm run build' in your Next.js project to generate the static files.")

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8080)