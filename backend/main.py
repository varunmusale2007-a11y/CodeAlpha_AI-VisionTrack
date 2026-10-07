import os
import base64
import time
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from detector import ObjectDetector
from tracker import ByteTrackTracker

# Initialize FastAPI App
app = FastAPI(
    title="AI-VisionTrack API",
    description="Real-time object detection and ByteTrack multi-object tracking backend",
    version="1.0.0"
)

# Enable CORS for frontend communication (local development and production e.g. Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize EXACTLY ONE YOLO model instance and ONE pure ByteTrack tracking engine at startup
print("[System] Initializing AI-VisionTrack backend pipeline...")
detector = ObjectDetector(model_name="yolov8n.pt", conf_threshold=0.35)
tracker = ByteTrackTracker(high_thresh=0.45, match_thresh=0.35, max_time_lost=30)
start_time = time.time()
print("[System] Singleton YOLO detector and ByteTrack tracker ready.")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64 encoded image string to OpenCV BGR numpy array."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            raise ValueError("Failed to decode image buffer")
            
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")


@app.get("/")
def root():
    """Root endpoint providing API information."""
    return {
        "service": "AI-VisionTrack Backend",
        "status": "online",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "detect": "/api/detect (POST)",
            "reset_tracker": "/api/reset-tracker (POST)"
        }
    }


@app.get("/api/health")
def health_check():
    """Health check endpoint to verify backend status and single model."""
    uptime_seconds = round(time.time() - start_time, 2)
    return {
        "status": "online",
        "service": "AI-VisionTrack Backend",
        "version": "1.0.0",
        "model": "YOLOv8n (Single Instance)",
        "tracker": "ByteTrack (Memory Optimized)",
        "framework": "FastAPI + Ultralytics + OpenCV",
        "uptime_seconds": uptime_seconds
    }


@app.post("/api/detect")
async def detect_objects(request: Request):
    """
    Main Object Detection & Tracking Endpoint
    Architecture:
    Image Frame -> ONE YOLO Model -> Detections -> ByteTrack -> Tracking IDs -> JSON Response
    """
    t_start = time.perf_counter()
    image = None
    conf_thresh = 0.35
    tracking_enabled = True

    content_type = request.headers.get("content-type", "")

    # 1. Parse Image Payload
    if "application/json" in content_type:
        try:
            body = await request.json()
            if not body or "image" not in body:
                raise HTTPException(status_code=400, detail="Missing 'image' field in JSON request.")
            image = decode_base64_image(body["image"])
            conf_thresh = float(body.get("confidence", 0.35))
            tracking_enabled = bool(body.get("use_tracker", True))
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=400, detail=f"JSON decoding error: {str(e)}")
    elif "multipart/form-data" in content_type:
        try:
            form = await request.form()
            file = form.get("file")
            if not file:
                raise HTTPException(status_code=400, detail="Missing 'file' field in form data.")
            file_bytes = await file.read()
            nparr = np.frombuffer(file_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            conf_thresh = float(form.get("confidence", 0.35))
            tracking_enabled = form.get("use_tracker", "true").lower() in ("true", "1")
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=400, detail=f"Form processing error: {str(e)}")
    else:
        try:
            body = await request.json()
            if "image" in body:
                image = decode_base64_image(body["image"])
                conf_thresh = float(body.get("confidence", 0.35))
                tracking_enabled = bool(body.get("use_tracker", True))
            else:
                raise HTTPException(status_code=400, detail="Missing 'image' in payload.")
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported Content-Type. Please use application/json or multipart/form-data.")

    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Invalid or empty image.")

    h, w = image.shape[:2]

    # 2. Step 1: Run detection through the single YOLOv8 model instance
    raw_detections = detector.detect(image, conf_threshold=conf_thresh)

    # 3. Step 2: Feed detections into pure ByteTrack to associate persistent tracking IDs
    if tracking_enabled:
        detections = tracker.update(raw_detections)
    else:
        detections = []
        for idx, det in enumerate(raw_detections):
            detections.append({
                "class": det["class"],
                "confidence": det["confidence"],
                "track_id": idx + 1,
                "bbox": det["bbox"]
            })

    t_end = time.perf_counter()
    inference_time_ms = round((t_end - t_start) * 1000, 2)
    fps = round(1000.0 / inference_time_ms, 1) if inference_time_ms > 0 else 0.0

    # 4. Step 3: Return exact requested JSON schema
    return {
        "status": "success",
        "detections": detections,
        "metadata": {
            "total_objects": len(detections),
            "inference_time_ms": inference_time_ms,
            "fps": fps,
            "frame_width": w,
            "frame_height": h,
            "tracker_used": "ByteTrack" if tracking_enabled else "None"
        }
    }


@app.post("/api/reset-tracker")
def reset_tracker_endpoint():
    """Resets the ByteTrack tracker state."""
    tracker.reset()
    return {
        "status": "success",
        "message": "ByteTrack tracker state has been reset successfully."
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
