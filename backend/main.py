import base64
import time
from typing import Optional, List, Dict, Any
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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

# Initialize Detector and Tracker once at startup
print("[System] Loading YOLO and ByteTrack models...")
detector = ObjectDetector(model_name="yolov8n.pt", conf_threshold=0.35)
tracker = ByteTrackTracker(model_name="yolov8n.pt", conf_threshold=0.35)
start_time = time.time()
print("[System] Models initialized successfully. Backend ready.")


class DetectRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image string")
    confidence: Optional[float] = Field(0.35, ge=0.05, le=1.0, description="Confidence threshold for YOLO")
    use_tracker: Optional[bool] = Field(True, description="Whether to use ByteTrack tracking or standard detection")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode a base64 encoded image string to OpenCV BGR numpy array."""
    try:
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]
        
        image_bytes = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            raise ValueError("Failed to decode image from buffer")
            
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
    """Health check endpoint to verify backend status and models."""
    uptime_seconds = round(time.time() - start_time, 2)
    return {
        "status": "online",
        "service": "AI-VisionTrack Backend",
        "version": "1.0.0",
        "model": "YOLOv8n",
        "tracker": "ByteTrack",
        "framework": "FastAPI + Ultralytics + OpenCV",
        "uptime_seconds": uptime_seconds
    }


@app.post("/api/detect")
async def detect_objects(request: Request):
    """
    Main Object Detection & Tracking Endpoint
    Receives an image via JSON Base64 or Multipart Form Data,
    runs YOLO detection + ByteTrack tracking, and returns bounding boxes,
    class names, confidence scores, and persistent tracking IDs.
    """
    t_start = time.perf_counter()
    image = None
    conf_thresh = 0.35
    tracking_enabled = True

    content_type = request.headers.get("content-type", "")

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
        # Try fallback JSON parsing
        try:
            body = await request.json()
            if "image" in body:
                image = decode_base64_image(body["image"])
                conf_thresh = float(body.get("confidence", 0.35))
                tracking_enabled = bool(body.get("use_tracker", True))
            else:
                raise HTTPException(status_code=400, detail="Unsupported Content-Type or missing 'image' field.")
        except Exception:
            raise HTTPException(status_code=400, detail="Unsupported Content-Type. Please use application/json or multipart/form-data.")

    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Invalid or empty image.")

    h, w = image.shape[:2]

    # Perform Object Tracking with ByteTrack or Standard Detection
    if tracking_enabled:
        detections = tracker.track(image, conf_threshold=conf_thresh)
    else:
        detections = detector.detect(image, conf_threshold=conf_thresh)
        for idx, det in enumerate(detections):
            det["track_id"] = idx + 1

    t_end = time.perf_counter()
    inference_time_ms = round((t_end - t_start) * 1000, 2)
    fps = round(1000.0 / inference_time_ms, 1) if inference_time_ms > 0 else 0.0

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
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
