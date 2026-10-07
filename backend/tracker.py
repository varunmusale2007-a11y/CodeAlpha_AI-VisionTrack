import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

class ByteTrackTracker:
    """
    ByteTrack Object Tracker
    Maintains persistent tracking states across sequential video/webcam frames,
    associating detected objects with unique, persistent track IDs using ByteTrack.
    """
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35, tracker_type: str = "bytetrack.yaml"):
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        self.tracker_type = tracker_type
        
        # Resolve model path
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(model_path):
            print(f"[Tracker] Loading YOLO model for ByteTrack from {model_path}...")
            self.model = YOLO(model_path)
        else:
            print(f"[Tracker] Initializing YOLO model ({model_name})...")
            self.model = YOLO(model_name)
            
        # Fallback counter for untracked single-frame detections
        self._untracked_counter = 0
        self._frame_count = 0
        
        # Warmup tracker
        self._warmup()

    def _warmup(self):
        """Warm up the tracker with a zero tensor so the first real frame runs without lag."""
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.track(dummy, persist=True, tracker=self.tracker_type, conf=self.conf_threshold, verbose=False)
            print("[Tracker] ByteTrack tracker warmed up and ready.")
        except Exception as e:
            print(f"[Tracker] Warmup skipped: {e}")

    def track(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Processes a single frame sequentially through YOLO and ByteTrack.
        Preserves tracker state across frames (persist=True).
        
        Returns:
            List of dictionaries containing class, confidence, track_id, and bbox [x1, y1, x2, y2].
        """
        self._frame_count += 1
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        # Run YOLO with ByteTrack tracker and persist state
        results = self.model.track(
            source=image,
            persist=True,
            tracker=self.tracker_type,
            conf=conf,
            verbose=False
        )
        
        tracked_detections = []
        if not results or len(results) == 0:
            return tracked_detections
            
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return tracked_detections
            
        names = self.model.names
        
        for box in boxes:
            # Extract bounding box [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [round(float(c), 2) for c in xyxy]
            
            confidence = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])
            class_name = names.get(class_id, f"class_{class_id}")
            
            # Extract ByteTrack tracking ID
            if box.id is not None:
                track_id = int(box.id[0])
            else:
                # If object is detected in first frame before ByteTrack assigns ID
                self._untracked_counter += 1
                track_id = self._untracked_counter
                
            tracked_detections.append({
                "class": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2]
            })
            
        return tracked_detections

    def reset(self):
        """Resets tracking state (e.g., when a new camera session begins)."""
        try:
            # Re-initialize tracker predictor state if possible
            if hasattr(self.model, "predictor") and self.model.predictor is not None:
                if hasattr(self.model.predictor, "trackers"):
                    self.model.predictor.trackers = []
            self._untracked_counter = 0
            self._frame_count = 0
            print("[Tracker] Tracker state reset.")
        except Exception as e:
            print(f"[Tracker] Error during tracker reset: {e}")
