import os
import time
from typing import List, Dict, Any, Optional
import numpy as np
from ultralytics import YOLO

class ObjectDetector:
    """
    YOLO Object Detector
    Loads a lightweight pretrained YOLO model once at startup and performs
    high-speed real-time object detection on input frames.
    """
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35):
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        
        # Path to local models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, model_name)
        
        # If model exists locally in models dir, load it; otherwise load/download
        if os.path.exists(model_path):
            print(f"[Detector] Loading YOLO model from {model_path}...")
            self.model = YOLO(model_path)
        else:
            print(f"[Detector] Initializing YOLO model ({model_name})...")
            self.model = YOLO(model_name)
            # Save or copy to models dir if possible
            try:
                if os.path.exists(model_name) and not os.path.exists(model_path):
                    import shutil
                    shutil.copy(model_name, model_path)
            except Exception:
                pass
        
        # Warm up the model with a blank frame for instant first-frame inference
        self._warmup()

    def _warmup(self):
        """Warm up model on dummy tensor to avoid initial inference latency."""
        try:
            dummy = np.zeros((320, 320, 3), dtype=np.uint8)
            self.model.predict(dummy, verbose=False, conf=self.conf_threshold)
            print("[Detector] Model warmup complete.")
        except Exception as e:
            print(f"[Detector] Warmup skipped: {e}")

    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Run object detection on an OpenCV BGR frame.
        Returns a list of detected objects with class, confidence, and bbox coordinates.
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        results = self.model.predict(image, conf=conf, verbose=False)
        
        detections = []
        if not results or len(results) == 0:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return detections
            
        names = self.model.names
        
        for box in boxes:
            # Extract coordinates [x1, y1, x2, y2]
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [round(coord, 2) for coord in xyxy]
            
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = names.get(class_id, f"class_{class_id}")
            
            detections.append({
                "class": class_name,
                "class_id": class_id,
                "confidence": round(confidence, 2),
                "bbox": [x1, y1, x2, y2]
            })
            
        return detections
