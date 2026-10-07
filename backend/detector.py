import os
from typing import List, Dict, Any, Optional
import numpy as np
import torch
from ultralytics import YOLO

# Enforce low CPU memory limits for PyTorch in constrained environments (e.g. Render 512MB)
torch.set_grad_enabled(False)
try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass


class ObjectDetector:
    """
    Lightweight YOLO Object Detector (Singleton instance).
    Loads exactly ONE YOLO model instance in memory and reuses it
    for all detection requests to prevent Out-Of-Memory (OOM) errors.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ObjectDetector, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35):
        if self._initialized:
            return
            
        self.conf_threshold = conf_threshold
        self.model_name = model_name
        
        # Resolve model path in backend/models directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, model_name)
        
        if os.path.exists(model_path):
            print(f"[Detector] Loading single YOLO model from {model_path}...")
            self.model = YOLO(model_path)
        else:
            print(f"[Detector] Initializing single YOLO model ({model_name})...")
            self.model = YOLO(model_name)
            
        # Optimize model in memory
        try:
            if hasattr(self.model, "model") and self.model.model is not None:
                self.model.model.eval()
        except Exception:
            pass

        self._initialized = True
        print("[Detector] Single YOLO model ready for inference.")

    def detect(self, image: np.ndarray, conf_threshold: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Run object detection on an OpenCV BGR frame.
        Returns a list of raw detections: class name, class ID, confidence, and bbox coordinates.
        """
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        
        with torch.no_grad():
            results = self.model.predict(
                image,
                conf=conf,
                device="cpu",
                verbose=False,
                imgsz=640
            )
        
        detections: List[Dict[str, Any]] = []
        if not results or len(results) == 0:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return detections
            
        names = self.model.names
        
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            x1, y1, x2, y2 = [round(float(coord), 2) for coord in xyxy]
            
            confidence = round(float(box.conf[0]), 2)
            class_id = int(box.cls[0])
            class_name = names.get(class_id, f"class_{class_id}")
            
            detections.append({
                "class": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })
            
        return detections
