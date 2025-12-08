import os,sys
sys.path.append(os.path.dirname(__file__))

from ultralytics import YOLO
from base import BaseFaceDetector
from src.config import YOLO_MODELS


class YOLODetector(BaseFaceDetector):
    """
    Generic YOLO face detector supporting different YOLO model presets and custom models.
    
    Uses Ultralytics YOLO implementation for face detection. Supports both
    predefined model presets and custom model paths.
    
    Attributes:
        model: YOLO model instance
        confidence: Minimum confidence threshold for valid detections
    """

    def __init__(self, preset=None, model_path=None, confidence=0.5):
        """
        Initializes YOLO detector with either a preset or custom model.
        
        Args:
            preset (str, optional): Name of predefined YOLO model from YOLO_MODELS config.
            model_path (str, optional): Path to custom YOLO model weights (.pt file).
            confidence (float): Minimum confidence threshold (0-1). Default: 0.5.
            
        Raises:
            ValueError: If neither preset nor model_path is provided,
                       or if preset is not found in YOLO_MODELS.
        """
        if preset:
            if preset not in YOLO_MODELS:
                raise ValueError(f"Unknown YOLO preset '{preset}'. Available: {list(YOLO_MODELS.keys())}")
            self.model = YOLO(YOLO_MODELS[preset])
        elif model_path:
            self.model = YOLO(model_path)
        else:
            raise ValueError("You must provide either 'preset' or 'model_path'.")

        self.confidence = confidence

    def detect(self, frame):
        """
        Detects a face in the given frame using YOLO.
        
        Args:
            frame (numpy.ndarray): Input image in BGR or RGB format
            
        Returns:
            tuple or None: Bounding box as (x, y, width, height) in pixels,
                          or None if no valid detection meets confidence threshold.
                          Note: YOLO returns XYXY format, converted to XYWH here.
        """
        results = self.model(frame, verbose=False)
        if len(results) == 0:
            return None

        boxes = results[0].boxes
        if len(boxes) == 0:
            return None

        # Take highest confidence detection
        best_box = max(boxes, key=lambda b: b.conf[0])

        if best_box.conf[0] < self.confidence:
            return None

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()
        return (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

    def close(self):
        """Cleanup method (no actual cleanup needed for YOLO)."""
        pass
