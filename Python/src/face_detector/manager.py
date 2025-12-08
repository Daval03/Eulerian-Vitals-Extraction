import os
import sys
sys.path.append(os.path.dirname(__file__))

from collections import deque
from haar_detector import HaarCascadeDetector
from mtcnn_detector import MTCNNDetector
from yolo_detector import YOLODetector
from mediapipe_detector import MediaPipeDetector
from src.config import ROI_WEIGHTS, ROI_CHANGE_THRESHOLD


class FaceDetector:
    """
    Unified face detector manager with ROI stabilization.
    
    This class provides a common interface to multiple face detection models
    and includes stabilization to reduce jitter and false positives in
    face detection results.
    
    Attributes:
        MODELS (dict): Available face detection models.
        detector (BaseFaceDetector): Current detector instance.
        roi_history (deque): History of recent ROI detections for stabilization.
        stabilized_roi (tuple or None): Current stabilized ROI coordinates.
    
    Args:
        model_type (str): Type of detector to use. Options: 'haar', 'mtcnn', 'yolo', 'mediapipe'.
        **kwargs: Additional parameters passed to the specific detector.
    
    Raises:
        ValueError: If an invalid model_type is provided.
    """
    
    # Available face detection models
    MODELS = {
        "haar": HaarCascadeDetector,        # Haar Cascade classifier
        "mtcnn": MTCNNDetector,             # Multi-task Cascaded CNN
        "yolo": YOLODetector,               # YOLO-based detector
        "mediapipe": MediaPipeDetector      # MediaPipe Face Detection
    }

    def __init__(self, model_type="haar", **kwargs):
        """
        Initialize the face detector with specified model type.
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Invalid model type '{model_type}'. Available: {list(self.MODELS.keys())}")
        
        # Initialize the selected detector
        self.detector = self.MODELS[model_type](**kwargs)
        
        # ROI stabilization buffers
        self.roi_history = deque(maxlen=5)  # Store last 5 detections
        self.stabilized_roi = None  # Current stabilized ROI

    def stabilize_roi(self, new_roi):
        """
        Apply temporal smoothing to ROI using weighted averaging.
        
        Uses a weighted average of recent ROIs to reduce jitter in detection.
        Recent detections are given higher weight according to ROI_WEIGHTS.
        
        Args:
            new_roi (tuple or None): New ROI detection (x, y, w, h) or None.
            
        Returns:
            tuple or None: Stabilized ROI coordinates or previous stable ROI.
        """
        if new_roi is None:
            # Return previous stabilized ROI if no new detection
            return self.stabilized_roi

        # Add new detection to history
        self.roi_history.append(new_roi)

        # Need minimum history for stabilization
        if len(self.roi_history) < 3:
            return new_roi
        
        # Calculate weighted average of historical ROIs
        weights = ROI_WEIGHTS[:len(self.roi_history)]
        weights = [w / sum(weights) for w in weights]  # Normalize weights

        # Weighted average calculation for each dimension
        x_roi = int(sum(r[0] * w for r, w in zip(self.roi_history, weights)))
        y_roi = int(sum(r[1] * w for r, w in zip(self.roi_history, weights)))
        w_roi = int(sum(r[2] * w for r, w in zip(self.roi_history, weights)))
        h_roi = int(sum(r[3] * w for r, w in zip(self.roi_history, weights)))

        # Update and return stabilized ROI
        self.stabilized_roi = (x_roi, y_roi, w_roi, h_roi)
        return self.stabilized_roi

    def detect_face(self, frame):
        """
        Detect face in frame with stabilization and bounds checking.
        
        Performs face detection, applies stabilization, and ensures the ROI
        stays within frame boundaries.
        
        Args:
            frame (numpy.ndarray): Input image frame.
            
        Returns:
            tuple or None: Stabilized ROI coordinates (x, y, w, h) or None.
        """
        # Get detection from underlying detector
        roi = self.detector.detect(frame)
        if roi is None:
            return None

        # Skip stabilization if change is not significant
        if self.stabilized_roi and not self.is_significant_change(self.stabilized_roi, roi):
            return self.stabilized_roi
        
        # Apply stabilization to new detection
        stable = self.stabilize_roi(roi)
        if stable:
            sx, sy, sw, sh = stable
            
            # Ensure ROI stays within frame boundaries
            sx = max(0, sx)
            sy = max(0, sy)
            sw = min(frame.shape[1] - sx, sw)
            sh = min(frame.shape[0] - sy, sh)
            
            return (sx, sy, sw, sh)
        
        return None
    
    def is_significant_change(self, old_roi, new_roi):
        """
        Check if ROI change exceeds threshold to avoid unnecessary updates.
        
        Args:
            old_roi (tuple): Previous ROI (x, y, w, h).
            new_roi (tuple): Current ROI (x, y, w, h).
            
        Returns:
            bool: True if change is significant, False otherwise.
        """
        if old_roi is None or new_roi is None:
            return True

        # Extract coordinates
        x1, y1, w1, h1 = old_roi
        x2, y2, w2, h2 = new_roi

        # Calculate absolute differences
        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        dw = abs(w1 - w2)
        dh = abs(h1 - h2)

        # Check if any dimension change exceeds threshold
        return (dx > ROI_CHANGE_THRESHOLD or dy > ROI_CHANGE_THRESHOLD or 
                dw > ROI_CHANGE_THRESHOLD or dh > ROI_CHANGE_THRESHOLD)

    def close(self):
        """
        Release resources used by the detector.
        
        Should be called when the detector is no longer needed.
        """
        self.detector.close()