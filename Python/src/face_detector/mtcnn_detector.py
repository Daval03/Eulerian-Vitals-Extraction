import os,sys
sys.path.append(os.path.dirname(__file__))

import cv2
from mtcnn import MTCNN
from base import BaseFaceDetector

class MTCNNDetector(BaseFaceDetector):
    """
    Face detector using MTCNN (Multi-task Cascaded Convolutional Networks).
    
    Detects faces with configurable confidence threshold and returns bounding box
    for the highest-confidence detection that meets the threshold.
    
    Attributes:
        detector: MTCNN detector instance
        min_confidence: Minimum confidence threshold for valid detections
    """
    def __init__(self, min_confidence=0.9):
        """
        Initializes MTCNN detector with confidence threshold.
        
        Args:
            min_confidence (float): Minimum confidence score (0-1) for a detection 
                                   to be considered valid. Default: 0.9
        """
        self.detector = MTCNN()
        self.min_confidence = min_confidence

    def detect(self, frame):
        """
        Detects a face in the given frame.
        
        Args:
            frame (numpy.ndarray): Input image in BGR format
            
        Returns:
            tuple or None: Bounding box as (x, y, width, height) in pixels, 
                          or None if no face meets confidence threshold
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self.detector.detect_faces(rgb)
        if detections and detections[0]['confidence'] > self.min_confidence:
            return tuple(detections[0]['box'])
        return None

    def close(self):
        """Cleanup method (no actual cleanup needed for MTCNN)."""
        pass
