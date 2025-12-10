import mediapipe as mp
import cv2

class MediaPipeDetector:
    """
    Face detector using MediaPipe's face detection model.
    
    Detects a single face in an image with MediaPipe's short-range model.
    Returns bounding box coordinates (x, y, width, height) for the highest-confidence face.
    
    Attributes:
        mp_face_detection: MediaPipe face detection module
        detector: MediaPipe face detection instance with configured parameters
    """
    def __init__(self):
        """Initializes MediaPipe face detector with default parameters."""
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=0, 
            min_detection_confidence=0.5
        )
    
    def detect(self, frame):
        """
        Detects a face in the given frame.
        
        Args:
            frame (numpy.ndarray): Input image in BGR format (OpenCV default)
            
        Returns:
            tuple or None: Bounding box as (x, y, width, height) in pixels, 
                          or None if no face detected
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            return (x, y, width, height)
        return None
    def close(self):
        """Cleanup method (no actual cleanup needed for MediaPipe)."""
        pass