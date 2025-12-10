import cv2
import os,sys
sys.path.append(os.path.dirname(__file__))
from base import BaseFaceDetector

class HaarCascadeDetector(BaseFaceDetector):
    """
    Face detector using OpenCV's Haar Cascade classifier.
    
    This implementation uses the pre-trained frontal face Haar cascade
    classifier for face detection in grayscale images.
    
    Attributes:
        cascade (cv2.CascadeClassifier): Loaded Haar cascade classifier.
        scale_factor (float): Scale factor for multi-scale detection.
        min_neighbors (int): Minimum neighbors for detection reliability.
        min_size (tuple): Minimum face size (width, height) to detect.
    
    Args:
        scale_factor (float, optional): Scale factor between 1.01-1.5. Defaults to 1.1.
        min_neighbors (int, optional): Higher values reduce false positives. Defaults to 4.
        min_size (tuple, optional): Minimum face size. Defaults to (30, 30).
    """
    def __init__(self, scale_factor=1.1, min_neighbors=4, min_size=(30, 30)):
        """
        Initialize the Haar Cascade detector with specified parameters.
        """
        self.cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_size = min_size

    def detect(self, frame):
        """
        Detect a single face in the input frame.
        
        Args:
            frame (numpy.ndarray): Input image in BGR format.
            
        Returns:
            tuple or None: Coordinates of the first detected face as (x, y, w, h),
                          or None if no faces are detected.
                          
        Note:
            Only returns the first detected face. Converts the frame to grayscale
            internally as required by Haar Cascade.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.cascade.detectMultiScale(
            gray, self.scale_factor, self.min_neighbors, minSize=self.min_size
        )
        return tuple(faces[0]) if len(faces) else None

    def close(self):
        """
        Release detector resources.
        
        Note: Haar Cascade classifier doesn't require explicit cleanup
        in OpenCV, but this method is included for interface compatibility.
        """
        pass
