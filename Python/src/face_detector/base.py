from abc import ABC, abstractmethod

class BaseFaceDetector(ABC):
    """
    Abstract base class for face detection implementations.
    
    Defines the required interface for any face detector class.
    
    Methods:
        detect(frame): Detects faces in an image frame.
        close(): Releases any allocated resources.
    """
    
    @abstractmethod
    def detect(self, frame):
        """
        Detect faces in the provided frame.
        
        Args:
            frame (numpy.ndarray): Input image in BGR format.
            
        Returns:
            tuple or None: Face coordinates as (x, y, w, h) or None if no face detected.
        """
        pass

    @abstractmethod
    def close(self):
        """
        Release any resources used by the detector.
        
        Should be called when the detector is no longer needed.
        """
        pass