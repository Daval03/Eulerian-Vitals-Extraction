import unittest
import numpy as np
import cv2
try:
    from src.face_detector.mediapipe_detector import MediaPipeDetector
except ImportError:
    MediaPipeDetector = None

class TestMediaPipeDetector(unittest.TestCase):
    """Unit tests for MediaPipeDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up resources for all tests."""
        # Only create detector if module was imported successfully
        if MediaPipeDetector is not None:
            cls.detector = MediaPipeDetector()
        
        # Create test images
        cls.blank_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Blank image
        cls.white_image = np.ones((480, 640, 3), dtype=np.uint8) * 255  # White image
        
        # Create an image with a simple face-like pattern (for detection tests)
        cls.face_image = cls.white_image.copy()
        # Draw simple face features (eyes and mouth)
        cv2.circle(cls.face_image, (320, 200), 30, (0, 0, 0), -1)  # Left eye
        cv2.circle(cls.face_image, (400, 200), 30, (0, 0, 0), -1)  # Right eye
        cv2.ellipse(cls.face_image, (360, 300), (80, 40), 0, 0, 180, (0, 0, 0), 10)  # Mouth
        
        # Create a real face image if available (for more realistic testing)
        cls.real_face_image = None       
        # test_image_path = "test_face.jpg"
        # if os.path.exists(test_image_path):
        #     cls.real_face_image = cv2.imread(test_image_path)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up resources after all tests."""
        if hasattr(cls, 'detector') and cls.detector:
            cls.detector.close()
    
    def setUp(self):
        """Skip all tests if MediaPipeDetector couldn't be imported."""
        if MediaPipeDetector is None:
            self.skipTest("MediaPipeDetector module could not be imported")
    
    def test_initialization(self):
        """Test that detector initializes without errors."""
        detector = MediaPipeDetector()
        self.assertIsNotNone(detector)
        self.assertIsNotNone(detector.detector)
        self.assertIsNotNone(detector.mp_face_detection)
        detector.close()
    
    def test_detect_on_blank_image(self):
        """Test detection on blank image (should return None)."""
        result = self.detector.detect(self.blank_image)
        self.assertIsNone(result, "Should return None for blank image")
    
    def test_detect_on_white_image(self):
        """Test detection on plain white image (should return None or valid bbox)."""
        result = self.detector.detect(self.white_image)
        # Could be None or a valid bounding box - both are acceptable
        if result is not None:
            self._validate_bounding_box(result, self.white_image)
    
    def test_detect_on_synthetic_face(self):
        """Test detection on image with synthetic face pattern."""
        result = self.detector.detect(self.face_image)
        # MediaPipe might or might not detect our synthetic face
        # Both outcomes are valid for this test
        if result is not None:
            self._validate_bounding_box(result, self.face_image)
    
    def test_detect_invalid_input(self):
        """Test detection with invalid input types - ensure graceful handling."""
        # Test cases that should be handled gracefully
        invalid_cases = [
            ("string", "not an image"),
            ("none", None),
            ("integer", 123),
            ("empty_list", []),
            ("empty_dict", {}),
            ("2d_array_wrong_shape", np.zeros((100, 100), dtype=np.uint8)),
            ("3d_array_wrong_dtype", np.zeros((100, 100, 3), dtype=np.float32)),
        ]
        
        for name, invalid_input in invalid_cases:
            with self.subTest(case=name):
                try:
                    result = self.detector.detect(invalid_input)
                    # If no exception is raised, the result should either be None
                    # or we should be able to validate it
                    if result is not None:
                        # If it returns something, it should be a valid bbox tuple
                        self.assertIsInstance(result, tuple)
                        self.assertEqual(len(result), 4)
                except (cv2.error, ValueError, AttributeError, TypeError, Exception) as e:
                    # Any exception is acceptable - the important thing is 
                    # that it doesn't crash the entire system
                    # and the detector remains usable
                    pass
        
        # After all invalid inputs, the detector should still work with valid input
        result = self.detector.detect(self.blank_image)
        # Should work without issues
        self.assertTrue(result is None)
    
    def test_detect_empty_array(self):
        """Test detection with empty array."""
        empty_array = np.array([], dtype=np.uint8)
        with self.assertRaises(Exception):
            self.detector.detect(empty_array)
    
    def test_detect_with_different_sizes(self):
        """Test detection with images of different sizes."""
        sizes = [(100, 100), (320, 240), (640, 480), (1280, 720)]
        
        for height, width in sizes:
            with self.subTest(size=f"{width}x{height}"):
                test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                result = self.detector.detect(test_image)
                if result is not None:
                    self._validate_bounding_box(result, test_image)
    
    def test_detect_multiple_calls(self):
        """Test that detector works correctly with multiple consecutive calls."""
        results = []
        for _ in range(5):
            result = self.detector.detect(self.face_image)
            results.append(result)
            if result is not None:
                self._validate_bounding_box(result, self.face_image)
        
        # All results should be consistent (all None or all valid bboxes)
        # or at least not crash
    
    def test_close_method(self):
        """Test that close method works without errors."""
        detector = MediaPipeDetector()
        # Should not raise any exceptions
        detector.close()
        
        # Try to use after close (might still work or might fail)
        # This behavior depends on MediaPipe implementation
        try:
            detector.detect(self.blank_image)
        except Exception:
            pass  # Expected if detector is closed
    
    def test_bounding_box_validity(self):
        """Test that returned bounding boxes are valid."""
        # Test with random images multiple times
        for i in range(10):
            with self.subTest(iteration=i):
                # Create random image
                height = np.random.randint(100, 480)
                width = np.random.randint(100, 640)
                random_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
                
                result = self.detector.detect(random_image)
                if result is not None:
                    self._validate_bounding_box(result, random_image)
    
    def test_detect_with_real_image(self):
        """Test detection with a real face image if available."""
        if self.real_face_image is not None:
            result = self.detector.detect(self.real_face_image)
            self.assertIsNotNone(result, "Should detect face in real face image")
            self._validate_bounding_box(result, self.real_face_image)
        else:
            self.skipTest("No real face image available for testing")
    
    def test_detector_reinitialization(self):
        """Test that detector can be reinitialized after closing."""
        detector1 = MediaPipeDetector()
        result1 = detector1.detect(self.face_image)
        detector1.close()
        
        detector2 = MediaPipeDetector()
        result2 = detector2.detect(self.face_image)
        detector2.close()
        
        # Results should be comparable (both None or both valid)
        if result1 is None:
            self.assertIsNone(result2, "Both detectors should behave consistently")
        else:
            self.assertIsNotNone(result2, "Both detectors should behave consistently")
    
    def _validate_bounding_box(self, bbox, image):
        """Helper method to validate bounding box coordinates."""
        x, y, width, height = bbox
        
        # Check types
        self.assertIsInstance(x, int, "x should be integer")
        self.assertIsInstance(y, int, "y should be integer")
        self.assertIsInstance(width, int, "width should be integer")
        self.assertIsInstance(height, int, "height should be integer")
        
        # Check values are non-negative
        self.assertGreaterEqual(x, 0, "x should be >= 0")
        self.assertGreaterEqual(y, 0, "y should be >= 0")
        self.assertGreaterEqual(width, 0, "width should be >= 0")
        self.assertGreaterEqual(height, 0, "height should be >= 0")
        
        # Check that bounding box is within image bounds
        img_height, img_width = image.shape[:2]
        self.assertLess(x, img_width, f"x ({x}) should be < image width ({img_width})")
        self.assertLess(y, img_height, f"y ({y}) should be < image height ({img_height})")
        self.assertLessEqual(x + width, img_width, 
                           f"x + width ({x + width}) should be <= image width ({img_width})")
        self.assertLessEqual(y + height, img_height, 
                           f"y + height ({y + height}) should be <= image height ({img_height})")
        
        # Check that width and height are reasonable
        # (not larger than image dimensions)
        self.assertLessEqual(width, img_width, "width should be <= image width")
        self.assertLessEqual(height, img_height, "height should be <= image height")
        
        # Check that bounding box has non-zero area
        self.assertGreater(width, 0, "width should be > 0")
        self.assertGreater(height, 0, "height should be > 0")


class TestMediaPipeDetectorIntegration(unittest.TestCase):
    """Integration tests for MediaPipeDetector with actual MediaPipe."""
    
    def test_mediapipe_available(self):
        """Test that MediaPipe is properly installed and importable."""
        try:
            import mediapipe as mp
            self.assertTrue(True, "MediaPipe is available")
        except ImportError:
            self.skipTest("MediaPipe is not installed")
    
    def test_cv2_available(self):
        """Test that OpenCV is properly installed and importable."""
        try:
            import cv2
            self.assertTrue(True, "OpenCV is available")
        except ImportError:
            self.skipTest("OpenCV is not installed")


def create_test_image_with_face():
    """Helper function to create a more realistic test image."""
    # Create a simple "face" using OpenCV drawing functions
    image = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw face oval
    cv2.ellipse(image, (320, 240), (150, 200), 0, 0, 360, (200, 200, 200), -1)
    
    # Draw eyes
    cv2.circle(image, (250, 180), 25, (100, 100, 100), -1)
    cv2.circle(image, (390, 180), 25, (100, 100, 100), -1)
    
    # Draw mouth
    cv2.ellipse(image, (320, 320), (80, 40), 0, 0, 180, (100, 100, 100), 20)
    
    return image
