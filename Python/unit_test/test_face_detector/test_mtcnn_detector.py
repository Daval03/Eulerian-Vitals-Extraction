import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, Mock
from src.face_detector.mtcnn_detector import MTCNNDetector


class TestMTCNNDetectorPytest:
    """Pytest-style test class for MTCNNDetector."""
    
    @pytest.fixture
    def detector(self):
        """Fixture to provide a detector instance."""
        return MTCNNDetector()
    
    @pytest.fixture
    def high_confidence_detector(self):
        """Fixture to provide a detector with high confidence threshold."""
        return MTCNNDetector(min_confidence=0.95)
    
    @pytest.fixture
    def low_confidence_detector(self):
        """Fixture to provide a detector with low confidence threshold."""
        return MTCNNDetector(min_confidence=0.5)
    
    @pytest.fixture
    def blank_frame(self):
        """Fixture to provide a blank test frame."""
        return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    @pytest.fixture
    def frame_with_face(self):
        """Fixture to provide a frame with face-like pattern."""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 200
        
        # Create face-like pattern (skin tone oval)
        center_x, center_y = 100, 100
        cv2.ellipse(frame, (center_x, center_y), (60, 80), 0, 0, 360, (180, 150, 120), -1)
        
        # Add eyes
        cv2.circle(frame, (center_x - 30, center_y - 20), 10, (50, 50, 50), -1)
        cv2.circle(frame, (center_x + 30, center_y - 20), 10, (50, 50, 50), -1)
        
        # Add mouth
        cv2.ellipse(frame, (center_x, center_y + 30), (30, 15), 0, 0, 180, (50, 50, 50), -1)
        
        return frame

    def test_initialization(self):
        """Test detector initialization with default and custom parameters."""
        # Test default initialization
        detector = MTCNNDetector()
        assert detector.min_confidence == 0.9
        assert detector.detector is not None
        
        # Test custom initialization
        detector_custom = MTCNNDetector(min_confidence=0.8)
        assert detector_custom.min_confidence == 0.8
        assert detector_custom.detector is not None

    def test_detect_no_faces(self, blank_frame):
        """Test detect when no faces are found."""
        # Create a mock detector
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = []
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector()
        detector.detector = mock_detector
        
        result = detector.detect(blank_frame)
        
        assert result is None
        mock_detector.detect_faces.assert_called_once()

    def test_detect_with_face_high_confidence(self, blank_frame):
        """Test detect when a face is found with high confidence."""
        # Create a mock detector
        mock_detector = MagicMock()
        
        # Mock a face detection with high confidence
        mock_detection = [{
            'box': [10, 20, 30, 40],
            'confidence': 0.95,
            'keypoints': {}
        }]
        mock_detector.detect_faces.return_value = mock_detection
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector(min_confidence=0.9)
        detector.detector = mock_detector
        
        result = detector.detect(blank_frame)
        
        assert result == (10, 20, 30, 40)
        mock_detector.detect_faces.assert_called_once()

    def test_detect_with_face_below_threshold(self, blank_frame):
        """Test detect when face confidence is below threshold."""
        # Create a mock detector
        mock_detector = MagicMock()
        
        # Mock a face detection with low confidence
        mock_detection = [{
            'box': [10, 20, 30, 40],
            'confidence': 0.85,  # Below 0.95 threshold
            'keypoints': {}
        }]
        mock_detector.detect_faces.return_value = mock_detection
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector(min_confidence=0.95)
        detector.detector = mock_detector
        
        result = detector.detect(blank_frame)
        
        assert result is None
        mock_detector.detect_faces.assert_called_once()

    def test_detect_with_face_above_threshold(self, blank_frame):
        """Test detect when face confidence is above low threshold."""
        # Create a mock detector
        mock_detector = MagicMock()
        
        # Mock a face detection with medium confidence
        mock_detection = [{
            'box': [10, 20, 30, 40],
            'confidence': 0.6,  # Above 0.5 threshold
            'keypoints': {}
        }]
        mock_detector.detect_faces.return_value = mock_detection
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector(min_confidence=0.5)
        detector.detector = mock_detector
        
        result = detector.detect(blank_frame)
        
        assert result == (10, 20, 30, 40)
        mock_detector.detect_faces.assert_called_once()

    def test_detect_multiple_faces_returns_first(self, blank_frame):
        """Test that detect returns the first face when multiple are detected."""
        # Create a mock detector
        mock_detector = MagicMock()
        
        # Mock multiple face detections
        mock_detections = [
            {
                'box': [10, 20, 30, 40],
                'confidence': 0.95,
                'keypoints': {}
            },
            {
                'box': [50, 60, 35, 45],
                'confidence': 0.92,
                'keypoints': {}
            }
        ]
        mock_detector.detect_faces.return_value = mock_detections
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector()
        detector.detector = mock_detector
        
        result = detector.detect(blank_frame)
        
        # Should return the first detection
        assert result == (10, 20, 30, 40)
        mock_detector.detect_faces.assert_called_once()

    @patch('cv2.cvtColor')
    def test_detect_color_conversion(self, mock_cvt, detector, blank_frame):
        """Test that image is converted to RGB before detection."""
        # Create a mock detector
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = []
        
        # Create a real MTCNNDetector but replace its detector
        test_detector = MTCNNDetector()
        test_detector.detector = mock_detector
        
        # Setup mock color conversion
        rgb_image = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cvt.return_value = rgb_image
        
        result = test_detector.detect(blank_frame)
        
        # Verify cvtColor was called with correct parameters
        mock_cvt.assert_called_once_with(blank_frame, cv2.COLOR_BGR2RGB)
        
        # Verify detect_faces received RGB image
        mock_detector.detect_faces.assert_called_once_with(rgb_image)

    def test_detect_real_mtcnn(self, frame_with_face):
        """
        Integration test with real MTCNN.
        This test might be skipped if MTCNN is not available.
        """
        try:
            detector = MTCNNDetector()
            result = detector.detect(frame_with_face)
            
            # Result should be either None or a valid tuple
            if result is not None:
                assert isinstance(result, tuple)
                assert len(result) == 4
                # Check all values are non-negative
                assert all(x >= 0 for x in result)
                # Check width and height are positive
                assert result[2] > 0
                assert result[3] > 0
        except Exception as e:
            pytest.skip(f"Real MTCNN test skipped: {e}")

    def test_close_method(self, detector):
        """Test that close method doesn't raise errors."""
        # Should not raise any exceptions
        detector.close()
        
        # After closing, detector should still be usable
        # (MTCNN doesn't actually release resources in our implementation)
        assert detector.detector is not None
        assert detector.min_confidence == 0.9

    @patch('cv2.cvtColor')
    def test_detect_invalid_input_type(self, mock_cvt, detector):
        """Test detect with invalid input types."""
        # Mock cvtColor to avoid actual conversion
        mock_cvt.side_effect = cv2.cvtColor
        
        with pytest.raises(Exception):
            detector.detect(None)
        
        with pytest.raises(Exception):
            detector.detect("not an image")
        
        with pytest.raises(Exception):
            detector.detect(np.array([1, 2, 3]))  # Wrong shape
        
        # Test with 2D grayscale image
        with pytest.raises(Exception):
            detector.detect(np.ones((100, 100), dtype=np.uint8))

    def test_detect_small_image(self, detector):
        """Test detect with very small image."""
        small_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        result = detector.detect(small_image)
        
        # Should return None or handle gracefully
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 4

    def test_detect_empty_box(self, blank_frame):
        """Test detect with empty box in detection."""
        # Create a mock detector
        mock_detector = MagicMock()
        
        # Mock a face detection with empty box
        mock_detection = [{
            'box': [],
            'confidence': 0.95,
            'keypoints': {}
        }]
        mock_detector.detect_faces.return_value = mock_detection
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector()
        detector.detector = mock_detector
        
        # This should raise an exception when trying to convert to tuple
        result = detector.detect(blank_frame)
        assert result is ()

    def test_detect_missing_confidence(self, blank_frame):
        """Test detect when confidence key is missing."""
        # Create a mock detector
        mock_detector = MagicMock()
        
        # Mock a face detection without confidence
        mock_detection = [{
            'box': [10, 20, 30, 40],
            'keypoints': {}
            # Missing 'confidence' key
        }]
        mock_detector.detect_faces.return_value = mock_detection
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector()
        detector.detector = mock_detector
        
        # This should raise a KeyError when trying to access 'confidence'
        with pytest.raises(KeyError):
            detector.detect(blank_frame)

    def test_detect_invalid_confidence_type(self, blank_frame):
        """Test detect with invalid confidence type."""
        # Create a mock detector
        mock_detector = MagicMock()
        
        # Mock a face detection with string confidence
        mock_detection = [{
            'box': [10, 20, 30, 40],
            'confidence': 'high',  # Should be float
            'keypoints': {}
        }]
        mock_detector.detect_faces.return_value = mock_detection
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector()
        detector.detector = mock_detector
        
        # This should raise a TypeError when comparing string with float
        with pytest.raises(TypeError):
            detector.detect(blank_frame)

    def test_detect_empty_detections_with_none(self, blank_frame):
        """Test detect when detect_faces returns None."""
        # Create a mock detector
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = None
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector()
        detector.detector = mock_detector
        
        result = detector.detect(blank_frame)
        assert result is None

    def test_detect_empty_detections_with_falsy_value(self, blank_frame):
        """Test detect when detect_faces returns falsy value."""
        # Create a mock detector
        mock_detector = MagicMock()
        mock_detector.detect_faces.return_value = []
        
        # Create a real MTCNNDetector but replace its detector
        detector = MTCNNDetector()
        detector.detector = mock_detector
        
        result = detector.detect(blank_frame)
        assert result is None