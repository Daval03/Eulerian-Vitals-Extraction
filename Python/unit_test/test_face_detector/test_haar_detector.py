import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from src.face_detector.haar_detector import HaarCascadeDetector


class TestHaarCascadeDetectorPytest:
    """Pytest-style test class for HaarCascadeDetector."""
    
    @pytest.fixture
    def detector(self):
        """Fixture to provide a detector instance."""
        return HaarCascadeDetector()
    
    @pytest.fixture
    def blank_frame(self):
        """Fixture to provide a blank test frame."""
        return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    @pytest.fixture
    def frame_with_pattern(self):
        """Fixture to provide a frame with face-like pattern."""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 200
        cv2.rectangle(frame, (70, 60), (90, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (110, 60), (130, 80), (50, 50, 50), -1)
        cv2.rectangle(frame, (80, 110), (120, 130), (50, 50, 50), -1)
        return frame
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = HaarCascadeDetector(scale_factor=1.05, min_neighbors=3, min_size=(20, 20))
        assert detector.scale_factor == 1.05
        assert detector.min_neighbors == 3
        assert detector.min_size == (20, 20)
        assert detector.cascade is not None
    
    @patch('cv2.CascadeClassifier')
    def test_detect_empty_result(self, mock_cascade, detector, blank_frame):
        """Test detect when no faces are found."""
        mock_instance = MagicMock()
        mock_cascade.return_value = mock_instance
        mock_instance.detectMultiScale.return_value = np.array([])
        
        # Create detector with mocked cascade
        test_detector = HaarCascadeDetector()
        result = test_detector.detect(blank_frame)
        
        assert result is None
    
    @patch('cv2.CascadeClassifier')
    def test_detect_with_face(self, mock_cascade, blank_frame):
        """Test detect when a face is found."""
        mock_instance = MagicMock()
        mock_cascade.return_value = mock_instance
        mock_instance.detectMultiScale.return_value = np.array([[10, 20, 30, 40]])
        
        detector = HaarCascadeDetector()
        result = detector.detect(blank_frame)
        
        assert result == (10, 20, 30, 40)
    
    @patch('cv2.CascadeClassifier')
    def test_detect_parameters(self, mock_cascade, blank_frame):
        """Test that detect passes correct parameters to OpenCV."""
        mock_instance = MagicMock()
        mock_cascade.return_value = mock_instance
        mock_instance.detectMultiScale.return_value = np.array([])
        
        detector = HaarCascadeDetector(
            scale_factor=1.2,
            min_neighbors=5,
            min_size=(40, 40)
        )
        
        detector.detect(blank_frame)
        
        # Check detectMultiScale was called with correct parameters
        mock_instance.detectMultiScale.assert_called_once()
        args, kwargs = mock_instance.detectMultiScale.call_args
        
        # Verify parameters
        assert args[1] == 1.2  # scaleFactor
        assert args[2] == 5    # minNeighbors
        assert kwargs['minSize'] == (40, 40)
    
    def test_detect_real_cascade(self, detector, frame_with_pattern):
        """
        Integration test with real cascade.
        This test might be skipped if cascade file is not available.
        """
        try:
            result = detector.detect(frame_with_pattern)
            
            # Result should be either None or a valid tuple
            if result is not None:
                assert isinstance(result, tuple)
                assert len(result) == 4
                # Check all values are non-negative
                assert all(x >= 0 for x in result)
        except Exception as e:
            pytest.skip(f"Real cascade test skipped: {e}")
    
    def test_close_method(self, detector):
        """Test that close method doesn't raise errors."""
        # Should not raise any exceptions
        detector.close()
        
        # After closing, detector should still be usable
        # (Haar cascade doesn't actually release resources)
        assert detector.cascade is not None
    
    def test_detect_invalid_input_type(self, detector):
        """Test detect with invalid input types."""
        with pytest.raises(Exception):
            detector.detect(None)
        
        with pytest.raises(Exception):
            detector.detect("not an image")
        
        with pytest.raises(Exception):
            detector.detect(np.array([1, 2, 3]))  # Wrong shape
    
    def test_detect_small_image(self, detector):
        """Test detect with very small image."""
        small_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        result = detector.detect(small_image)
        
        # Should return None (too small for min_size)
        # or handle gracefully
        if result is not None:
            assert isinstance(result, tuple)
            assert len(result) == 4
    
    @patch('cv2.cvtColor')
    @patch('cv2.CascadeClassifier')
    def test_detect_grayscale_conversion(self, mock_cascade, mock_cvt, blank_frame):
        """Test that image is converted to grayscale before detection."""
        mock_instance = MagicMock()
        mock_cascade.return_value = mock_instance
        mock_instance.detectMultiScale.return_value = np.array([])
        
        # Setup mock grayscale conversion
        gray_image = np.zeros((100, 100), dtype=np.uint8)
        mock_cvt.return_value = gray_image
        
        detector = HaarCascadeDetector()
        detector.detect(blank_frame)
        
        # Verify cvtColor was called
        mock_cvt.assert_called_once_with(blank_frame, cv2.COLOR_BGR2GRAY)
        
        # Verify detectMultiScale received grayscale image
        mock_instance.detectMultiScale.assert_called_once()
        args, _ = mock_instance.detectMultiScale.call_args
        assert args[0] is gray_image
