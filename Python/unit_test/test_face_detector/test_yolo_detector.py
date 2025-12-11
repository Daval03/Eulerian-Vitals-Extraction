import cv2
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.face_detector.yolo_detector import YOLODetector
from src.config import YOLO_MODELS


class TestYOLODetectorPytest:
    """Pytest-style test class for YOLODetector."""
    
    @pytest.fixture
    def mock_yolo_config(self):
        """Fixture to mock YOLO_MODELS config."""
        # Get actual keys from the real config to ensure test matches
        actual_keys = list(YOLO_MODELS.keys())
        return {key: f'/fake/path/{key}-face.pt' for key in actual_keys}
    
    @pytest.fixture
    def detector_with_preset(self, mock_yolo_config):
        """Fixture to provide a detector instance with preset."""
        with patch.dict('src.face_detector.yolo_detector.YOLO_MODELS', mock_yolo_config):
            with patch('src.face_detector.yolo_detector.YOLO') as mock_yolo_class:
                mock_model = MagicMock()
                mock_yolo_class.return_value = mock_model
                # Use the first available preset from the mock config
                first_preset = list(mock_yolo_config.keys())[0]
                return YOLODetector(preset=first_preset, confidence=0.6)
    
    @pytest.fixture
    def detector_with_custom_model(self):
        """Fixture to provide a detector instance with custom model."""
        with patch('src.face_detector.yolo_detector.YOLO') as mock_yolo_class:
            mock_model = MagicMock()
            mock_yolo_class.return_value = mock_model
            return YOLODetector(model_path='/custom/model.pt', confidence=0.7)
    
    @pytest.fixture
    def blank_frame(self):
        """Fixture to provide a blank test frame."""
        return np.ones((100, 100, 3), dtype=np.uint8) * 255
    
    @pytest.fixture
    def frame_with_face(self):
        """Fixture to provide a frame with a simple face pattern."""
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 200
        # Draw a simple face pattern
        cv2.circle(frame, (100, 80), 30, (100, 100, 100), -1)  # Head
        cv2.circle(frame, (85, 70), 5, (50, 50, 50), -1)  # Left eye
        cv2.circle(frame, (115, 70), 5, (50, 50, 50), -1)  # Right eye
        cv2.ellipse(frame, (100, 95), (15, 8), 0, 0, 180, (50, 50, 50), 2)  # Mouth
        return frame
    
    def test_initialization_with_preset(self, mock_yolo_config):
        """Test detector initialization with preset."""
        with patch.dict('src.face_detector.yolo_detector.YOLO_MODELS', mock_yolo_config):
            with patch('src.face_detector.yolo_detector.YOLO') as mock_yolo_class:
                mock_model = MagicMock()
                mock_yolo_class.return_value = mock_model
                
                # Use a preset that exists in the mock config
                test_preset = list(mock_yolo_config.keys())[0]
                detector = YOLODetector(preset=test_preset, confidence=0.75)
                
                # Check that YOLO was called with correct preset path
                mock_yolo_class.assert_called_once_with(mock_yolo_config[test_preset])
                assert detector.confidence == 0.75
                assert detector.model == mock_model
    
    def test_initialization_with_custom_model(self):
        """Test detector initialization with custom model path."""
        with patch('src.face_detector.yolo_detector.YOLO') as mock_yolo_class:
            mock_model = MagicMock()
            mock_yolo_class.return_value = mock_model
            
            custom_path = '/path/to/custom/model.pt'
            detector = YOLODetector(model_path=custom_path, confidence=0.8)
            
            # Check that YOLO was called with custom path
            mock_yolo_class.assert_called_once_with(custom_path)
            assert detector.confidence == 0.8
    
    def test_initialization_no_arguments(self, mock_yolo_config):
        """Test that initialization fails without preset or model_path."""
        with patch.dict('src.face_detector.yolo_detector.YOLO_MODELS', mock_yolo_config):
            with pytest.raises(ValueError) as exc_info:
                YOLODetector()
            
            assert "You must provide either 'preset' or 'model_path'" in str(exc_info.value)
    
    def test_initialization_invalid_preset(self, mock_yolo_config):
        """Test that initialization fails with invalid preset."""
        with patch.dict('src.face_detector.yolo_detector.YOLO_MODELS', mock_yolo_config):
            with pytest.raises(ValueError) as exc_info:
                YOLODetector(preset='invalid_preset')
            
            # Check the error message contains the expected format
            error_msg = str(exc_info.value)
            assert "Unknown YOLO preset 'invalid_preset'" in error_msg
            # Check that it lists the available presets (without assuming order)
            assert "Available: " in error_msg
    
    @patch('src.face_detector.yolo_detector.YOLO')
    def test_detect_no_results(self, mock_yolo_class, blank_frame):
        """Test detect when no results are returned."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Mock empty results
        mock_results = MagicMock()
        mock_results.__len__.return_value = 0
        mock_model.return_value = [mock_results]
        
        detector = YOLODetector(preset='yolov8n')
        result = detector.detect(blank_frame)
        
        assert result is None
    
    @patch('src.face_detector.yolo_detector.YOLO')
    def test_detect_no_boxes(self, mock_yolo_class, blank_frame):
        """Test detect when results have no boxes."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Mock results with no boxes
        mock_results = MagicMock()
        mock_results.__len__.return_value = 1
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 0
        mock_results.boxes = mock_boxes
        mock_model.return_value = [mock_results]
        
        detector = YOLODetector(preset='yolov8n')
        result = detector.detect(blank_frame)
        
        assert result is None
    
    @patch('src.face_detector.yolo_detector.YOLO')
    def test_detect_with_face_high_confidence(self, mock_yolo_class, blank_frame):
        """Test detect when a face is found with high confidence."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Mock results with a face detection
        mock_results = MagicMock()
        mock_results.__len__.return_value = 1
        
        # Mock boxes with one detection
        mock_box = MagicMock()
        mock_box.conf = [0.85]  # Above confidence threshold
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([10.0, 20.0, 50.0, 60.0])
        
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.__iter__.return_value = [mock_box]
        
        mock_results.boxes = mock_boxes
        mock_model.return_value = [mock_results]
        
        detector = YOLODetector(preset='yolov8n', confidence=0.5)
        result = detector.detect(blank_frame)
        
        # Check converted coordinates (XYXY to XYWH)
        assert result == (10, 20, 40, 40)
    
    @patch('src.face_detector.yolo_detector.YOLO')
    def test_detect_low_confidence(self, mock_yolo_class, blank_frame):
        """Test detect when confidence is below threshold."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Mock results with low confidence detection
        mock_results = MagicMock()
        mock_results.__len__.return_value = 1
        
        mock_box = MagicMock()
        mock_box.conf = [0.3]  # Below confidence threshold
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([10.0, 20.0, 50.0, 60.0])
        
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.__iter__.return_value = [mock_box]
        
        mock_results.boxes = mock_boxes
        mock_model.return_value = [mock_results]
        
        detector = YOLODetector(preset='yolov8n', confidence=0.5)
        result = detector.detect(blank_frame)
        
        assert result is None
    
    @patch('src.face_detector.yolo_detector.YOLO')
    def test_detect_multiple_faces(self, mock_yolo_class, blank_frame):
        """Test detect when multiple faces are found (should return highest confidence)."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Mock results with multiple detections
        mock_results = MagicMock()
        mock_results.__len__.return_value = 1
        
        # Create multiple mock boxes with different confidences
        mock_box1 = MagicMock()
        mock_box1.conf = [0.6]
        mock_box1.xyxy = [MagicMock()]
        mock_box1.xyxy[0].cpu.return_value.numpy.return_value = np.array([10.0, 20.0, 50.0, 60.0])
        
        mock_box2 = MagicMock()
        mock_box2.conf = [0.9]  # Highest confidence
        mock_box2.xyxy = [MagicMock()]
        mock_box2.xyxy[0].cpu.return_value.numpy.return_value = np.array([100.0, 120.0, 160.0, 180.0])
        
        mock_box3 = MagicMock()
        mock_box3.conf = [0.7]
        mock_box3.xyxy = [MagicMock()]
        mock_box3.xyxy[0].cpu.return_value.numpy.return_value = np.array([200.0, 220.0, 250.0, 260.0])
        
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 3
        mock_boxes.__iter__.return_value = [mock_box1, mock_box2, mock_box3]
        
        mock_results.boxes = mock_boxes
        mock_model.return_value = [mock_results]
        
        detector = YOLODetector(preset='yolov8n', confidence=0.5)
        result = detector.detect(blank_frame)
        
        # Should return box with highest confidence (mock_box2)
        assert result == (100, 120, 60, 60)
    
    def test_detect_parameters_passed_to_yolo(self, mock_yolo_config, blank_frame):
        """Test that detect passes correct parameters to YOLO model."""
        with patch.dict('src.face_detector.yolo_detector.YOLO_MODELS', mock_yolo_config):
            with patch('src.face_detector.yolo_detector.YOLO') as mock_yolo_class:
                mock_model = MagicMock()
                mock_yolo_class.return_value = mock_model
                
                # Mock empty results
                mock_results = MagicMock()
                mock_results.__len__.return_value = 0
                mock_model.return_value = [mock_results]
                
                # Use a preset that exists in the mock config
                test_preset = list(mock_yolo_config.keys())[0]
                detector = YOLODetector(preset=test_preset)
                detector.detect(blank_frame)
                
                # Verify model was called with correct parameters
                mock_model.assert_called_once_with(blank_frame, verbose=False)
    
    def test_close_method(self, detector_with_preset):
        """Test that close method doesn't raise errors."""
        # Should not raise any exceptions
        detector_with_preset.close()
        
        # After closing, detector should still be usable
        # (YOLO doesn't need explicit cleanup in this implementation)
        assert detector_with_preset.model is not None
    
    def test_detect_invalid_input_type(self, detector_with_preset):
        """Test detect with invalid input types."""
        # YOLO model might raise exceptions for invalid inputs,
        # or the detector might handle them gracefully.
        # Since the current implementation doesn't explicitly validate inputs,
        # we'll test that the model is called (which may raise its own exceptions)
        
        with patch.object(detector_with_preset.model, '__call__') as mock_call:
            # Test with None - this should either raise or return None
            try:
                result = detector_with_preset.detect(None)
                # If no exception, result should be None (or handle gracefully)
                if result is not None:
                    pytest.fail("detect should return None or raise for None input")
            except Exception:
                pass  # Exception is acceptable
            
            # Test with string
            try:
                result = detector_with_preset.detect("not an image")
                if result is not None:
                    pytest.fail("detect should return None or raise for string input")
            except Exception:
                pass  # Exception is acceptable
            
            # Test with wrong shape array
            try:
                result = detector_with_preset.detect(np.array([1, 2, 3]))
                if result is not None:
                    pytest.fail("detect should return None or raise for wrong shape")
            except Exception:
                pass  # Exception is acceptable
    
    def test_detect_small_image(self, detector_with_preset):
        """Test detect with very small image."""
        small_image = np.ones((10, 10, 3), dtype=np.uint8) * 255
        with patch.object(detector_with_preset.model, '__call__') as mock_call:
            mock_results = MagicMock()
            mock_results.__len__.return_value = 0
            mock_call.return_value = [mock_results]
            
            result = detector_with_preset.detect(small_image)
            
            # Should handle small images gracefully
            assert result is None
    
    def test_detect_rgb_vs_bgr(self, detector_with_preset):
        """Test that detect works with both RGB and BGR images."""
        rgb_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        with patch.object(detector_with_preset.model, '__call__') as mock_call:
            # Mock empty results
            mock_results = MagicMock()
            mock_results.__len__.return_value = 0
            mock_call.return_value = [mock_results]
            
            # Both should work without errors
            result_rgb = detector_with_preset.detect(rgb_image)
            result_bgr = detector_with_preset.detect(bgr_image)
            
            assert result_rgb is None
            assert result_bgr is None
    
    def test_confidence_thresholds(self, mock_yolo_config):
        """Test different confidence threshold values."""
        with patch.dict('src.face_detector.yolo_detector.YOLO_MODELS', mock_yolo_config):
            with patch('src.face_detector.yolo_detector.YOLO') as mock_yolo_class:
                mock_model = MagicMock()
                mock_yolo_class.return_value = mock_model
                
                # Test with various confidence values
                for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
                    detector = YOLODetector(preset='yolov8n', confidence=conf)
                    assert detector.confidence == conf
    
    @patch('src.face_detector.yolo_detector.YOLO')
    def test_float_coordinates_conversion(self, mock_yolo_class, blank_frame):
        """Test that float coordinates are properly converted to integers."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        
        # Mock results with float coordinates
        mock_results = MagicMock()
        mock_results.__len__.return_value = 1
        
        mock_box = MagicMock()
        mock_box.conf = [0.8]
        mock_box.xyxy = [MagicMock()]
        # Float coordinates that should be converted to int
        mock_box.xyxy[0].cpu.return_value.numpy.return_value = np.array([10.7, 20.3, 50.9, 60.5])
        
        mock_boxes = MagicMock()
        mock_boxes.__len__.return_value = 1
        mock_boxes.__iter__.return_value = [mock_box]
        
        mock_results.boxes = mock_boxes
        mock_model.return_value = [mock_results]
        
        detector = YOLODetector(preset='yolov8n', confidence=0.5)
        result = detector.detect(blank_frame)
        
        # Coordinates should be converted to integers
        assert result == (10, 20, 40, 40)  # x1:10.7->10, y1:20.3->20, w:40.2->40, h:40.2->40
