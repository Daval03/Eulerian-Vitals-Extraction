import pytest
import numpy as np
import cv2
from unittest.mock import MagicMock
from Code.face_detector import FaceDetector

@pytest.fixture
def face_detector():
    return FaceDetector()

@pytest.fixture
def sample_frame():
    # Crear un frame de prueba (imagen negra con un rectángulo blanco que simula una cara)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(frame, (200, 150), (400, 350), (255, 255, 255), -1)
    return frame

@pytest.fixture
def mock_detection():
    detection = MagicMock()
    detection.location_data.relative_bounding_box.xmin = 0.3125  # 200/640
    detection.location_data.relative_bounding_box.ymin = 0.3125  # 150/480
    detection.location_data.relative_bounding_box.width = 0.3125  # 200/640
    detection.location_data.relative_bounding_box.height = 0.4167  # 200/480
    return detection

def test_face_detector_initialization():
    detector = FaceDetector(model_selection=1, min_detection_confidence=0.7)
    assert detector is not None
    assert detector.roi_history.maxlen == 5
    assert detector.stabilized_roi is None

def test_stabilize_roi_with_empty_history(face_detector):
    test_roi = (100, 100, 200, 200)
    stabilized = face_detector.stabilize_roi(test_roi)
    assert stabilized == test_roi

def test_stabilize_roi_with_history(face_detector):
    # Llenar el historial con ROIs similares
    for i in range(5):
        face_detector.roi_history.append((100 + i, 100 + i, 200 + i, 200 + i))
    
    new_roi = (105, 105, 205, 205)
    stabilized = face_detector.stabilize_roi(new_roi)
    
    # Verificar que la ROI estabilizada es diferente a la nueva
    assert stabilized != new_roi
    # Verificar que está dentro del rango de valores
    assert 100 <= stabilized[0] <= 105
    assert 100 <= stabilized[1] <= 105
    assert 200 <= stabilized[2] <= 205
    assert 200 <= stabilized[3] <= 205

def test_is_significant_change(face_detector):
    # Caso sin ROIs previas
    assert face_detector.is_significant_change(None, (100, 100, 200, 200)) is True
    
    # Cambio pequeño (dentro del umbral)
    old_roi = (100, 100, 200, 200)
    new_roi = (101, 101, 201, 201)
    assert face_detector.is_significant_change(old_roi, new_roi) is False
    
    # Cambio grande en X (fuera del umbral)
    new_roi = (100 + 20, 100, 200, 200)
    assert face_detector.is_significant_change(old_roi, new_roi) is False

def test_detect_face_with_face(face_detector, sample_frame, mock_detection):
    # Mockear el proceso de detección
    face_detector.face_detection.process = MagicMock(return_value=MagicMock(detections=[mock_detection]))
    
    roi = face_detector.detect_face(sample_frame)
    assert roi is not None
    assert len(roi) == 4
    x, y, w, h = roi
    
    # Verificar que la ROI está dentro del frame
    assert 0 <= x < sample_frame.shape[1]
    assert 0 <= y < sample_frame.shape[0]
    assert x + w <= sample_frame.shape[1]
    assert y + h <= sample_frame.shape[0]

def test_detect_face_without_face(face_detector):
    empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    face_detector.face_detection.process = MagicMock(return_value=MagicMock(detections=[]))
    
    roi = face_detector.detect_face(empty_frame)
    assert roi is None

def test_detect_face_stabilization(face_detector, mock_detection):
    # Configurar mock para devolver diferentes ROIs en cada llamada
    mock_results = MagicMock()
    mock_results.detections = [mock_detection]
    face_detector.face_detection.process = MagicMock(return_value=mock_results)
    
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Primera detección
    roi1 = face_detector.detect_face(frame)
    
    # Modificar ligeramente la detección
    mock_detection.location_data.relative_bounding_box.xmin = 0.325
    mock_detection.location_data.relative_bounding_box.ymin = 0.325
    
    # Segunda detección
    roi2 = face_detector.detect_face(frame)
    
    # Tercera detección
    mock_detection.location_data.relative_bounding_box.xmin = 0.3375
    mock_detection.location_data.relative_bounding_box.ymin = 0.3375
    roi3 = face_detector.detect_face(frame)
    
    # Verificar que las ROIs se estabilizan
    assert roi1 != roi2
    assert roi2 != roi3
    assert face_detector.stabilized_roi is not None

def test_detect_face_edge_cases(face_detector, mock_detection):
    # Configurar mock
    face_detector.face_detection.process = MagicMock(return_value=MagicMock(detections=[mock_detection]))
    
    # Caso donde la ROI calculada estaría fuera del frame
    mock_detection.location_data.relative_bounding_box.xmin = 0.9
    mock_detection.location_data.relative_bounding_box.width = 0.2
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    roi = face_detector.detect_face(frame)
    assert roi is not None
    x, y, w, h = roi
    assert x + w <= frame.shape[1]  # No se sale del frame
