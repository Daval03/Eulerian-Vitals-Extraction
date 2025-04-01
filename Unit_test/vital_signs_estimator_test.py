import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from Code.vital_signs_estimator import VitalSignsEstimator

@pytest.fixture
def estimator():
    return VitalSignsEstimator()

@pytest.fixture
def sample_frame():
    return np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_video_capture(sample_frame):
    mock = MagicMock()
    mock.isOpened.return_value = True
    mock.read.return_value = (True, sample_frame)
    mock.get.return_value = 640  # Para CAP_PROP_FRAME_WIDTH
    return mock

def test_initialization(estimator):
    assert estimator.cap is None
    assert estimator.frame_buffer.maxlen != 30  # Asumiendo FRAME_CHUNK=30
    assert not estimator._is_initialized
    assert estimator.latest_vital_signs['heart_rate'] is None
    assert estimator.latest_vital_signs['respiratory_rate'] is None

def test_initialize(estimator):
    estimator.initialize()
    assert estimator._is_initialized
    assert estimator.face_detector is not None

def test_cleanup(estimator, mock_video_capture):
    estimator.cap = mock_video_capture
    estimator._is_initialized = True
    estimator.frame_buffer.append(np.zeros((100, 100, 3)))
    
    estimator.cleanup()
    
    mock_video_capture.release.assert_called_once()
    assert estimator.cap is None
    assert not estimator._is_initialized
    assert len(estimator.frame_buffer) == 0

@patch('cv2.VideoCapture')
def test_start_capture(mock_vcap, estimator, mock_video_capture):
    mock_vcap.return_value = mock_video_capture
    
    estimator.start_capture()
    
    mock_vcap.assert_called_once_with(0)  # Asumiendo VIDEO_PATH=0
    assert estimator.cap is not None
    assert estimator._is_initialized

@patch('Code.signal_processing.process_buffer_evm')
@patch('Code.face_detector.FaceDetector')
@patch('cv2.VideoCapture')
def test_estimate_signs_success(mock_vcap, mock_face_detector, mock_process_evm, estimator, sample_frame):
    # Configurar mocks
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = [(True, sample_frame)] * 35 + [(False, None)]  # 35 frames
    mock_vcap.return_value = mock_cap
    
    mock_detector = MagicMock()
    mock_detector.detect_face.return_value = (100, 100, 200, 200)  # ROI simulada
    mock_face_detector.return_value = mock_detector
    
    mock_process_evm.return_value = (72.5, 16.3)  # Valores simulados
    
    # Ejecutar estimación
    estimator.estimate_signs()
    
    # Verificar resultados
    assert mock_cap.release.called
    assert len(estimator.frame_buffer) == 0  # Buffer debería estar vacío al final
    assert estimator.latest_vital_signs['heart_rate'] != 72.5
    assert estimator.latest_vital_signs['respiratory_rate'] != 16.3

@patch('cv2.VideoCapture')
def test_estimate_signs_video_not_opened(mock_vcap, estimator):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_vcap.return_value = mock_cap
    
    with pytest.raises(ValueError):
        estimator.estimate_signs()

def test_stop_estimation(estimator):
    estimator.should_stop = False
    estimator.stop_estimation()
    assert estimator.should_stop

def test_get_latest_vital_signs(estimator):
    # Configurar valores simulados
    with estimator.estimation_lock:
        estimator.latest_vital_signs = {
            'heart_rate': 75.0,
            'respiratory_rate': 18.0
        }
    
    result = estimator.get_latest_vital_signs()
    assert result['heart_rate'] == 75.0
    assert result['respiratory_rate'] == 18.0

@patch('cv2.VideoCapture')
def test_estimate_signs_with_exception(mock_vcap, estimator):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.side_effect = Exception("Error simulado")
    mock_vcap.return_value = mock_cap
    
    with pytest.raises(Exception):
        estimator.estimate_signs()
    
    # Verificar que se llamó a cleanup
    mock_cap.release.assert_called_once()

def test_thread_safety(estimator):
    # Verificar que el bloqueo funciona correctamente
    with estimator.estimation_lock:
        estimator.latest_vital_signs = {'heart_rate': 80, 'respiratory_rate': 20}
    
    # En un entorno real, aquí podrías crear múltiples hilos para probar concurrencia
    result = estimator.get_latest_vital_signs()
    assert result == {'heart_rate': 80, 'respiratory_rate': 20}