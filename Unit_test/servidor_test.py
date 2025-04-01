import pytest
from unittest.mock import MagicMock, patch
from Code.servidor import app

@pytest.fixture
def client():
    """Cliente de prueba para la aplicación Flask"""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def mock_estimator():
    """Mock del estimador de signos vitales"""
    mock = MagicMock()
    mock.get_latest_vital_signs.return_value = {
        'heart_rate': 75,
        'respiratory_rate': 16
    }
    return mock

def test_start_endpoint(client, mock_estimator):
    """Prueba que el endpoint /start funciona correctamente"""
    with patch('Code.servidor.estimator', mock_estimator):
        response = client.get('/start')
        assert response.status_code == 200
        assert b"Estimation started" in response.data

def test_stop_endpoint(client, mock_estimator):
    """Prueba que el endpoint /stop funciona correctamente"""
    with patch('Code.servidor.estimator', mock_estimator), \
         patch('Code.servidor.is_running', True):
        response = client.get('/stop')
        assert response.status_code == 200
        assert b"stopped successfully" in response.data

def test_get_vital_signs(client, mock_estimator):
    """Prueba que el endpoint /vital-signs devuelve datos correctos"""
    with patch('Code.servidor.estimator', mock_estimator):
        response = client.get('/vital-signs')
        assert response.status_code == 200
        data = response.get_json()
        assert data['heart_rate'] == 75
        assert data['respiratory_rate'] == 16

# def test_calibrate_endpoint(client):
#     """Prueba básica que el endpoint /calibrate responde"""
#     with patch('cv2.VideoCapture') as mock_video:
#         mock_video.return_value.read.return_value = (True, b'mock_frame')
#         response = client.get('/calibrate')
#         assert response.status_code == 200
#         assert b'frame' in response.data