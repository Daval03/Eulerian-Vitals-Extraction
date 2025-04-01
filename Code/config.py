import os
from collections import deque

# Configuración de TensorFlow
os.environ["TF_DISABLE_XNNPACK"] = "0"

# Parámetros globales
WINDOW_SIZE = 5
HEART_RATE_WINDOW = deque(maxlen=WINDOW_SIZE)
RESP_RATE_WINDOW = deque(maxlen=WINDOW_SIZE)

# Parámetros de procesamiento
ALPHA = 75
LOW_HEART = 0.83
HIGH_HEART = 3.0
LOW_RESP = 0.18
HIGH_RESP = 0.5
LEVELS = 1
FPS = 30
FRAME_CHUNK = 200
ROI_PADDING = 10

# Configuración de captura de video
VIDEO_PATH = 0  # Cámara por defecto, cambia a ruta de video si es necesario
FRAME_BUFFER = deque(maxlen=FRAME_CHUNK)

# Umbral para cambios en ROI
ROI_CHANGE_THRESHOLD = 20