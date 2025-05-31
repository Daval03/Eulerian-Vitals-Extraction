"""
Módulo Principal de Estimación de Signos Vitales

Implementa el sistema completo para estimación no invasiva de:
- Frecuencia cardíaca (pulsaciones por minuto)
- Frecuencia respiratoria (respiraciones por minuto)

Características principales:
- Pipeline completo: captura → detección facial → EVM → análisis
- Gestión thread-safe para operación en tiempo real
- Filtrado estadístico para resultados estables
- Manejo robusto de recursos (cámaras, memoria)

Componentes clave:
- VitalSignsEstimator: Clase principal del sistema
- Buffer circular para procesamiento por chunks
- Integración con FaceDetector y signal_processing
- Mecanismos de inicialización/limpieza seguros

Uso:
    estimator = VitalSignsEstimator()
    estimator.estimate_signs()  # Ejecutar en hilo separado
    vitals = estimator.get_latest_vital_signs()
"""

import cv2
import threading
import numpy as np
from collections import deque
from Code.config import (FRAME_CHUNK, ROI_PADDING, VIDEO_PATH)
from Code.preprocessing import preprocess_frame, postprocess_frame
from Code.face_detector import FaceDetector
from Code.signal_processing import process_buffer_evm

class VitalSignsEstimator:
    """
    Estimador de signos vitales desde video usando EVM.
    
    Pipeline: Video → Detección facial → Buffer → EVM → Filtrado estadístico → Resultados
    Thread-safe con gestión automática de recursos.
    """
    def __init__(self):
        self.cap = None
        self.face_detector = None
        self.frame_buffer = deque(maxlen=FRAME_CHUNK)
        self.estimation_lock = threading.Lock()
        self.should_stop = False
        self._is_initialized = False
        self.heart_rate_history = deque(maxlen=5)
        self.resp_rate_history = deque(maxlen=5) 
        self.latest_vital_signs = {
            'heart_rate': None,
            'respiratory_rate': None
        }

    def initialize(self):
        """Inicializa los recursos necesarios"""
        if not self._is_initialized:
            self.face_detector = FaceDetector()  # Recrear el detector de rostros
            self._is_initialized = True
    
    def cleanup(self):
        """Limpia todos los recursos de manera segura"""
        try:
            # Liberar la captura de video
            if self.cap and hasattr(self.cap, 'isOpened') and self.cap.isOpened():
                self.cap.release()
            self.cap = None
            
            # Liberar el face_detector de manera segura
            if hasattr(self.face_detector, 'close'):
                try:
                    # Verificar si el gráfico todavía existe antes de cerrar
                    if hasattr(self.face_detector, '_graph') and self.face_detector._graph is not None:
                        self.face_detector.close()
                except Exception as e:
                    print(f"Advertencia al cerrar face_detector: {str(e)}")
            
            # Resetear otros estados
            self.should_stop = False
            self.frame_buffer.clear()
            self._is_initialized = False
        
        except Exception as e:
            print(f"Error durante cleanup: {str(e)}")
    
    def _apply_statistical_filter(self, new_hr, new_rr):
        """Aplica filtro de mediana móvil a los nuevos valores"""
        if new_hr is not None:
            self.heart_rate_history.append(new_hr)
        if new_rr is not None:
            self.resp_rate_history.append(new_rr)
        
        # Calcular mediana de los últimos valores
        filtered_hr = np.median(list(self.heart_rate_history)) if self.heart_rate_history else None
        filtered_rr = np.median(list(self.resp_rate_history)) if self.resp_rate_history else None
        
        return filtered_hr, filtered_rr
    
    def stop_estimation(self):
        """Solicita la detención de la estimación"""
        self.should_stop = True
    
    def start_capture(self):
        """Inicia la captura de video, buscando un puerto de cámara disponible o usando uno manual."""
        self.cleanup()  # Limpiar antes de iniciar
        self.initialize()
        
        # Buscar un puerto de cámara disponible
        for port in range(3):  # Probar los puertos 0, 1, 2
            self.cap = cv2.VideoCapture(port)
            if self.cap.isOpened():
                print(f"Cámara encontrada en el puerto {port}.")
                break
        else:
            # Si no se encontró ninguna cámara, usar el VIDEO_PATH
            self.cap = cv2.VideoCapture(VIDEO_PATH)
            print(f"Usando VIDEO_PATH: {VIDEO_PATH}.")
        
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir el video/cámara.")
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Dimensiones del video: {frame_width}x{frame_height}")

    def start_capture(self):
        """Inicia la captura de video, buscando un puerto de cámara disponible o usando uno manual."""
        self.cleanup()  # Limpiar antes de iniciar
        self.initialize()
        # Si no se encontró ninguna cámara, usar el VIDEO_PATH
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        print(f"Usando VIDEO_PATH: {VIDEO_PATH}.")
        
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir el video/cámara.")
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Dimensiones del video: {frame_width}x{frame_height}")

    def estimate_signs(self):
        """Estima los signos vitales."""
        self.should_stop = False
        try:
            #Leer frames con la camara
            self.start_capture()
            while self.cap.isOpened() and not self.should_stop:
                ret, frame = self.cap.read()
                if not ret:
                    break
                #Pre-procesamiento
                process_frame = preprocess_frame(frame)
                #IA para la ROI
                roi = self.face_detector.detect_face(process_frame)
                if roi:
                    x, y, w, h = roi
                    # Aplicar padding
                    x = max(0, x - ROI_PADDING)
                    y = max(0, y - ROI_PADDING)
                    w = min(frame.shape[1] - x, w + 2 * ROI_PADDING)
                    h = min(frame.shape[0] - y, h + 2 * ROI_PADDING)
                    
                    process_frame = frame[y:y+h, x:x+w]
                else:
                    print("No se detectó un rostro en este frame.")
                    continue
                #Post-procesamiento
                process_frame = postprocess_frame(process_frame)
                #Buffer de datos
                self.frame_buffer.append(process_frame)
                if len(self.frame_buffer) == FRAME_CHUNK:
                    #EVM
                    heart_rate, resp_rate = process_buffer_evm(self.frame_buffer)
                    
                    filtered_hr, filtered_rr = self._apply_statistical_filter(heart_rate, resp_rate)

                    # Actualizar los últimos signos vitales
                    with self.estimation_lock:
                        self.latest_vital_signs = {
                            'heart_rate': filtered_hr,
                            'respiratory_rate': filtered_rr}
                    self.frame_buffer.clear()
        except Exception as e:
            print(f"Ocurrió un error: {e}")
            raise
        finally:
            self.cleanup()

    def get_latest_vital_signs(self):
        """Devuelve los últimos signos vitales estimados."""
        with self.estimation_lock:
            return self.latest_vital_signs.copy()
