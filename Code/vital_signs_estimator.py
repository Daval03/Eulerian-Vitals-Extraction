import cv2
import threading
from collections import deque
from Code.config import (FRAME_CHUNK, ROI_PADDING, VIDEO_PATH)
from Code.preprocessing import preprocess_frame, postprocess_frame
from Code.face_detector import FaceDetector
from Code.signal_processing import process_buffer_evm

class VitalSignsEstimator:
    def __init__(self):
        self.cap = None
        self.face_detector = None
        #self.face_detector = FaceDetector()
        self.frame_buffer = deque(maxlen=FRAME_CHUNK)
        self.estimation_lock = threading.Lock()
        self.should_stop = False
        self._is_initialized = False
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

    def stop_estimation(self):
        """Solicita la detención de la estimación"""
        self.should_stop = True

    def start_capture(self):
        """Inicia la captura de video."""
        self.cleanup()  # Limpiar antes de iniciar
        self.initialize()
        
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        if not self.cap.isOpened():
            raise ValueError("No se pudo abrir el video/cámara.")
        
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Dimensiones del video: {frame_width}x{frame_height}")
    
    # def start_capture(self):
    #     """Inicia la captura de video, buscando un puerto de cámara disponible o usando uno manual."""
    #     self.cleanup()  # Limpiar antes de iniciar
    #     self.initialize()
        
    #     # Buscar un puerto de cámara disponible
    #     for port in range(3):  # Probar los puertos 0, 1, 2
    #         self.cap = cv2.VideoCapture(port)
    #         if self.cap.isOpened():
    #             print(f"Cámara encontrada en el puerto {port}.")
    #             break
    #     else:
    #         # Si no se encontró ninguna cámara, usar el VIDEO_PATH
    #         self.cap = cv2.VideoCapture(VIDEO_PATH)
    #         print(f"Usando VIDEO_PATH: {VIDEO_PATH}.")
        
    #     if not self.cap.isOpened():
    #         raise ValueError("No se pudo abrir el video/cámara.")
        
    #     frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    #     frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    #     print(f"Dimensiones del video: {frame_width}x{frame_height}")

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
                    # Actualizar los últimos signos vitales
                    with self.estimation_lock:
                        self.latest_vital_signs = {
                            'heart_rate': heart_rate,
                            'respiratory_rate': resp_rate}
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
