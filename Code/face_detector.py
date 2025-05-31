"""
Módulo de Detección Facial con Estabilización de ROI

Este módulo implementa detección de rostros usando MediaPipe con técnicas avanzadas de:

    Estabilización temporal de la región de interés (ROI)

    Filtrado de movimientos bruscos

    Manejo robusto de bordes de imagen

Funcionalidades clave:

    Detección facial precisa con MediaPipe Face Detection

    Estabilización de ROI usando promedio ponderado temporal

    Detección de cambios significativos en la posición

    Validación de ROI dentro de los límites de la imagen

Componentes principales:

    FaceDetector: Clase principal que encapsula toda la funcionalidad

    Sistema de historial de ROI para estabilización

    Mecanismos de ponderación temporal (weights)

    Umbrales configurables para detección de cambios

Uso:
detector = FaceDetector()
roi = detector.detect_face(frame_video)
if roi:
x, y, w, h = roi # ROI estabilizada
"""

import mediapipe as mp
from collections import deque
from Code.config import ROI_CHANGE_THRESHOLD, ROI_WEIGHTS

class FaceDetector:
    """
    Detector de rostros con estabilización temporal de ROI.
    
    Usa MediaPipe para detección y aplica filtros para reducir movimientos bruscos
    de la región de interés, mejorando la estabilidad para análisis de signos vitales.
    """
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        """
        Inicializa el detector con MediaPipe.
        
        Args:
            model_selection: 0=cerca (2m), 1=lejos (5m)
            min_detection_confidence: Umbral de confianza mínima
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection, 
            min_detection_confidence=min_detection_confidence
        )
        self.roi_history = deque(maxlen=5)
        self.stabilized_roi = None

    def stabilize_roi(self, new_roi):
        """
        Estabiliza ROI usando media ponderada temporal.
        
        Args:
            new_roi: Nueva ROI detectada (x, y, w, h)
        Returns:
            tuple: ROI estabilizada o None
        """
        if new_roi is None:
            return self.stabilized_roi

        self.roi_history.append(new_roi)

        if len(self.roi_history) < 3:
            return new_roi

        # Calcular media ponderada de coordenadas
        weights = ROI_WEIGHTS[:len(self.roi_history)]
        weights = [w/sum(weights) for w in weights]

        x_stable = int(sum(x * w for x, w in zip([r[0] for r in self.roi_history], weights)))
        y_stable = int(sum(y * w for y, w in zip([r[1] for r in self.roi_history], weights)))
        w_stable = int(sum(w * weight for w, weight in zip([r[2] for r in self.roi_history], weights)))
        h_stable = int(sum(h * weight for h, weight in zip([r[3] for r in self.roi_history], weights)))

        self.stabilized_roi = (x_stable, y_stable, w_stable, h_stable)
        return self.stabilized_roi

    def is_significant_change(self, old_roi, new_roi):
        """
        Determina si el cambio entre ROIs es significativo.
        
        Args:
            old_roi: ROI anterior (x, y, w, h)
            new_roi: ROI nueva (x, y, w, h)
        Returns:
            bool: True si el cambio supera el umbral
        """
        if old_roi is None or new_roi is None:
            return True

        x1, y1, w1, h1 = old_roi
        x2, y2, w2, h2 = new_roi

        dx = abs(x1 - x2)
        dy = abs(y1 - y2)
        dw = abs(w1 - w2)
        dh = abs(h1 - h2)

        return dx > ROI_CHANGE_THRESHOLD or dy > ROI_CHANGE_THRESHOLD or \
               dw > ROI_CHANGE_THRESHOLD or dh > ROI_CHANGE_THRESHOLD

    def detect_face(self, frame):
        """
        Detecta rostro y retorna ROI estabilizada.
        
        Args:
            frame: Frame de video (numpy array)
        Returns:
            tuple: (x, y, w, h) de la ROI estabilizada o None
        """
        results = self.face_detection.process(frame)
        if results.detections:
            bbox = results.detections[0].location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x_min = int(bbox.xmin * iw)
            y_min = int(bbox.ymin * ih)
            width = int(bbox.width * iw)
            height = int(bbox.height * ih)
            current_roi = (x_min, y_min, width, height)

            # Solo actualizar la ROI si hay un cambio significativo
            stable_roi = self.stabilize_roi(current_roi)

            if stable_roi:
                sx, sy, sw, sh = stable_roi
                # Asegurar que la ROI estabilizada no se salga del frame
                sx = max(0, sx)
                sy = max(0, sy)
                sw = min(frame.shape[1] - sx, sw)
                sh = min(frame.shape[0] - sy, sh)

                return (sx, sy, sw, sh)

        return None

    def close(self):
        """Libera recursos de MediaPipe."""
        self.face_detection.close()