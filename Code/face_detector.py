import mediapipe as mp
from collections import deque
from Code.config import ROI_CHANGE_THRESHOLD

class FaceDetector:
    def __init__(self, model_selection=0, min_detection_confidence=0.5):
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=model_selection, 
            min_detection_confidence=min_detection_confidence
        )
        self.roi_history = deque(maxlen=5)
        self.stabilized_roi = None

    def stabilize_roi(self, new_roi):
        if new_roi is None:
            return self.stabilized_roi

        self.roi_history.append(new_roi)

        if len(self.roi_history) < 3:
            return new_roi

        # Calcular media ponderada de coordenadas
        weights = [0.1, 0.15, 0.2, 0.25, 0.3][:len(self.roi_history)]
        weights = [w/sum(weights) for w in weights]

        x_stable = int(sum(x * w for x, w in zip([r[0] for r in self.roi_history], weights)))
        y_stable = int(sum(y * w for y, w in zip([r[1] for r in self.roi_history], weights)))
        w_stable = int(sum(w * weight for w, weight in zip([r[2] for r in self.roi_history], weights)))
        h_stable = int(sum(h * weight for h, weight in zip([r[3] for r in self.roi_history], weights)))

        self.stabilized_roi = (x_stable, y_stable, w_stable, h_stable)
        return self.stabilized_roi

    def is_significant_change(self, old_roi, new_roi):
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
        self.face_detection.close()