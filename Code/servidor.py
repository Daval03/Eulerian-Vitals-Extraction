from flask import Flask, jsonify, Response
from flask_cors import CORS
from Code.vital_signs_estimator import VitalSignsEstimator
from Code.config import (VIDEO_PATH)
import threading
import cv2

app = Flask(__name__)
CORS(app)

# Instancia global del estimador
estimator = VitalSignsEstimator()
estimation_thread = None
is_running = False

@app.route('/start', methods=['GET'])
def start_estimation():
    global estimation_thread, is_running
    
    if is_running:
        return jsonify({"status": "Estimation already running"}), 400
    
    try:
        # Reinicializar el estimador
        estimator.initialize()
        is_running = True
        estimation_thread = threading.Thread(target=run_estimation, daemon=True)
        estimation_thread.start()
        return jsonify({"status": "Estimation started"}), 200
    except Exception as e:
        print(f"Error al iniciar estimación: {e}")
        return jsonify({"status": "Error starting estimation", "error": str(e)}), 500

@app.route('/stop', methods=['GET'])
def stop_estimation():
    global is_running, estimation_thread
    if not is_running:
        return jsonify({"status": "No estimation in progress"}), 400
    try:
        is_running = False
        # Detener el estimador "suavemente"
        if hasattr(estimator, 'stop_estimation'):
            estimator.stop_estimation()
        
        # Manejar el hilo
        current_thread = estimation_thread
        estimation_thread = None
        
        if current_thread:
            current_thread.join(timeout=1.0)
        
        # Limpieza silenciosa - ignoramos errores de cierre
        try:
            if hasattr(estimator, 'cleanup'):
                estimator.cleanup()
        except Exception as e:
            print(f"Info: Error no crítico durante limpieza: {str(e)}")
        
        return jsonify({"status": "Estimation stopped successfully"}), 200
    
    except Exception as e:
        print(f"Error al detener estimación: {str(e)}")
        return jsonify({"status": "Error stopping estimation", "error": str(e)}), 500

@app.route('/vital-signs', methods=['GET'])
def get_vital_signs():
    try:
        vital_signs = estimator.get_latest_vital_signs()
        if vital_signs['heart_rate'] is not None:
            return jsonify(vital_signs), 200
        else:
            return jsonify({"status": "No vital signs available yet"}), 404
    except Exception as e:
        return jsonify({"status": "Error getting vital signs", "error": str(e)}), 500

@app.route('/calibrate', methods=['GET'])
def calibrate_camera():
    """
    Endpoint to stream raw camera feed for calibration
    """
    def generate_frames():
        cap = cv2.VideoCapture(VIDEO_PATH)  # Use the same video path as in estimator
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                # Convert to bytes for streaming
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        finally:
            cap.release()
    
    return Response(generate_frames(), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def run_estimation():
    global is_running, estimation_thread
    try:
        estimator.estimate_signs()
    except Exception as e:
        print(f"Error en estimación: {e}")
    finally:
        is_running = False
        estimation_thread = None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
