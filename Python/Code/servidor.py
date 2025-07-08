"""
Flask-server for estimering av vitalparametere

Denne modulen implementerer en webserver som:
- Eksponerer REST-endepunkter for å kontrollere estimeringsprosessen
- Håndterer beregning av vitalparametere i bakgrunnen
- Tilbyr videostrøm for kalibrering

Hovedendepunkter:
- /start: Starter estimeringsprosessen
- /stop: Stopper prosessen
- /vital-signs: Returnerer de sist beregnede verdiene
- /calibrate: Videostrøm i sanntid

Hovedkomponenter:
- VitalSignsEstimator: Klasse som utfører beregningene (importert)
- Estimeringstråd: Utfører prosesseringen i bakgrunnen
- Synkroniseringsmekanismer for sikker tilstandskontroll

Bruk:
    python server.py
"""

from flask import Flask, jsonify, Response
from flask_cors import CORS
from Code.vital_signs_estimator import VitalSignsEstimator
from Code.config import (VIDEO_PATH)
import threading
import cv2
import socket
import time

app = Flask(__name__)
CORS(app)

# Global instans av estimatoren
estimator = VitalSignsEstimator()
estimation_thread = None
is_running = False

# UDP Discovery Konfigurasjon
DISCOVERY_PORT = 50001  # Port for discovery-meldinger
DISCOVERY_MESSAGE = b"VITAL_SIGN_SERVER_DISCOVERY"
BROADCAST_INTERVAL = 5  # Sekunder mellom hver broadcast

def get_local_ip():
    """Henter den lokale IP-adressen til serveren."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # trenger ikke å være nåbar
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1' # Fallback
    finally:
        s.close()
    return IP

def broadcast_presence():
    """Sender UDP broadcast-meldinger for serveroppdagelse."""
    server_ip = get_local_ip()
    # Bruker serverens faktiske port, som er 5000 for Flask-appen
    message = f"VITAL_SIGN_SERVER_INFO:{server_ip}:5000".encode('utf-8')

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    # Bruker 255.255.255.255 for en bredere broadcast, men dette kan være begrenset av nettverkskonfigurasjonen.
    # En spesifikk subnett-broadcast (f.eks. 192.168.1.255) kan være mer pålitelig i noen nettverk.
    broadcast_address = '<broadcast>' # Bruker standard broadcast-adresse

    print(f"Starter serveroppdagelses-broadcast på port {DISCOVERY_PORT}...")
    print(f"Serverinformasjon som kringkastes: {message.decode()}")

    while True:
        try:
            sock.sendto(message, (broadcast_address, DISCOVERY_PORT))
            # print(f"Sendte discovery broadcast: {message.decode()}") # Kan være for støyende
        except Exception as e:
            print(f"Feil under sending av discovery broadcast: {e}")
        time.sleep(BROADCAST_INTERVAL)

@app.route('/start', methods=['GET'])
def start_estimation():
    """
    Endepunkt for å starte estimeringsprosessen.
    
    Returnerer:
        JSON med status for operasjonen:
        - 200: Estimering startet korrekt
        - 400: Estimering pågår allerede
        - 500: Feil ved oppstart
    """
    global estimation_thread, is_running
    
    if is_running:
        return jsonify({"status": "Estimering pågår allerede"}), 400
    
    try:
        # Reinitialiser estimatoren
        estimator.initialize()
        is_running = True
        estimation_thread = threading.Thread(target=run_estimation, daemon=True)
        estimation_thread.start()
        return jsonify({"status": "Estimering startet"}), 200
    except Exception as e:
        print(f"Feil ved start av estimering: {e}")
        return jsonify({"status": "Feil ved start av estimering", "error": str(e)}), 500

@app.route('/stop', methods=['GET'])
def stop_estimation():
    global is_running, estimation_thread
    if not is_running:
        return jsonify({"status": "Ingen estimering pågår"}), 400
    try:
        is_running = False
        # Stopp estimatoren "mykt"
        if hasattr(estimator, 'stop_estimation'):
            estimator.stop_estimation()
        
        # Håndter tråden
        current_thread = estimation_thread
        estimation_thread = None
        
        if current_thread:
            current_thread.join(timeout=1.0)
        
        # Stille opprydding - ignorer lukkefeil
        try:
            if hasattr(estimator, 'cleanup'):
                estimator.cleanup()
        except Exception as e:
            print(f"Info: Ikke-kritisk feil under opprydding: {str(e)}")
        
        return jsonify({"status": "Estimering stoppet vellykket"}), 200
    
    except Exception as e:
        print(f"Feil ved stopping av estimering: {str(e)}")
        return jsonify({"status": "Feil ved stopping av estimering", "error": str(e)}), 500

@app.route('/vital-signs', methods=['GET'])
def get_vital_signs():
    try:
        vital_signs = estimator.get_latest_vital_signs()
        if vital_signs['heart_rate'] is not None:
            return jsonify(vital_signs), 200
        else:
            return jsonify({"status": "Ingen vitalparametere tilgjengelig ennå"}), 404
    except Exception as e:
        return jsonify({"status": "Feil ved henting av vitalparametere", "error": str(e)}), 500

@app.route('/calibrate', methods=['GET'])
def calibrate_camera():
    """
    Endepunkt for å strømme rå kamerastrøm for kalibrering
    """
    def generate_frames():
        cap = cv2.VideoCapture(VIDEO_PATH)  # Bruk samme videobane som i estimatoren
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Kod ramme som JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    continue
                
                # Konverter til bytes for strømming
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
        print(f"Feil i estimering: {e}")
    finally:
        is_running = False
        estimation_thread = None

if __name__ == '__main__':
    # Start broadcast-tråden
    broadcast_thread = threading.Thread(target=broadcast_presence, daemon=True)
    broadcast_thread.start()

    server_ip = get_local_ip()
    print(f"Flask server kjører på http://{server_ip}:5000")
    print(f"Trykk CTRL+C for å avslutte.")
    app.run(debug=True, host='0.0.0.0', port=5000)
