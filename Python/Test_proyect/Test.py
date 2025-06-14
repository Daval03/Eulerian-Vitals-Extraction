import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from Code.config import (FRAME_CHUNK, ROI_PADDING, VIDEO_PATH)
from Code.preprocessing import preprocess_frame, postprocess_frame
from Code.signal_processing import process_buffer_evm
from Code.face_detector import FaceDetector

# Cargar los datos ground truth
gt_filename = 'dataset_real/ground_truth_4.txt'
VIDEO_PATH = "dataset_real/vid_4.mp4"

heart_rate_history = deque(maxlen=5)
resp_rate_history = deque(maxlen=5) 



gt_data = np.loadtxt(gt_filename)
gt_hr = gt_data[1, :]  # Segunda fila: frecuencia cardíaca (HR)
gt_chunk_hrs = [np.mean(gt_hr[i*FRAME_CHUNK:(i+1)*FRAME_CHUNK]) for i in range(len(gt_hr)//FRAME_CHUNK)]

def _apply_statistical_filter(new_hr, new_rr):
    """Aplica filtro de mediana móvil a los nuevos valores"""
    if new_hr is not None:
        heart_rate_history.append(new_hr)
    if new_rr is not None:
        resp_rate_history.append(new_rr)
    
    # Calcular mediana de los últimos valores
    filtered_hr = np.median(list(heart_rate_history)) if heart_rate_history else None
    filtered_rr = np.median(list(resp_rate_history)) if resp_rate_history else None
    
    return filtered_hr, filtered_rr

def main():
    frame_buffer = deque(maxlen=FRAME_CHUNK)
    face_detector = FaceDetector()
    chunk_counter = 0
    estimated_hrs = []
    ground_truth_hrs = []
    
    try:
        cap = cv2.VideoCapture(VIDEO_PATH)
        while cap.isOpened() and chunk_counter < len(gt_chunk_hrs):
            ret, frame = cap.read()
            if not ret:
                break
                
            process_frame = preprocess_frame(frame)
            roi = face_detector.detect_face(process_frame)
            
            if roi:
                x, y, w, h = roi
                x = max(0, x - ROI_PADDING)
                y = max(0, y - ROI_PADDING)
                w = min(frame.shape[1] - x, w + 2 * ROI_PADDING)
                h = min(frame.shape[0] - y, h + 2 * ROI_PADDING)
                process_frame = frame[y:y+h, x:x+w]
            else:
                print("No se detectó un rostro en este frame.")
                continue
                
            process_frame = postprocess_frame(process_frame)
            frame_buffer.append(process_frame)
            
            if len(frame_buffer) == FRAME_CHUNK:
                heart_rate, resp_rate = process_buffer_evm(frame_buffer)
                filtered_hr, filtered_rr = _apply_statistical_filter(heart_rate, resp_rate)
                print("PR: ",filtered_hr, "RR: ",filtered_rr)
                gt_hr = gt_chunk_hrs[chunk_counter]
                error = abs(filtered_hr - gt_hr)
                
                print(f"Chunk {chunk_counter + 1}:")
                print(f"  - HR estimada: {filtered_hr:.2f} bpm")
                print(f"  - HR real: {gt_hr:.2f} bpm")
                print(f"  - Error absoluto: {error:.2f} bpm")
                print("-" * 40)
                
                estimated_hrs.append(filtered_hr)
                ground_truth_hrs.append(gt_hr)
                
                frame_buffer.clear()
                chunk_counter += 1
                
        # Graficar los resultados
        plt.figure(figsize=(12, 6))
        chunks = range(1, len(estimated_hrs)+1)
        plt.plot(chunks, estimated_hrs, 'b-o', label='Estimado (EVM)')
        plt.plot(chunks, ground_truth_hrs, 'r--s', label='Ground Truth')
        plt.xlabel('Número de Chunk')
        plt.ylabel('Frecuencia Cardíaca (bpm)')
        plt.title('Comparación: Frecuencia Cardíaca Estimada vs Ground Truth')
        plt.legend()
        plt.grid(True)
        plt.xticks(chunks)
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        print(f"Ocurrió un error: {e}")
        raise

if __name__ == "__main__":
    main()