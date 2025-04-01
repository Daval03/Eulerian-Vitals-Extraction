import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from Code.config import (FRAME_CHUNK, ROI_PADDING, VIDEO_PATH)
from Code.preprocessing import preprocess_frame, postprocess_frame
from Code.signal_processing import process_buffer_evm
from Code.face_detector import FaceDetector

# Cargar los datos ground truth
gt_filename = 'dataset_real/ground_truth_3.txt'
VIDEO_PATH = "dataset_real/vid_3.mp4"
gt_data = np.loadtxt(gt_filename)
gt_hr = gt_data[1, :]  # Segunda fila: frecuencia cardíaca (HR)
gt_chunk_hrs = [np.mean(gt_hr[i*FRAME_CHUNK:(i+1)*FRAME_CHUNK]) for i in range(len(gt_hr)//FRAME_CHUNK)]

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
                gt_hr = gt_chunk_hrs[chunk_counter]
                error = abs(heart_rate - gt_hr)
                
                print(f"Chunk {chunk_counter + 1}:")
                print(f"  - HR estimada: {heart_rate:.2f} bpm")
                print(f"  - HR real: {gt_hr:.2f} bpm")
                print(f"  - Error absoluto: {error:.2f} bpm")
                print("-" * 40)
                
                estimated_hrs.append(heart_rate)
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