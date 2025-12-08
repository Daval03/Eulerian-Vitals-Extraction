import cv2
import os
import sys
import time
import numpy as np

# Configurar path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from collections import deque
from src.face_detector.manager import FaceDetector
from src.evm.evm_manager import process_video_evm_vital_signs
from src.utils.ground_truth_handler import setup_ground_truth
from src.config import TARGET_ROI_SIZE

NUM = 1
BUFFER_SIZE = 200
gt_handler, VIDEO_PATH = setup_ground_truth(NUM, "/mnt/c/Self-Study/TFG/dataset_2", BUFFER_SIZE)

def process_video_with_evm(video_source, buffer_size=BUFFER_SIZE):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir {video_source}")
        return
    
    face_detector = FaceDetector(model_type='mediapipe')
    frame_count = 0
    measurement_count = 0
    frame_buffer = []
    
    # Historial de mediciones
    hr_history = deque(maxlen=10)
    #rr_history = deque(maxlen=10)
    
    # Estadísticas de rendimiento (estilo ROI)
    processing_times = []  # Tiempo por chunk
    
    # Estadísticas de precisión
    hr_errors = []
    hr_absolute_errors = []
    hr_predictions = []
    hr_ground_truths = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 70)
    print("PROCESAMIENTO CON EULERIAN VIDEO MAGNIFICATION (EVM)")
    print("=" * 70)
    print(f"Video: {video_source}")
    print(f"Frames totales: {total_frames}, FPS original: {fps:.1f}")
    print(f"Tamaño de buffer: {buffer_size} frames")
    print("=" * 70)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            roi = face_detector.detect_face(frame)
            
            if roi:                
                x, y, w, h = roi
                roi_frame = frame[y:y+h, x:x+w]
                roi_frame = cv2.resize(roi_frame, TARGET_ROI_SIZE)
                frame_buffer.append(roi_frame)

                if len(frame_buffer) >= buffer_size:
                    
                    true_hr = gt_handler.get_hr_for_chunk(measurement_count)
                    if true_hr is None:
                        print(f"⚠️  No hay más ground truth para chunk {measurement_count}")
                        break

                    measurement_count += 1
                    
                    # Medir tiempo de procesamiento EVM
                    proc_start = time.time()
                    result = process_video_evm_vital_signs(frame_buffer)
                    proc_end = time.time()
                    processing_time = proc_end - proc_start
                    processing_times.append(processing_time)
                    
                    hr = result["heart_rate"]

                    if hr is not None:
                        hr_history.append(hr)

                    filtered_hr = np.median(list(hr_history)) if hr_history else None
                        
                    if filtered_hr is not None:
                        error = filtered_hr - true_hr
                        abs_error = abs(error)
                        hr_errors.append(error)
                        hr_absolute_errors.append(abs_error)
                        hr_predictions.append(filtered_hr)
                        hr_ground_truths.append(true_hr)

                    # Mostrar progreso (estilo ROI)
                    time_per_frame = processing_time / buffer_size
                    current_fps = 1.0 / time_per_frame if time_per_frame > 0 else 0
                    
                    print(f"\nChunk {measurement_count} (frames {frame_count-buffer_size+1}-{frame_count}):")
                    print(f"  Tiempo chunk: {processing_time:.2f}s | Tiempo/frame: {time_per_frame*1000:.1f}ms | FPS: {current_fps:.1f}")
                    print(f"  HR estimada: {filtered_hr:.1f} BPM | GT: {true_hr:.1f} BPM | Error: {error:+.1f} BPM")

                    # Limpiar buffer para siguiente chunk
                    frame_buffer.clear()
                    
            else:
                if frame_count % 200 == 0:
                    print(f"Frame {frame_count}: ⚠️  No se detectó rostro")
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Procesamiento detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        face_detector.close()
        
        # ESTADÍSTICAS FINALES (ESTILO ROI)
        print(f"\n{'='*70}")
        print("RESULTADOS DEL BENCHMARK DE PROCESAMIENTO EVM")
        print(f"{'='*70}")
        
        if processing_times:
            total_proc_time = sum(processing_times)
            avg_proc_time = total_proc_time / len(processing_times)
            max_proc_time = max(processing_times)
            min_proc_time = min(processing_times)
            
            # Calcular métricas por frame (estilo ROI)
            avg_time_per_frame = avg_proc_time / buffer_size
            max_time_per_frame = max_proc_time / buffer_size
            min_time_per_frame = min_proc_time / buffer_size
            avg_fps = 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0
            
            frames_procesados = len(processing_times) * buffer_size
            
            print(f"Frames procesados: {frames_procesados}")
            print(f"Chunks procesados: {len(processing_times)}")
            print(f"Tiempo total procesamiento EVM: {total_proc_time:.2f} segundos")
            print(f"Tiempo promedio por frame: {avg_time_per_frame*1000:.1f} ms")
            print(f"Tiempo máximo por frame: {max_time_per_frame*1000:.1f} ms")
            print(f"Tiempo mínimo por frame: {min_time_per_frame*1000:.1f} ms")
            print(f"FPS promedio: {avg_fps:.1f}")
            print(f"Velocidad relativa: {avg_fps/fps*100:.1f}% del FPS original")
        
        # ESTADÍSTICAS DE PRECISIÓN
        print(f"\n{'='*70}")
        print("ESTADÍSTICAS DE PRECISIÓN (HEART RATE)")
        print(f"{'='*70}")
        
        if hr_absolute_errors:
            mae = np.mean(hr_absolute_errors)
            me = np.mean(hr_errors)
            rmse = np.sqrt(np.mean(np.array(hr_errors)**2))
            std_error = np.std(hr_errors)
            max_error = max(hr_absolute_errors)
            min_error = min(hr_absolute_errors)
            
            # Calcular porcentaje de predicciones dentro de rangos
            within_5 = sum(1 for e in hr_absolute_errors if e <= 5) / len(hr_absolute_errors) * 100
            within_10 = sum(1 for e in hr_absolute_errors if e <= 10) / len(hr_absolute_errors) * 100
            within_15 = sum(1 for e in hr_absolute_errors if e <= 15) / len(hr_absolute_errors) * 100
            
            # Calcular correlación
            if len(hr_predictions) > 1:
                correlation = np.corrcoef(hr_predictions, hr_ground_truths)[0, 1]
            else:
                correlation = None
            
            print(f"Número de mediciones: {len(hr_absolute_errors)}")
            print(f"\nMétricas de Error:")
            print(f"  Mean Absolute Error (MAE): {mae:.2f} BPM")
            print(f"  Mean Error (ME): {me:+.2f} BPM")
            print(f"  Root Mean Square Error (RMSE): {rmse:.2f} BPM")
            print(f"  Desviación estándar del error: {std_error:.2f} BPM")
            print(f"  Error máximo: {max_error:.2f} BPM")
            print(f"  Error mínimo: {min_error:.2f} BPM")
            
            print(f"\nDistribución de Errores:")
            print(f"  Dentro de ±5 BPM: {within_5:.1f}%")
            print(f"  Dentro de ±10 BPM: {within_10:.1f}%")
            print(f"  Dentro de ±15 BPM: {within_15:.1f}%")
            
            if correlation is not None:
                print(f"\nCorrelación con Ground Truth: {correlation:.3f}")
            
            print(f"\nRango de Valores:")
            print(f"  HR predicho: {min(hr_predictions):.1f} - {max(hr_predictions):.1f} BPM")
            print(f"  HR ground truth: {min(hr_ground_truths):.1f} - {max(hr_ground_truths):.1f} BPM")
        
        print(f"{'='*70}\n")

if __name__ == "__main__":
    process_video_with_evm(VIDEO_PATH)
