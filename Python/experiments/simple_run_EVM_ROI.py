import cv2
import os
import sys
import time
import numpy as np
from collections import deque

# Configurar path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from src.face_detector.manager import FaceDetector
from src.evm.evm_manager import process_video_evm_vital_signs
from src.utils.ground_truth_handler import setup_ground_truth
from src.config import TARGET_ROI_SIZE

NUM = 49
BUFFER_SIZE = 200
gt_handler, VIDEO_PATH = setup_ground_truth(NUM, "/mnt/c/Self-Study/TFG/dataset_2", BUFFER_SIZE)

def benchmark_completo(video_source, buffer_size=BUFFER_SIZE):
    """
    Benchmark completo que mide:
    1. Detección facial (ROI)
    2. Procesamiento EVM
    3. Ciclo completo end-to-end
    4. Precisión de predicciones
    """
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
    rr_history = deque(maxlen=10)
    
    # Métricas de tiempo
    detection_times = []  # Tiempo de detección facial por frame
    processing_times = []  # Tiempo de procesamiento EVM por chunk
    end_to_end_times = []  # Tiempo total por chunk (detección + EVM)
    
    # Métricas de precisión
    hr_errors = []
    hr_absolute_errors = []
    hr_predictions = []
    hr_ground_truths = []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print("=" * 80)
    print("BENCHMARK COMPLETO - DETECCIÓN + EVM + END-TO-END")
    print("=" * 80)
    print(f"Video: {video_source}")
    print(f"Frames totales: {total_frames}, FPS original: {fps:.1f}")
    print(f"Tamaño de buffer: {buffer_size} frames")
    print("=" * 80)
    
    try:
        chunk_detection_time = 0  # Acumulador para tiempo de detección del chunk actual
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # MÉTRICA 1: Tiempo de detección facial
            detection_start = time.time()
            roi = face_detector.detect_face(frame)
            detection_end = time.time()
            detection_time = detection_end - detection_start
            detection_times.append(detection_time)
            chunk_detection_time += detection_time
            
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
                    
                    # MÉTRICA 2: Tiempo de procesamiento EVM
                    proc_start = time.time()
                    results = process_video_evm_vital_signs(frame_buffer)
                    proc_end = time.time()
                    processing_time = proc_end - proc_start
                    processing_times.append(processing_time)
                    
                    # MÉTRICA 3: Tiempo end-to-end (detección + EVM)
                    total_chunk_time = chunk_detection_time + processing_time
                    end_to_end_times.append(total_chunk_time)
                    
                    hr = results['heart_rate']
                    rr = results['respiratory_rate']
                    
                    hr_history.append(hr)
                    rr_history.append(rr)
                    
                    filtered_hr = np.median(list(hr_history)) if hr_history else None
                    filtered_rr = np.median(list(rr_history)) if rr_history else None
                    
                    # MÉTRICA 4: Precisión
                    if filtered_hr is not None:
                        error = filtered_hr - true_hr
                        abs_error = abs(error)
                        hr_errors.append(error)
                        hr_absolute_errors.append(abs_error)
                        hr_predictions.append(filtered_hr)
                        hr_ground_truths.append(true_hr)

                    # Mostrar progreso
                    avg_detection_time = chunk_detection_time / buffer_size
                    evm_time_per_frame = processing_time / buffer_size
                    total_time_per_frame = total_chunk_time / buffer_size
                    current_fps = 1.0 / total_time_per_frame if total_time_per_frame > 0 else 0
                    
                    print(f"\n{'─'*80}")
                    print(f"Chunk {measurement_count} (frames {frame_count-buffer_size+1}-{frame_count}):")
                    print(f"  Detección ROI:  {chunk_detection_time:.2f}s | {avg_detection_time*1000:.1f}ms/frame")
                    print(f"  Procesamiento EVM: {processing_time:.2f}s | {evm_time_per_frame*1000:.1f}ms/frame")
                    print(f"  Total End-to-End:  {total_chunk_time:.2f}s | {total_time_per_frame*1000:.1f}ms/frame | {current_fps:.1f} FPS")
                    print(f"  HR: {filtered_hr:.1f} BPM | GT: {true_hr:.1f} BPM | Error: {error:+.1f} BPM")
                    if filtered_rr:
                        print(f"  RR: {filtered_rr:.1f} BPM")
                    
                    # Resetear acumuladores
                    frame_buffer.clear()
                    chunk_detection_time = 0
                    
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
        
        # ============================================================================
        # REPORTE FINAL DE MÉTRICAS
        # ============================================================================
        
        print(f"\n{'='*80}")
        print("RESULTADOS DEL BENCHMARK COMPLETO")
        print(f"{'='*80}")
        
        # SECCIÓN 1: MÉTRICAS DE DETECCIÓN FACIAL (ROI)
        print(f"\n{'─'*80}")
        print("1. MÉTRICAS DE DETECCIÓN FACIAL (ROI)")
        print(f"{'─'*80}")
        
        if detection_times:
            total_detection_time = sum(detection_times)
            avg_detection_time = total_detection_time / len(detection_times)
            max_detection_time = max(detection_times)
            min_detection_time = min(detection_times)
            detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
            
            print(f"Frames procesados: {len(detection_times)}")
            print(f"Tiempo total detección: {total_detection_time:.2f} segundos")
            print(f"Tiempo promedio por frame: {avg_detection_time*1000:.1f} ms")
            print(f"Tiempo máximo: {max_detection_time*1000:.1f} ms")
            print(f"Tiempo mínimo: {min_detection_time*1000:.1f} ms")
            print(f"FPS de detección: {detection_fps:.1f}")
            print(f"Velocidad relativa: {detection_fps/fps*100:.1f}% del FPS original")
        
        # SECCIÓN 2: MÉTRICAS DE PROCESAMIENTO EVM
        print(f"\n{'─'*80}")
        print("2. MÉTRICAS DE PROCESAMIENTO EVM")
        print(f"{'─'*80}")
        
        if processing_times:
            total_evm_time = sum(processing_times)
            avg_evm_time = total_evm_time / len(processing_times)
            max_evm_time = max(processing_times)
            min_evm_time = min(processing_times)
            
            avg_evm_time_per_frame = avg_evm_time / buffer_size
            evm_fps = 1.0 / avg_evm_time_per_frame if avg_evm_time_per_frame > 0 else 0
            
            print(f"Chunks procesados: {len(processing_times)}")
            print(f"Frames procesados en EVM: {len(processing_times) * buffer_size}")
            print(f"Tiempo total procesamiento EVM: {total_evm_time:.2f} segundos")
            print(f"Tiempo promedio por chunk: {avg_evm_time:.2f} segundos")
            print(f"Tiempo promedio por frame: {avg_evm_time_per_frame*1000:.1f} ms")
            print(f"Tiempo máximo por chunk: {max_evm_time:.2f} segundos")
            print(f"Tiempo mínimo por chunk: {min_evm_time:.2f} segundos")
            print(f"FPS de procesamiento EVM: {evm_fps:.1f}")
            print(f"Velocidad relativa: {evm_fps/fps*100:.1f}% del FPS original")
        
        # SECCIÓN 3: MÉTRICAS END-TO-END (CICLO COMPLETO)
        print(f"\n{'─'*80}")
        print("3. MÉTRICAS END-TO-END (DETECCIÓN + EVM)")
        print(f"{'─'*80}")
        
        if end_to_end_times:
            total_e2e_time = sum(end_to_end_times)
            avg_e2e_time = total_e2e_time / len(end_to_end_times)
            max_e2e_time = max(end_to_end_times)
            min_e2e_time = min(end_to_end_times)
            
            avg_e2e_time_per_frame = avg_e2e_time / buffer_size
            e2e_fps = 1.0 / avg_e2e_time_per_frame if avg_e2e_time_per_frame > 0 else 0
            
            print(f"Chunks procesados: {len(end_to_end_times)}")
            print(f"Tiempo total end-to-end: {total_e2e_time:.2f} segundos")
            print(f"Tiempo promedio por chunk: {avg_e2e_time:.2f} segundos")
            print(f"Tiempo promedio por frame: {avg_e2e_time_per_frame*1000:.1f} ms")
            print(f"Tiempo máximo por chunk: {max_e2e_time:.2f} segundos")
            print(f"Tiempo mínimo por chunk: {min_e2e_time:.2f} segundos")
            print(f"FPS end-to-end: {e2e_fps:.1f}")
            print(f"Velocidad relativa: {e2e_fps/fps*100:.1f}% del FPS original")
            
            # Desglose de tiempos
            if detection_times and processing_times:
                avg_detection_percentage = (sum(detection_times[:len(processing_times)*buffer_size]) / 
                                           (len(processing_times)*buffer_size)) / avg_e2e_time_per_frame * 100
                avg_evm_percentage = avg_evm_time_per_frame / avg_e2e_time_per_frame * 100
                
                print(f"\nDesglose de tiempo promedio:")
                print(f"  Detección ROI: {avg_detection_percentage:.1f}%")
                print(f"  Procesamiento EVM: {avg_evm_percentage:.1f}%")
        
        # SECCIÓN 4: MÉTRICAS DE PRECISIÓN (HEART RATE)
        print(f"\n{'─'*80}")
        print("4. MÉTRICAS DE PRECISIÓN (HEART RATE)")
        print(f"{'─'*80}")
        
        if hr_absolute_errors:
            mae = np.mean(hr_absolute_errors)
            me = np.mean(hr_errors)
            rmse = np.sqrt(np.mean(np.array(hr_errors)**2))
            std_error = np.std(hr_errors)
            max_error = max(hr_absolute_errors)
            min_error = min(hr_absolute_errors)
            
            # Porcentajes dentro de rangos
            within_5 = sum(1 for e in hr_absolute_errors if e <= 5) / len(hr_absolute_errors) * 100
            within_10 = sum(1 for e in hr_absolute_errors if e <= 10) / len(hr_absolute_errors) * 100
            within_15 = sum(1 for e in hr_absolute_errors if e <= 15) / len(hr_absolute_errors) * 100
            
            # Correlación
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
        
        # SECCIÓN 5: RESUMEN EJECUTIVO
        print(f"\n{'='*80}")
        print("RESUMEN EJECUTIVO")
        print(f"{'='*80}")
        
        if detection_times and processing_times and end_to_end_times and hr_absolute_errors:
            print(f"\nRendimiento:")
            print(f"  FPS End-to-End: {e2e_fps:.1f} (Detección: {detection_fps:.1f}, EVM: {evm_fps:.1f})")
            print(f"  Tiempo por medición: {avg_e2e_time:.2f}s ({buffer_size} frames)")
            
            print(f"\nPrecisión:")
            print(f"  MAE: {mae:.2f} BPM | RMSE: {rmse:.2f} BPM")
            print(f"  {within_10:.1f}% de predicciones dentro de ±10 BPM")
            if correlation is not None:
                print(f"  Correlación: {correlation:.3f}")
        
        print(f"\n{'='*80}\n")

if __name__ == "__main__":
    benchmark_completo(VIDEO_PATH)