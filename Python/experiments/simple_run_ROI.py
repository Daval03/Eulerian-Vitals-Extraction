import cv2
import os
import sys
import time

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from src.face_detector.manager import FaceDetector

VIDEO_PARTH = "/mnt/c/Self-Study/TFG/dataset_2/subject1/vid.mp4"

def benchmark_face_detection(video_source):
    """
    Benchmark de detección facial - solo mide tiempo y FPS para detect_face()
    """
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video {video_source}")
        return
    
    face_detector = FaceDetector(model_type='mediapipe')
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Video: {video_source}")
    print(f"Frames totales: {total_frames}, FPS original: {fps:.1f}")
    print("-" * 50)
    
    frame_count = 0
    detection_times = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            # Solo medimos detect_face()
            start_time = time.time()
            roi = face_detector.detect_face(frame)
            end_time = time.time()
            
            detection_time = end_time - start_time
            detection_times.append(detection_time)

            # Mostrar progreso cada 200 frames
            if frame_count % 200 == 0:
                avg_time = sum(detection_times[-200:]) / min(200, len(detection_times[-200:]))
                current_fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"Frame {frame_count}: Tiempo detección: {avg_time*1000:.1f}ms, FPS: {current_fps:.1f}")
            

    except KeyboardInterrupt:
        print("\nBenchmark detenido por el usuario")
    except Exception as e:
        print(f"Error durante el benchmark: {e}")
    finally:
        cap.release()
        face_detector.close()
        
        # Estadísticas finales
        if detection_times:
            total_time = sum(detection_times)
            avg_time = total_time / len(detection_times)
            max_time = max(detection_times)
            min_time = min(detection_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            
            print(f"\n{'='*50}")
            print("RESULTADOS DEL BENCHMARK DE DETECCIÓN FACIAL")
            print(f"{'='*50}")
            print(f"Frames procesados: {frame_count}")
            print(f"Tiempo total detección: {total_time:.2f} segundos")
            print(f"Tiempo promedio por frame: {avg_time*1000:.1f} ms")
            print(f"Tiempo máximo: {max_time*1000:.1f} ms")
            print(f"Tiempo mínimo: {min_time*1000:.1f} ms")
            print(f"FPS promedio: {avg_fps:.1f}")
            print(f"Velocidad relativa: {avg_fps/fps*100:.1f}% del FPS original")

# Ejecutar benchmark
if __name__ == "__main__":
    benchmark_face_detection(VIDEO_PARTH)