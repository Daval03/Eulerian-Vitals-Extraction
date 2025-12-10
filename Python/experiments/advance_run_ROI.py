import cv2
import os
import sys
import time
import numpy as np
import json
from datetime import datetime
from utils_experiments import *
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from src.face_detector.manager import FaceDetector

# ==================== CONFIGURACIÓN ====================
MODEL_TYPE = 'yolo'  # Cambiar aquí: 'haar', 'mediapipe', 'mtcnn', 'yolo'
YOLO_MODEL = 'yolov12n'
BUFFER_SIZE = 50    # Cambiar según necesites
DATASET_PATH = "/mnt/c/Self-Study/TFG/dataset_2"
# =======================================================

class ROIMetrics:
    def __init__(self):
        self.detection_times = []
        self.successful_detections = 0
        self.total_frames = 0
        self.roi_positions = []  # [(x, y), ...]
        self.roi_sizes = []      # [(w, h), ...]
        self.consecutive_failures = []
        self.current_failure_count = 0
        
    def add_detection(self, detection_time, roi):
        self.detection_times.append(detection_time)
        self.total_frames += 1
        
        if roi is not None:
            self.successful_detections += 1
            x, y, w, h = roi
            self.roi_positions.append((x, y))
            self.roi_sizes.append((w, h))
            
            if self.current_failure_count > 0:
                self.consecutive_failures.append(self.current_failure_count)
                self.current_failure_count = 0
        else:
            self.current_failure_count += 1
    
    def calculate_jitter(self):
        """Calcula el jitter espacial promedio entre frames consecutivos"""
        if len(self.roi_positions) < 2:
            return None
        
        jitters = []
        for i in range(1, len(self.roi_positions)):
            x1, y1 = self.roi_positions[i-1]
            x2, y2 = self.roi_positions[i]
            jitter = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            jitters.append(jitter)
        
        return np.mean(jitters)
    
    def calculate_size_variance(self):
        """Calcula la varianza en el tamaño del ROI"""
        if len(self.roi_sizes) < 2:
            return None, None
        
        widths = [w for w, h in self.roi_sizes]
        heights = [h for w, h in self.roi_sizes]
        
        return np.std(widths), np.std(heights)
    
    def get_summary(self):
        """Retorna un resumen de todas las métricas"""
        detection_rate = (self.successful_detections / self.total_frames * 100) if self.total_frames > 0 else 0
        
        avg_time = np.mean(self.detection_times) if self.detection_times else 0
        std_time = np.std(self.detection_times) if self.detection_times else 0
        median_time = np.median(self.detection_times) if self.detection_times else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        jitter = self.calculate_jitter()
        std_width, std_height = self.calculate_size_variance()
        
        max_consecutive_failures = max(self.consecutive_failures) if self.consecutive_failures else 0
        avg_consecutive_failures = np.mean(self.consecutive_failures) if self.consecutive_failures else 0
        
        return {
            # Rendimiento
            'avg_detection_time_ms': avg_time * 1000,
            'std_detection_time_ms': std_time * 1000,
            'median_detection_time_ms': median_time * 1000,
            'fps': fps,
            
            # Calidad/Precisión
            'detection_rate_percent': detection_rate,
            'successful_detections': self.successful_detections,
            'total_frames': self.total_frames,
            
            # Estabilidad
            'jitter_avg_pixels': jitter,
            'roi_width_std': std_width,
            'roi_height_std': std_height,
            
            # Robustez
            'max_consecutive_failures': max_consecutive_failures,
            'avg_consecutive_failures': avg_consecutive_failures,
            'num_failure_episodes': len(self.consecutive_failures)
        }


def benchmark_video(video_path, model_type, buffer_size, yolo_model=None):
    """Benchmark de un video individual"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir {video_path}")
        return None
    
    if model_type == 'yolo' and yolo_model:
        face_detector = FaceDetector(model_type=model_type, preset=yolo_model)
    else:
        face_detector = FaceDetector(model_type=model_type)

    metrics = ROIMetrics()
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  📹 FPS original: {video_fps:.1f}, Total frames: {total_frames}")
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Medir solo detect_face()
            start_time = time.time()
            roi = face_detector.detect_face(frame)
            end_time = time.time()
            
            detection_time = end_time - start_time
            metrics.add_detection(detection_time, roi)
            
            # Progreso cada buffer_size frames
            if frame_count % buffer_size == 0:
                current_metrics = metrics.get_summary()
                print(f"    Frame {frame_count}/{total_frames}: "
                      f"Avg time: {current_metrics['avg_detection_time_ms']:.1f}ms, "
                      f"FPS: {current_metrics['fps']:.1f}, "
                      f"Detection: {current_metrics['detection_rate_percent']:.1f}%")
    
    except KeyboardInterrupt:
        print("\n⏸️  Benchmark interrumpido")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        face_detector.close()
    
    summary = metrics.get_summary()
    summary['video_fps'] = video_fps
    summary['frames_processed'] = frame_count
    
    return summary


def benchmark_all_subjects(dataset_path, model_type, buffer_size,  yolo_model=None):  # Eliminado num_subjects
    """Ejecuta benchmark en todos los sujetos y agrega resultados"""
    
    print("=" * 80)
    if model_type == 'yolo' and yolo_model:
        print(f"BENCHMARK DE DETECCIÓN ROI - {model_type.upper()} ({yolo_model})")
    else:
        print(f"BENCHMARK DE DETECCIÓN ROI - {model_type.upper()}")
    print(f"Buffer size: {buffer_size} frames")
    print("=" * 80)
    
    all_results = []
    failed_videos = []
    
    # Obtener lista dinámica de sujetos
    subjects = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item.startswith("subject"):
            subjects.append(item)
    
    # Ordenar sujetos numéricamente
    subjects.sort(key=lambda x: int(x[7:]) if x[7:].isdigit() else 0)
    
    print(f"📁 Encontrados {len(subjects)} sujetos en el dataset")
    
    for subject_dir in subjects:
        video_path = os.path.join(dataset_path, subject_dir, "vid.mp4")
        
        if not os.path.exists(video_path):
            print(f"⚠️  {subject_dir}: Video no encontrado")
            failed_videos.append(subject_dir)
            continue
        
        print(f"\n📊 {subject_dir}: {video_path}")
        
        result = benchmark_video(video_path, model_type, buffer_size, yolo_model)
        
        if result:
            result['subject_id'] = subject_dir  # Guardar nombre completo del directorio
            result['video_path'] = video_path
            if model_type == 'yolo' and yolo_model:
                result['yolo_model'] = yolo_model
            all_results.append(result)
            
            print(f"  ✅ Completado - FPS: {result['fps']:.1f}, "
                  f"Detection: {result['detection_rate_percent']:.1f}%, "
                  f"Jitter: {result['jitter_avg_pixels']:.2f}px")
        else:
            failed_videos.append(subject_dir)
    
    # Calcular estadísticas agregadas
    if all_results:
        print("\n" + "=" * 80)
        print("RESULTADOS AGREGADOS")
        print("=" * 80)
        
        aggregate_stats = calculate_aggregate_stats(all_results)
        print_aggregate_stats(aggregate_stats)
        
        # Guardar resultados
        save_results(all_results, aggregate_stats, model_type, buffer_size)
    
    if failed_videos:
        print(f"\n⚠️  Videos fallidos: {failed_videos}")
    
    return all_results


def calculate_aggregate_stats(results):
    """Calcula estadísticas agregadas de todos los videos"""
    stats = {}
    
    metrics_to_aggregate = [
        'avg_detection_time_ms', 'fps', 'detection_rate_percent',
        'jitter_avg_pixels', 'roi_width_std', 'roi_height_std',
        'max_consecutive_failures', 'avg_consecutive_failures'
    ]
    
    for metric in metrics_to_aggregate:
        values = [r[metric] for r in results if r.get(metric) is not None]
        if values:
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
            stats[f'{metric}_median'] = np.median(values)
            stats[f'{metric}_min'] = np.min(values)
            stats[f'{metric}_max'] = np.max(values)
    
    stats['total_videos'] = len(results)
    stats['total_frames'] = sum(r['frames_processed'] for r in results)
    stats['total_detections'] = sum(r['successful_detections'] for r in results)
    
    return stats


def print_aggregate_stats(stats):
    """Imprime estadísticas de forma legible"""
    print(f"\n📈 MÉTRICAS DE RENDIMIENTO")
    print(f"  Tiempo detección promedio: {stats['avg_detection_time_ms_mean']:.2f} ± {stats['avg_detection_time_ms_std']:.2f} ms")
    print(f"  FPS promedio: {stats['fps_mean']:.2f} ± {stats['fps_std']:.2f}")
    print(f"  FPS [min, max]: [{stats['fps_min']:.2f}, {stats['fps_max']:.2f}]")
    
    print(f"\n🎯 MÉTRICAS DE CALIDAD")
    print(f"  Tasa detección: {stats['detection_rate_percent_mean']:.2f} ± {stats['detection_rate_percent_std']:.2f}%")
    print(f"  Total frames procesados: {stats['total_frames']}")
    print(f"  Total detecciones exitosas: {stats['total_detections']}")
    
    print(f"\n📍 MÉTRICAS DE ESTABILIDAD")
    print(f"  Jitter promedio: {stats['jitter_avg_pixels_mean']:.2f} ± {stats['jitter_avg_pixels_std']:.2f} px")
    print(f"  Std ancho ROI: {stats['roi_width_std_mean']:.2f} ± {stats['roi_width_std_std']:.2f} px")
    print(f"  Std alto ROI: {stats['roi_height_std_mean']:.2f} ± {stats['roi_height_std_std']:.2f} px")
    
    print(f"\n🔄 MÉTRICAS DE ROBUSTEZ")
    print(f"  Fallos consecutivos (max): {stats['max_consecutive_failures_mean']:.1f} ± {stats['max_consecutive_failures_std']:.1f}")
    print(f"  Fallos consecutivos (avg): {stats['avg_consecutive_failures_mean']:.2f} ± {stats['avg_consecutive_failures_std']:.2f}")

def save_results(results, aggregate_stats, model_type, buffer_size, yolo_model=None):
    """Guarda resultados en archivo JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type == 'yolo' and yolo_model:
        filename = f"results/roi_benchmark_{model_type}_{yolo_model}_{timestamp}.json"
    else:
        filename = f"results/roi_benchmark_{model_type}_{timestamp}.json"
    
    output = {
        'configuration': {
            'model_type': model_type,
            'yolo_model': yolo_model,
            'buffer_size': buffer_size,
            'timestamp': timestamp,
            'num_videos': len(results)
        },
        'aggregate_statistics': aggregate_stats,
        'individual_results': results
    }
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=convert_numpy_types)
    
    print(f"\n💾 Resultados guardados en: {filename}")


if __name__ == "__main__":
    benchmark_all_subjects(
        dataset_path=DATASET_PATH,
        model_type=MODEL_TYPE,
        buffer_size=BUFFER_SIZE,
        yolo_model=YOLO_MODEL
    )