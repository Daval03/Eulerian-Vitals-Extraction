import cv2
import os
import sys
import time
import numpy as np
import json
from datetime import datetime
from collections import deque

from utils_experiments import *
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from src.face_detector.manager import FaceDetector
from src.evm.evm_manager import process_video_evm_vital_signs
from src.utils.ground_truth_handler import setup_ground_truth
from src.config import TARGET_ROI_SIZE

# ==================== CONFIGURACIÓN ====================
BUFFER_SIZE = 200
DATASET_PATH = "/mnt/c/Self-Study/TFG/dataset_2"
# =======================================================

class EVMMetrics:
    def __init__(self):
        # Métricas de rendimiento
        self.processing_times = []  # Tiempo por chunk
        self.frames_per_chunk = []
        
        # Métricas de precisión (Heart Rate)
        self.hr_errors = []
        self.hr_absolute_errors = []
        self.hr_predictions = []
        self.hr_ground_truths = []
        
        # Métricas de detección ROI
        self.successful_roi_detections = 0
        self.failed_roi_detections = 0
        self.total_frames = 0
        
        # Chunks procesados
        self.total_chunks = 0
        
    def add_chunk_result(self, processing_time, frames_in_chunk, hr_pred, hr_true):
        """Agrega resultado de un chunk procesado"""
        self.processing_times.append(processing_time)
        self.frames_per_chunk.append(frames_in_chunk)
        self.total_chunks += 1
        
        if hr_pred is not None and hr_true is not None:
            error = hr_pred - hr_true
            abs_error = abs(error)
            self.hr_errors.append(error)
            self.hr_absolute_errors.append(abs_error)
            self.hr_predictions.append(hr_pred)
            self.hr_ground_truths.append(hr_true)
    
    def add_frame_detection(self, roi_detected):
        """Registra si se detectó ROI en un frame"""
        self.total_frames += 1
        if roi_detected:
            self.successful_roi_detections += 1
        else:
            self.failed_roi_detections += 1
    
    def get_summary(self):
        """Retorna un resumen de todas las métricas"""
        summary = {}
        
        # Métricas de rendimiento
        if self.processing_times:
            total_proc_time = sum(self.processing_times)
            avg_proc_time = np.mean(self.processing_times)
            std_proc_time = np.std(self.processing_times)
            max_proc_time = max(self.processing_times)
            min_proc_time = min(self.processing_times)
            
            # Calcular métricas por frame
            avg_frames_chunk = np.mean(self.frames_per_chunk) if self.frames_per_chunk else 0
            avg_time_per_frame = avg_proc_time / avg_frames_chunk if avg_frames_chunk > 0 else 0
            avg_fps = 1.0 / avg_time_per_frame if avg_time_per_frame > 0 else 0
            
            summary['performance'] = {
                'total_processing_time_s': total_proc_time,
                'avg_chunk_time_s': avg_proc_time,
                'std_chunk_time_s': std_proc_time,
                'max_chunk_time_s': max_proc_time,
                'min_chunk_time_s': min_proc_time,
                'avg_time_per_frame_ms': avg_time_per_frame * 1000,
                'avg_fps': avg_fps,
                'total_chunks': self.total_chunks,
                'frames_processed': sum(self.frames_per_chunk)
            }
        
        # Métricas de precisión (Heart Rate)
        if self.hr_absolute_errors:
            mae = np.mean(self.hr_absolute_errors)
            me = np.mean(self.hr_errors)
            rmse = np.sqrt(np.mean(np.array(self.hr_errors)**2))
            std_error = np.std(self.hr_errors)
            max_error = max(self.hr_absolute_errors)
            min_error = min(self.hr_absolute_errors)
            
            # Porcentaje de predicciones dentro de rangos
            within_5 = sum(1 for e in self.hr_absolute_errors if e <= 5) / len(self.hr_absolute_errors) * 100
            within_10 = sum(1 for e in self.hr_absolute_errors if e <= 10) / len(self.hr_absolute_errors) * 100
            within_15 = sum(1 for e in self.hr_absolute_errors if e <= 15) / len(self.hr_absolute_errors) * 100
            
            # Correlación
            correlation = None
            if len(self.hr_predictions) > 1:
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        correlation = np.corrcoef(self.hr_predictions, self.hr_ground_truths)[0, 1]
                        if np.isnan(correlation) or np.isinf(correlation):
                            correlation = None
                except:
                    correlation = None
            
            summary['accuracy'] = {
                'num_measurements': len(self.hr_absolute_errors),
                'mae_bpm': mae,
                'me_bpm': me,
                'rmse_bpm': rmse,
                'std_error_bpm': std_error,
                'max_error_bpm': max_error,
                'min_error_bpm': min_error,
                'within_5bpm_percent': within_5,
                'within_10bpm_percent': within_10,
                'within_15bpm_percent': within_15,
                'correlation': correlation,
                'hr_pred_min': min(self.hr_predictions),
                'hr_pred_max': max(self.hr_predictions),
                'hr_true_min': min(self.hr_ground_truths),
                'hr_true_max': max(self.hr_ground_truths)
            }
        
        # Métricas de detección ROI
        roi_detection_rate = (self.successful_roi_detections / self.total_frames * 100) if self.total_frames > 0 else 0
        summary['roi_detection'] = {
            'total_frames': self.total_frames,
            'successful_detections': self.successful_roi_detections,
            'failed_detections': self.failed_roi_detections,
            'detection_rate_percent': roi_detection_rate
        }
        
        return summary


def benchmark_video(video_path, subject_num, buffer_size):
    """Benchmark de un video individual con EVM"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir {video_path}")
        return None
    
    # Setup ground truth
    gt_handler, _ = setup_ground_truth(subject_num, DATASET_PATH, buffer_size)
    
    face_detector = FaceDetector(model_type='mediapipe')
    metrics = EVMMetrics()
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"  📹 FPS original: {video_fps:.1f}, Total frames: {total_frames}")
    
    frame_count = 0
    measurement_count = 0
    frame_buffer = []
    
    # Historial para filtrado
    hr_history = deque(maxlen=10)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            roi = face_detector.detect_face(frame)
            metrics.add_frame_detection(roi is not None)
            
            if roi:
                x, y, w, h = roi
                roi_frame = frame[y:y+h, x:x+w]
                roi_frame = cv2.resize(roi_frame, TARGET_ROI_SIZE)
                frame_buffer.append(roi_frame)
                
                if len(frame_buffer) >= buffer_size:
                    true_hr = gt_handler.get_hr_for_chunk(measurement_count)
                    if true_hr is None:
                        print(f"    ⚠️  No hay más ground truth para chunk {measurement_count}")
                        break
                    
                    measurement_count += 1
                    
                    # Medir tiempo de procesamiento EVM
                    proc_start = time.time()
                    results = process_video_evm_vital_signs(frame_buffer)
                    proc_end = time.time()
                    processing_time = proc_end - proc_start
                    
                    hr = results['heart_rate']
                    hr_history.append(hr)
                    filtered_hr = np.median(list(hr_history)) if hr_history else None
                    
                    # Agregar resultado del chunk
                    metrics.add_chunk_result(
                        processing_time=processing_time,
                        frames_in_chunk=len(frame_buffer),
                        hr_pred=filtered_hr,
                        hr_true=true_hr
                    )
                    
                    # Progreso cada chunk
                    if frame_count % buffer_size == 0:
                        current_metrics = metrics.get_summary()
                        perf = current_metrics['performance']
                        acc = current_metrics.get('accuracy', {})
                        print(f"    Chunk {measurement_count}: "
                              f"Avg time: {perf['avg_time_per_frame_ms']:.1f}ms/frame, "
                              f"FPS: {perf['avg_fps']:.1f}, "
                              f"MAE: {acc.get('mae_bpm', 0):.1f} BPM")
                    
                    frame_buffer.clear()
    
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
    summary['buffer_size'] = buffer_size
    
    return summary


def benchmark_all_subjects(dataset_path, buffer_size):
    """Ejecuta benchmark en todos los sujetos y agrega resultados"""
    
    print("=" * 80)
    print(f"BENCHMARK DE PROCESAMIENTO EVM CON MEDIAPIPE")
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
    
    print(f"🔍 Encontrados {len(subjects)} sujetos en el dataset")
    
    for subject_dir in subjects:
        video_path = os.path.join(dataset_path, subject_dir, "vid.mp4")
        
        if not os.path.exists(video_path):
            print(f"⚠️  {subject_dir}: Video no encontrado")
            failed_videos.append(subject_dir)
            continue
        
        # Extraer número de sujeto
        subject_num = int(subject_dir[7:]) if subject_dir[7:].isdigit() else None
        if subject_num is None:
            print(f"⚠️  {subject_dir}: No se pudo extraer número de sujeto")
            failed_videos.append(subject_dir)
            continue
        
        print(f"\n📊 {subject_dir}: {video_path}")
        
        result = benchmark_video(video_path, subject_num, buffer_size)
        
        if result and 'accuracy' in result:
            result['subject_id'] = subject_dir
            result['subject_num'] = subject_num
            result['video_path'] = video_path
            all_results.append(result)
            
            perf = result['performance']
            acc = result['accuracy']
            roi = result['roi_detection']
            
            print(f"  ✅ Completado - FPS: {perf['avg_fps']:.1f}, "
                  f"MAE: {acc['mae_bpm']:.1f} BPM, "
                  f"ROI Detection: {roi['detection_rate_percent']:.1f}%")
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
        save_results(all_results, aggregate_stats, buffer_size)
    
    if failed_videos:
        print(f"\n⚠️  Videos fallidos: {failed_videos}")
    
    return all_results


def calculate_aggregate_stats(results):
    """Calcula estadísticas agregadas de todos los videos"""
    stats = {}
    
    # Métricas de rendimiento a agregar
    perf_metrics = [
        'avg_chunk_time_s', 'avg_time_per_frame_ms', 'avg_fps'
    ]
    
    for metric in perf_metrics:
        values = [r['performance'][metric] for r in results if metric in r.get('performance', {})]
        if values:
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
            stats[f'{metric}_median'] = np.median(values)
            stats[f'{metric}_min'] = np.min(values)
            stats[f'{metric}_max'] = np.max(values)
    
    # Métricas de precisión a agregar
    acc_metrics = [
        'mae_bpm', 'me_bpm', 'rmse_bpm', 'std_error_bpm',
        'within_5bpm_percent', 'within_10bpm_percent', 'within_15bpm_percent'
    ]
    
    for metric in acc_metrics:
        values = [r['accuracy'][metric] for r in results if metric in r.get('accuracy', {})]
        if values:
            stats[f'{metric}_mean'] = np.mean(values)
            stats[f'{metric}_std'] = np.std(values)
            stats[f'{metric}_median'] = np.median(values)
            stats[f'{metric}_min'] = np.min(values)
            stats[f'{metric}_max'] = np.max(values)
    
    # Métricas de ROI detection
    roi_metrics = ['detection_rate_percent']
    for metric in roi_metrics:
        values = [r['roi_detection'][metric] for r in results if metric in r.get('roi_detection', {})]
        if values:
            stats[f'roi_{metric}_mean'] = np.mean(values)
            stats[f'roi_{metric}_std'] = np.std(values)
    
    stats['total_videos'] = len(results)
    stats['total_chunks'] = sum(r['performance']['total_chunks'] for r in results if 'performance' in r)
    stats['total_frames'] = sum(r['performance']['frames_processed'] for r in results if 'performance' in r)
    stats['total_measurements'] = sum(r['accuracy']['num_measurements'] for r in results if 'accuracy' in r)
    
    return stats


def print_aggregate_stats(stats):
    """Imprime estadísticas de forma legible"""
    print(f"\n📈 MÉTRICAS DE RENDIMIENTO")
    print(f"  Tiempo procesamiento/frame: {stats['avg_time_per_frame_ms_mean']:.2f} ± {stats['avg_time_per_frame_ms_std']:.2f} ms")
    print(f"  FPS promedio: {stats['avg_fps_mean']:.2f} ± {stats['avg_fps_std']:.2f}")
    print(f"  FPS [min, max]: [{stats['avg_fps_min']:.2f}, {stats['avg_fps_max']:.2f}]")
    print(f"  Tiempo/chunk: {stats['avg_chunk_time_s_mean']:.2f} ± {stats['avg_chunk_time_s_std']:.2f} s")
    
    print(f"\n🎯 MÉTRICAS DE PRECISIÓN (HEART RATE)")
    print(f"  MAE: {stats['mae_bpm_mean']:.2f} ± {stats['mae_bpm_std']:.2f} BPM")
    print(f"  ME: {stats['me_bpm_mean']:+.2f} ± {stats['me_bpm_std']:.2f} BPM")
    print(f"  RMSE: {stats['rmse_bpm_mean']:.2f} ± {stats['rmse_bpm_std']:.2f} BPM")
    print(f"  Std Error: {stats['std_error_bpm_mean']:.2f} ± {stats['std_error_bpm_std']:.2f} BPM")
    
    print(f"\n📊 DISTRIBUCIÓN DE ERRORES")
    print(f"  Dentro de ±5 BPM: {stats['within_5bpm_percent_mean']:.1f} ± {stats['within_5bpm_percent_std']:.1f}%")
    print(f"  Dentro de ±10 BPM: {stats['within_10bpm_percent_mean']:.1f} ± {stats['within_10bpm_percent_std']:.1f}%")
    print(f"  Dentro de ±15 BPM: {stats['within_15bpm_percent_mean']:.1f} ± {stats['within_15bpm_percent_std']:.1f}%")
    
    print(f"\n🔍 DETECCIÓN ROI")
    print(f"  Tasa detección: {stats['roi_detection_rate_percent_mean']:.2f} ± {stats['roi_detection_rate_percent_std']:.2f}%")
    
    print(f"\n📋 RESUMEN GENERAL")
    print(f"  Total videos: {stats['total_videos']}")
    print(f"  Total frames procesados: {stats['total_frames']}")
    print(f"  Total chunks procesados: {stats['total_chunks']}")
    print(f"  Total mediciones HR: {stats['total_measurements']}")


def save_results(results, aggregate_stats, buffer_size):
    """Guarda resultados en archivo JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/evm_benchmark_mediapipe_{timestamp}.json"
    
    output = {
        'configuration': {
            'model_type': 'mediapipe',
            'buffer_size': buffer_size,
            'timestamp': timestamp,
            'num_videos': len(results)
        },
        'aggregate_statistics': aggregate_stats,
        'individual_results': results
    }
    
    # Crear directorio si no existe
    os.makedirs('results', exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=convert_numpy_types)
    
    print(f"\n💾 Resultados guardados en: {filename}")


if __name__ == "__main__":
    benchmark_all_subjects(
        dataset_path=DATASET_PATH,
        buffer_size=50
    )
    benchmark_all_subjects(
        dataset_path=DATASET_PATH,
        buffer_size=100
    )