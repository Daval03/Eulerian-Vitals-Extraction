import cv2
import os
import sys
import time
import numpy as np
import json
import psutil
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
MODEL_TYPE = 'mediapipe'  # 'haar', 'mediapipe', 'mtcnn', 'yolo'
YOLO_MODEL = None   # Solo si MODEL_TYPE == 'yolo'
BUFFER_SIZE = 200
DATASET_PATH = "/mnt/c/Self-Study/TFG/dataset_2"
# =======================================================

class CompleteMetrics:
    """Métricas combinadas de ROI, EVM y recursos del sistema"""
    def __init__(self):
        # Métricas de tiempo por etapa
        self.detection_times = []      # Tiempo detección ROI por frame
        self.processing_times = []     # Tiempo procesamiento EVM por chunk
        self.end_to_end_times = []     # Tiempo total por chunk
        
        # Métricas de ROI
        self.successful_roi_detections = 0
        self.failed_roi_detections = 0
        self.total_frames = 0
        self.roi_positions = []
        self.roi_sizes = []
        self.consecutive_failures = []
        self.current_failure_count = 0
        
        # Métricas de precisión HR
        self.hr_errors = []
        self.hr_absolute_errors = []
        self.hr_predictions = []
        self.hr_ground_truths = []
        
        # Métricas de chunks
        self.total_chunks = 0
        self.frames_per_chunk = []
        self.chunk_detection_times = []  # Tiempo acumulado de detección por chunk
        
        # Métricas de recursos (importantes para RPi4)
        self.cpu_usage_samples = []
        self.memory_usage_samples = []
        self.temperature_samples = []
        self.cpu_freq_samples = []
        
    def add_frame_detection(self, detection_time, roi):
        """Registra resultado de detección de un frame"""
        self.detection_times.append(detection_time)
        self.total_frames += 1
        
        if roi is not None:
            self.successful_roi_detections += 1
            x, y, w, h = roi
            self.roi_positions.append((x, y))
            self.roi_sizes.append((w, h))
            
            if self.current_failure_count > 0:
                self.consecutive_failures.append(self.current_failure_count)
                self.current_failure_count = 0
        else:
            self.failed_roi_detections += 1
            self.current_failure_count += 1
    
    def add_chunk_result(self, processing_time, chunk_detection_time, frames_in_chunk, hr_pred, hr_true):
        """Agrega resultado de un chunk procesado"""
        self.processing_times.append(processing_time)
        self.chunk_detection_times.append(chunk_detection_time)
        self.end_to_end_times.append(chunk_detection_time + processing_time)
        self.frames_per_chunk.append(frames_in_chunk)
        self.total_chunks += 1
        
        if hr_pred is not None and hr_true is not None:
            error = hr_pred - hr_true
            abs_error = abs(error)
            self.hr_errors.append(error)
            self.hr_absolute_errors.append(abs_error)
            self.hr_predictions.append(hr_pred)
            self.hr_ground_truths.append(hr_true)
    
    def add_system_resources(self):
        """Captura métricas de recursos del sistema (RPi4)"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            self.cpu_usage_samples.append(cpu_percent)
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.memory_usage_samples.append(memory.percent)
            
            # CPU frequency (importante para throttling en RPi4)
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                self.cpu_freq_samples.append(cpu_freq.current)
            
            # Temperature (crítico para RPi4)
            try:
                temps = psutil.sensors_temperatures()
                if 'cpu_thermal' in temps:
                    temp = temps['cpu_thermal'][0].current
                    self.temperature_samples.append(temp)
                elif 'cpu-thermal' in temps:
                    temp = temps['cpu-thermal'][0].current
                    self.temperature_samples.append(temp)
            except:
                pass
        except Exception as e:
            pass
    
    def calculate_roi_jitter(self):
        """Calcula el jitter espacial del ROI"""
        if len(self.roi_positions) < 2:
            return None
        
        jitters = []
        for i in range(1, len(self.roi_positions)):
            x1, y1 = self.roi_positions[i-1]
            x2, y2 = self.roi_positions[i]
            jitter = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            jitters.append(jitter)
        
        return np.mean(jitters)
    
    def calculate_roi_size_variance(self):
        """Calcula la varianza en el tamaño del ROI"""
        if len(self.roi_sizes) < 2:
            return None, None
        
        widths = [w for w, h in self.roi_sizes]
        heights = [h for w, h in self.roi_sizes]
        
        return np.std(widths), np.std(heights)
    
    def get_summary(self):
        """Retorna resumen completo de todas las métricas"""
        summary = {}
        
        # ===== MÉTRICAS DE RENDIMIENTO POR ETAPA =====
        if self.detection_times:
            avg_detection_time = np.mean(self.detection_times)
            detection_fps = 1.0 / avg_detection_time if avg_detection_time > 0 else 0
            
            summary['roi_performance'] = {
                'total_frames': self.total_frames,
                'avg_detection_time_ms': avg_detection_time * 1000,
                'std_detection_time_ms': np.std(self.detection_times) * 1000,
                'median_detection_time_ms': np.median(self.detection_times) * 1000,
                'max_detection_time_ms': max(self.detection_times) * 1000,
                'min_detection_time_ms': min(self.detection_times) * 1000,
                'detection_fps': detection_fps
            }
        
        if self.processing_times:
            avg_proc_time = np.mean(self.processing_times)
            avg_frames_chunk = np.mean(self.frames_per_chunk) if self.frames_per_chunk else 0
            evm_time_per_frame = avg_proc_time / avg_frames_chunk if avg_frames_chunk > 0 else 0
            evm_fps = 1.0 / evm_time_per_frame if evm_time_per_frame > 0 else 0
            
            summary['evm_performance'] = {
                'total_chunks': self.total_chunks,
                'avg_chunk_time_s': avg_proc_time,
                'std_chunk_time_s': np.std(self.processing_times),
                'median_chunk_time_s': np.median(self.processing_times),
                'max_chunk_time_s': max(self.processing_times),
                'min_chunk_time_s': min(self.processing_times),
                'avg_time_per_frame_ms': evm_time_per_frame * 1000,
                'evm_fps': evm_fps
            }
        
        if self.end_to_end_times:
            avg_e2e_time = np.mean(self.end_to_end_times)
            avg_frames_chunk = np.mean(self.frames_per_chunk) if self.frames_per_chunk else 0
            e2e_time_per_frame = avg_e2e_time / avg_frames_chunk if avg_frames_chunk > 0 else 0
            e2e_fps = 1.0 / e2e_time_per_frame if e2e_time_per_frame > 0 else 0
            
            # Desglose porcentual
            avg_chunk_detection = np.mean(self.chunk_detection_times) if self.chunk_detection_times else 0
            avg_chunk_evm = np.mean(self.processing_times) if self.processing_times else 0
            detection_percentage = (avg_chunk_detection / avg_e2e_time * 100) if avg_e2e_time > 0 else 0
            evm_percentage = (avg_chunk_evm / avg_e2e_time * 100) if avg_e2e_time > 0 else 0
            
            summary['end_to_end_performance'] = {
                'avg_chunk_time_s': avg_e2e_time,
                'std_chunk_time_s': np.std(self.end_to_end_times),
                'median_chunk_time_s': np.median(self.end_to_end_times),
                'max_chunk_time_s': max(self.end_to_end_times),
                'min_chunk_time_s': min(self.end_to_end_times),
                'avg_time_per_frame_ms': e2e_time_per_frame * 1000,
                'end_to_end_fps': e2e_fps,
                'detection_time_percentage': detection_percentage,
                'evm_time_percentage': evm_percentage
            }
        
        # ===== MÉTRICAS DE CALIDAD ROI =====
        roi_detection_rate = (self.successful_roi_detections / self.total_frames * 100) if self.total_frames > 0 else 0
        jitter = self.calculate_roi_jitter()
        std_width, std_height = self.calculate_roi_size_variance()
        
        max_consecutive_failures = max(self.consecutive_failures) if self.consecutive_failures else 0
        avg_consecutive_failures = np.mean(self.consecutive_failures) if self.consecutive_failures else 0
        
        summary['roi_quality'] = {
            'detection_rate_percent': roi_detection_rate,
            'successful_detections': self.successful_roi_detections,
            'failed_detections': self.failed_roi_detections,
            'jitter_avg_pixels': jitter,
            'roi_width_std': std_width,
            'roi_height_std': std_height,
            'max_consecutive_failures': max_consecutive_failures,
            'avg_consecutive_failures': avg_consecutive_failures,
            'num_failure_episodes': len(self.consecutive_failures)
        }
        
        # ===== MÉTRICAS DE PRECISIÓN HR =====
        if self.hr_absolute_errors:
            mae = np.mean(self.hr_absolute_errors)
            me = np.mean(self.hr_errors)
            rmse = np.sqrt(np.mean(np.array(self.hr_errors)**2))
            std_error = np.std(self.hr_errors)
            
            within_5 = sum(1 for e in self.hr_absolute_errors if e <= 5) / len(self.hr_absolute_errors) * 100
            within_10 = sum(1 for e in self.hr_absolute_errors if e <= 10) / len(self.hr_absolute_errors) * 100
            within_15 = sum(1 for e in self.hr_absolute_errors if e <= 15) / len(self.hr_absolute_errors) * 100
            
            correlation = None
            if len(self.hr_predictions) > 1:
                try:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        correlation = np.corrcoef(self.hr_predictions, self.hr_ground_truths)[0, 1]
                        if np.isnan(correlation) or np.isinf(correlation):
                            correlation = None
                except:
                    correlation = None
            
            summary['hr_accuracy'] = {
                'num_measurements': len(self.hr_absolute_errors),
                'mae_bpm': mae,
                'me_bpm': me,
                'rmse_bpm': rmse,
                'std_error_bpm': std_error,
                'max_error_bpm': max(self.hr_absolute_errors),
                'min_error_bpm': min(self.hr_absolute_errors),
                'within_5bpm_percent': within_5,
                'within_10bpm_percent': within_10,
                'within_15bpm_percent': within_15,
                'correlation': correlation,
                'hr_pred_min': min(self.hr_predictions),
                'hr_pred_max': max(self.hr_predictions),
                'hr_true_min': min(self.hr_ground_truths),
                'hr_true_max': max(self.hr_ground_truths)
            }
        
        # ===== MÉTRICAS DE RECURSOS DEL SISTEMA (RPi4) =====
        if self.cpu_usage_samples:
            summary['system_resources'] = {
                'cpu_usage_avg_percent': np.mean(self.cpu_usage_samples),
                'cpu_usage_max_percent': max(self.cpu_usage_samples),
                'cpu_usage_std_percent': np.std(self.cpu_usage_samples),
                'num_cpu_samples': len(self.cpu_usage_samples)
            }
        
        if self.memory_usage_samples:
            summary['system_resources']['memory_usage_avg_percent'] = np.mean(self.memory_usage_samples)
            summary['system_resources']['memory_usage_max_percent'] = max(self.memory_usage_samples)
            summary['system_resources']['memory_usage_std_percent'] = np.std(self.memory_usage_samples)
        
        if self.cpu_freq_samples:
            summary['system_resources']['cpu_freq_avg_mhz'] = np.mean(self.cpu_freq_samples)
            summary['system_resources']['cpu_freq_min_mhz'] = min(self.cpu_freq_samples)
            summary['system_resources']['cpu_freq_max_mhz'] = max(self.cpu_freq_samples)
            
            # Detección de throttling (importante en RPi4)
            freq_std = np.std(self.cpu_freq_samples)
            summary['system_resources']['cpu_freq_std_mhz'] = freq_std
            summary['system_resources']['potential_throttling'] = freq_std > 100
        
        if self.temperature_samples:
            summary['system_resources']['temperature_avg_celsius'] = np.mean(self.temperature_samples)
            summary['system_resources']['temperature_max_celsius'] = max(self.temperature_samples)
            summary['system_resources']['temperature_min_celsius'] = min(self.temperature_samples)
            summary['system_resources']['temperature_std_celsius'] = np.std(self.temperature_samples)
            
            # Alertas de temperatura (RPi4 throttlea a ~80°C)
            max_temp = max(self.temperature_samples)
            summary['system_resources']['temperature_warning'] = max_temp > 75
            summary['system_resources']['temperature_critical'] = max_temp > 80
        
        return summary


def benchmark_video(video_path, subject_num, model_type, buffer_size, yolo_model=None):
    """Benchmark completo de un video individual"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    gt_handler, _ = setup_ground_truth(subject_num, DATASET_PATH, buffer_size)
    
    if model_type == 'yolo' and yolo_model:
        face_detector = FaceDetector(model_type=model_type, preset=yolo_model)
    else:
        face_detector = FaceDetector(model_type=model_type)
    
    metrics = CompleteMetrics()
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    measurement_count = 0
    frame_buffer = []
    chunk_detection_time = 0
    
    hr_history = deque(maxlen=10)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if frame_count % 30 == 0:
                metrics.add_system_resources()
            
            detection_start = time.time()
            roi = face_detector.detect_face(frame)
            detection_end = time.time()
            detection_time = detection_end - detection_start
            
            metrics.add_frame_detection(detection_time, roi)
            chunk_detection_time += detection_time
            
            if roi:
                x, y, w, h = roi
                roi_frame = frame[y:y+h, x:x+w]
                roi_frame = cv2.resize(roi_frame, TARGET_ROI_SIZE)
                frame_buffer.append(roi_frame)
                
                if len(frame_buffer) >= buffer_size:
                    true_hr = gt_handler.get_hr_for_chunk(measurement_count)
                    if true_hr is None:
                        break
                    
                    measurement_count += 1
                    
                    proc_start = time.time()
                    results = process_video_evm_vital_signs(frame_buffer)
                    proc_end = time.time()
                    processing_time = proc_end - proc_start
                    
                    hr = results['heart_rate']
                    hr_history.append(hr)
                    filtered_hr = np.median(list(hr_history)) if hr_history else None
                    
                    metrics.add_chunk_result(
                        processing_time=processing_time,
                        chunk_detection_time=chunk_detection_time,
                        frames_in_chunk=len(frame_buffer),
                        hr_pred=filtered_hr,
                        hr_true=true_hr
                    )
                    
                    metrics.add_system_resources()
                    
                    frame_buffer.clear()
                    chunk_detection_time = 0
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass
    finally:
        cap.release()
        face_detector.close()
    
    summary = metrics.get_summary()
    summary['video_fps'] = video_fps
    summary['buffer_size'] = buffer_size
    summary['model_type'] = model_type
    if model_type == 'yolo' and yolo_model:
        summary['yolo_model'] = yolo_model
    
    return summary


def benchmark_all_subjects(dataset_path, model_type, buffer_size, yolo_model=None):
    """Ejecuta benchmark completo en todos los sujetos"""
    
    all_results = []
    failed_videos = []
    
    subjects = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path) and item.startswith("subject"):
            subjects.append(item)
    
    subjects.sort(key=lambda x: int(x[7:]) if x[7:].isdigit() else 0)
    
    for subject_dir in subjects:
        video_path = os.path.join(dataset_path, subject_dir, "vid.mp4")
        
        if not os.path.exists(video_path):
            failed_videos.append(subject_dir)
            continue
        
        subject_num = int(subject_dir[7:]) if subject_dir[7:].isdigit() else None
        if subject_num is None:
            failed_videos.append(subject_dir)
            continue
        
        result = benchmark_video(video_path, subject_num, model_type, buffer_size, yolo_model)
        
        if result and 'hr_accuracy' in result:
            result['subject_id'] = subject_dir
            result['subject_num'] = subject_num
            result['video_path'] = video_path
            all_results.append(result)
        else:
            failed_videos.append(subject_dir)
    
    if all_results:
        aggregate_stats = calculate_aggregate_stats(all_results)
        save_results(all_results, aggregate_stats, model_type, buffer_size, yolo_model)
    
    return all_results


def calculate_aggregate_stats(results):
    """Calcula estadísticas agregadas de todos los videos"""
    stats = {}
    
    perf_categories = {
        'roi_performance': ['avg_detection_time_ms', 'detection_fps'],
        'evm_performance': ['avg_chunk_time_s', 'avg_time_per_frame_ms', 'evm_fps'],
        'end_to_end_performance': ['avg_chunk_time_s', 'avg_time_per_frame_ms', 'end_to_end_fps',
                                    'detection_time_percentage', 'evm_time_percentage']
    }
    
    for category, metrics in perf_categories.items():
        for metric in metrics:
            values = [r[category][metric] for r in results if category in r and metric in r[category]]
            if values:
                stats[f'{category}_{metric}_mean'] = np.mean(values)
                stats[f'{category}_{metric}_std'] = np.std(values)
                stats[f'{category}_{metric}_median'] = np.median(values)
                stats[f'{category}_{metric}_min'] = np.min(values)
                stats[f'{category}_{metric}_max'] = np.max(values)
    
    roi_metrics = ['detection_rate_percent', 'jitter_avg_pixels', 'roi_width_std', 'roi_height_std',
                   'max_consecutive_failures', 'avg_consecutive_failures']
    
    for metric in roi_metrics:
        values = [r['roi_quality'][metric] for r in results if 'roi_quality' in r and metric in r['roi_quality']
                  and r['roi_quality'][metric] is not None]
        if values:
            stats[f'roi_quality_{metric}_mean'] = np.mean(values)
            stats[f'roi_quality_{metric}_std'] = np.std(values)
    
    hr_metrics = ['mae_bpm', 'me_bpm', 'rmse_bpm', 'std_error_bpm',
                  'within_5bpm_percent', 'within_10bpm_percent', 'within_15bpm_percent']
    
    for metric in hr_metrics:
        values = [r['hr_accuracy'][metric] for r in results if 'hr_accuracy' in r and metric in r['hr_accuracy']]
        if values:
            stats[f'hr_accuracy_{metric}_mean'] = np.mean(values)
            stats[f'hr_accuracy_{metric}_std'] = np.std(values)
    
    if any('system_resources' in r for r in results):
        resource_metrics = ['cpu_usage_avg_percent', 'cpu_usage_max_percent',
                           'memory_usage_avg_percent', 'memory_usage_max_percent',
                           'cpu_freq_avg_mhz', 'temperature_avg_celsius', 'temperature_max_celsius']
        
        for metric in resource_metrics:
            values = [r['system_resources'][metric] for r in results 
                     if 'system_resources' in r and metric in r['system_resources']]
            if values:
                stats[f'system_{metric}_mean'] = np.mean(values)
                stats[f'system_{metric}_std'] = np.std(values)
                stats[f'system_{metric}_max'] = np.max(values)
        
        throttling_cases = sum(1 for r in results if r.get('system_resources', {}).get('potential_throttling', False))
        temp_warning_cases = sum(1 for r in results if r.get('system_resources', {}).get('temperature_warning', False))
        temp_critical_cases = sum(1 for r in results if r.get('system_resources', {}).get('temperature_critical', False))
        
        stats['system_throttling_cases'] = throttling_cases
        stats['system_temp_warning_cases'] = temp_warning_cases
        stats['system_temp_critical_cases'] = temp_critical_cases
    
    stats['total_videos'] = len(results)
    stats['total_chunks'] = sum(r.get('evm_performance', {}).get('total_chunks', 0) for r in results)
    stats['total_frames'] = sum(r.get('roi_performance', {}).get('total_frames', 0) for r in results)
    stats['total_measurements'] = sum(r.get('hr_accuracy', {}).get('num_measurements', 0) for r in results)
    
    return stats


def save_results(results, aggregate_stats, model_type, buffer_size, yolo_model=None):
    """Guarda resultados en archivo JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if model_type == 'yolo' and yolo_model:
        filename = f"results/complete_benchmark_{model_type}_{yolo_model}_{timestamp}.json"
    else:
        filename = f"results/complete_benchmark_{model_type}_{timestamp}.json"
    
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
    
    os.makedirs('results', exist_ok=True)
    
    with open(filename, 'w') as f:
        json.dump(output, f, indent=2, default=convert_numpy_types)


if __name__ == "__main__":
    benchmark_all_subjects(
        dataset_path=DATASET_PATH,
        model_type=MODEL_TYPE,
        buffer_size=BUFFER_SIZE,
        yolo_model=YOLO_MODEL if MODEL_TYPE == 'yolo' else None
    )