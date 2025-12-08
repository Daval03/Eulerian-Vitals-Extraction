import numpy as np
from scipy import signal as sp_signal

def extract_temporal_signal(video_tensor, use_green_channel=True):
    """
    Extract temporal signal by spatially averaging each frame.
    
    Args:
        video_tensor: Frame tensor (T x H x W x C)
        use_green_channel: If True, use only green channel (better for HR)
    
    Returns:
        np.ndarray: 1D temporal signal
    """
    signal = []
    
    for frame in video_tensor:
        if use_green_channel and frame.ndim == 3:
            mean_val = np.mean(frame[:, :, 1])  # Green channel (index 1 in BGR)
        else:
            mean_val = np.mean(frame)
        signal.append(mean_val)
    
    return np.array(signal)


def preprocess_signal(signal):
    """
    Preprocess signal: detrending and normalization.
    
    Args:
        signal: 1D temporal signal
    
    Returns:
        np.ndarray: Preprocessed signal or None if fails
    """
    if signal is None or len(signal) < 20:
        return None
    
    # 1. Detrending (remove linear trend)
    signal_detrended = sp_signal.detrend(signal)
    
    # 2. Normalization
    signal_std = np.std(signal_detrended)
    if signal_std < 1e-8:
        return None
    
    signal_normalized = signal_detrended / signal_std
    
    return signal_normalized


def apply_hamming_window(signal):
    """
    Apply Hamming window to signal.
    
    Args:
        signal: Temporal signal
    
    Returns:
        np.ndarray: Windowed signal
    """
    window = np.hamming(len(signal))
    return signal * window


def calculate_power_spectrum(signal, fps):
    """
    Calculate power spectrum using FFT.
    
    Args:
        signal: Temporal signal
        fps: Frames per second
    
    Returns:
        tuple: (frequencies, power_spectrum)
    """
    # FFT
    fft_vals = np.fft.fft(signal)
    fft_freq = np.fft.fftfreq(len(signal), d=1.0/fps)
    
    # Power spectrum
    power_spectrum = np.abs(fft_vals) ** 2
    
    return fft_freq, power_spectrum


def find_dominant_frequency(fft_freq, power_spectrum, lowcut_hz, highcut_hz, 
                           min_bpm, max_bpm):
    """
    Find dominant frequency within specific range.
    
    Args:
        fft_freq: FFT frequency array
        power_spectrum: Power spectrum
        lowcut_hz: Minimum frequency (Hz)
        highcut_hz: Maximum frequency (Hz)
        min_bpm: Minimum valid BPM
        max_bpm: Maximum valid BPM
    
    Returns:
        float: Frequency in BPM or None
    """
    # Physiological range mask
    freq_mask = (fft_freq >= lowcut_hz) & (fft_freq <= highcut_hz)
    
    if not np.any(freq_mask):
        return None
    
    # Dominant peak
    masked_power = power_spectrum[freq_mask]
    masked_freqs = fft_freq[freq_mask]
    
    if len(masked_power) == 0:
        return None
    
    dominant_freq_hz = masked_freqs[np.argmax(masked_power)]
    
    # Convert to BPM/RPM
    frequency_bpm = abs(dominant_freq_hz * 60.0)
    
    # Validation
    if frequency_bpm < min_bpm or frequency_bpm > max_bpm:
        return None
    
    return frequency_bpm


def calculate_frequency_fft(temporal_signal, fps, lowcut_hz, highcut_hz, 
                           min_bpm, max_bpm):
    """
    Complete pipeline: calculate dominant frequency using FFT.
    
    Args:
        temporal_signal: 1D temporal signal
        fps: Frames per second
        lowcut_hz: Low cutoff frequency (Hz)
        highcut_hz: High cutoff frequency (Hz)
        min_bpm: Minimum valid BPM
        max_bpm: Maximum valid BPM
    
    Returns:
        float: Frequency in BPM or None if fails
    """
    # Preprocessing
    signal_processed = preprocess_signal(temporal_signal)
    if signal_processed is None:
        return None
    
    # Hamming window
    signal_windowed = apply_hamming_window(signal_processed)
    
    # FFT
    fft_freq, power_spectrum = calculate_power_spectrum(signal_windowed, fps)
    
    # Dominant frequency
    frequency_bpm = find_dominant_frequency(
        fft_freq, power_spectrum, lowcut_hz, highcut_hz, min_bpm, max_bpm
    )
    
    return frequency_bpm