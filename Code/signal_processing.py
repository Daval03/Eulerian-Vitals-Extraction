import numpy as np
import scipy.fftpack as fft
import cv2
from scipy.signal import butter, filtfilt
from sklearn.decomposition import FastICA, PCA
from Code.config import LOW_HEART, HIGH_HEART, LOW_RESP, HIGH_RESP, LEVELS, FPS, ALPHA

def build_laplacian_pyramid(frame, levels):
    """Construye una pirámide Laplaciana."""
    pyramid = []
    current = frame.copy()
    for _ in range(levels):
        down = cv2.pyrDown(current)  # Nivel k+1 (Gaussiana)
        up = cv2.pyrUp(down, dstsize=(current.shape[1], current.shape[0]))  # Expandir
        laplacian = cv2.subtract(current, up)  # L_k = G_k - expand(G_{k+1})
        pyramid.append(laplacian)
        current = down  # Continuar con el siguiente nivel Gaussiano
    pyramid.append(current)  # Añadir el último nivel Gaussiano (sin resta)
    return pyramid

def apply_ica_pca(signals):
    """Aplica ICA y PCA a las señales."""
    ica = FastICA(n_components=1, random_state=42)
    pca = PCA(n_components=1)
    return (ica.fit_transform(signals)[:, 0], 
            pca.fit_transform(signals)[:, 0])

def butter_bandpass(lowcut, highcut, fs, order=3):
    """Diseña un filtro de paso de banda Butterworth."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(signal, lowcut, highcut, fs):
    """Aplica un filtro de paso de banda."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=3)
    return filtfilt(b, a, signal, axis=0)

def estimate_frequency(signal, fps):
    """Estima la frecuencia dominante en una señal."""
    fft_result = np.abs(fft.rfft(signal, axis=0))
    freqs = fft.rfftfreq(len(signal), d=1.0 / fps)
    return freqs[np.argmax(fft_result)] * 60

def process_buffer_evm(frame_buffer):
    """Procesa el buffer de frames usando EVM y extrae signos vitales."""
    pyramids = [build_laplacian_pyramid(frame, LEVELS) for frame in frame_buffer]
    level_signals = np.array([[np.mean(p[level]) for p in pyramids] 
                               for level in range(LEVELS)]).T
    
    heart_signal, resp_signal = apply_ica_pca(level_signals)

    heart_signal = apply_bandpass_filter(heart_signal, LOW_HEART, HIGH_HEART, FPS)
    resp_signal = apply_bandpass_filter(resp_signal, LOW_RESP, HIGH_RESP, FPS) 

    heart_signal *= ALPHA
    resp_signal *= ALPHA

    heart_rate = estimate_frequency(heart_signal, FPS)
    resp_rate = estimate_frequency(resp_signal, FPS)
    
    return heart_rate, resp_rate