from scipy.signal import butter, filtfilt

def create_bandpass_filter(lowcut, highcut, nyquist, order=2):
    """
    Create Butterworth bandpass filter.
    
    Args:
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        nyquist: Nyquist frequency (FPS/2)
        order: Filter order
    
    Returns:
        tuple: (b, a) filter coefficients or None if fails
    """
    try:
        low_norm = max(0.01, min(lowcut / nyquist, 0.99))
        high_norm = max(0.01, min(highcut / nyquist, 0.99))
        
        if low_norm < high_norm:
            b, a = butter(order, [low_norm, high_norm], btype='band')
            return b, a
        else:
            return None
    except Exception as e:
        print(f"[FILTER] Error creating filter: {e}")
        return None


def apply_temporal_bandpass(tensor, lowcut, highcut, fps, axis=0):
    """
    Apply temporal bandpass filter to video tensor.
    
    Args:
        tensor: Video tensor (T x H x W x C)
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fps: Frames per second
        axis: Temporal axis (default=0)
    
    Returns:
        np.ndarray: Filtered tensor
    """
    if len(tensor) < 10:
        return tensor
    
    nyquist = fps / 2.0
    filter_coeffs = create_bandpass_filter(lowcut, highcut, nyquist)
    
    if filter_coeffs is None:
        return tensor
    
    b, a = filter_coeffs
    
    try:
        filtered = filtfilt(b, a, tensor, axis=axis)
        return filtered
    except Exception as e:
        print(f"[FILTER] Error applying filter: {e}")
        return tensor


def temporal_dual_bandpass_filter(video_tensor, fps, 
                                  low_heart, high_heart,
                                  low_resp, high_resp, axis=0):
    """
    Apply TWO bandpass filters simultaneously to video tensor.
    More efficient than filtering twice separately.
    
    Args:
        video_tensor: Tensor (T x H x W x C)
        fps: Frames per second
        low_heart: Heart rate low frequency (Hz)
        high_heart: Heart rate high frequency (Hz)
        low_resp: Respiration rate low frequency (Hz)
        high_resp: Respiration rate high frequency (Hz)
        axis: Temporal axis
    
    Returns:
        tuple: (filtered_hr_tensor, filtered_rr_tensor)
    """
    if len(video_tensor) < 10:
        return video_tensor, video_tensor
    
    # Heart rate filtering
    filtered_hr = apply_temporal_bandpass(
        video_tensor, low_heart, high_heart, fps, axis
    )
    
    # Respiration rate filtering
    filtered_rr = apply_temporal_bandpass(
        video_tensor, low_resp, high_resp, fps, axis
    )
    
    return filtered_hr, filtered_rr