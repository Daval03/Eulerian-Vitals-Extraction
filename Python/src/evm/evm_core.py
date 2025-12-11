import sys
import os
import numpy as np
sys.path.append(os.path.dirname(__file__))

from pyramid_processing import (
    build_video_pyramid_stack, 
    extract_pyramid_level
)
from temporal_filtering import apply_temporal_bandpass
from signal_analysis import extract_temporal_signal
from src.config import (
    FPS, LOW_HEART, HIGH_HEART, LOW_RESP, HIGH_RESP, ALPHA_HR, ALPHA_RR, LEVELS_RPI
)

class EVMProcessor:
    """
    Processor for Eulerian Video Magnification (EVM) with optimized configuration.
    
    Performs single-pass dual-band processing:
    - Builds Laplacian pyramids once
    - Applies two separate temporal filters (HR and RR frequency bands)
    - Amplifies each band with its corresponding alpha factor
    - Extracts both heart rate (HR) and respiratory rate (RR) temporal signals
    """
    
    def __init__(self, levels=LEVELS_RPI, alpha_hr=ALPHA_HR, alpha_rr=ALPHA_RR):
        """
        Initialize EVM processor.
        
        Args:
            levels: Number of pyramid levels (default from config)
            alpha_hr: Amplification factor for heart rate band
            alpha_rr: Amplification factor for respiratory rate band
        """
        self.levels = levels
        self.alpha_hr = alpha_hr
        self.alpha_rr = alpha_rr
    
    def process_dual_band(self, video_frames):
        """
        Single-pass EVM pipeline that extracts TWO temporal signals.
        
        Key innovation:
        - Builds Laplacian pyramids once
        - Applies two temporal filters to different pyramid levels
        - Amplifies each band separately
        - Returns both magnified signals
        
        Process flow:
        1. Build Laplacian pyramid stack from video frames
        2. Select optimal pyramid levels for HR and RR signals
        3. Extract tensor data for each level
        4. Apply bandpass temporal filtering (HR: 0.8-3 Hz, RR: 0.2-0.8 Hz)
        5. Amplify each filtered band
        6. Extract temporal signals (HR: green channel, RR: all channels average)
        
        Args:
            video_frames: List of ROI video frames in BGR format
            
        Returns:
            tuple: (hr_signal, rr_signal) - Magnified temporal signals
                   Returns (None, None) if video has fewer than 30 frames
        """
        if video_frames is None:
            return None, None
    
        if not isinstance(video_frames, (list, np.ndarray)):
            return None, None
        
        if len(video_frames) < 30:
            return None, None
        
        # STEP 1: Build Laplacian pyramids (SINGLE PASS)
        laplacian_pyramids = build_video_pyramid_stack(
            video_frames, levels=self.levels
        )
        
        num_levels = len(laplacian_pyramids[0])
        
        # STEP 2: Select optimal pyramid level for each signal
        level_hr = min(3, num_levels - 1)  # HR: level 3 (higher spatial freq)
        level_rr = min(2, num_levels - 1)  # RR: level 2 (lower spatial freq)
        
        # STEP 3: Extract tensor data from each pyramid level
        tensor_hr = extract_pyramid_level(laplacian_pyramids, level_hr)
        tensor_rr = extract_pyramid_level(laplacian_pyramids, level_rr)
        
        # STEP 4: Separate temporal filtering per frequency band
        filtered_tensor_hr = apply_temporal_bandpass(
            tensor_hr, LOW_HEART, HIGH_HEART, FPS, axis=0  # HR band: 0.8-3 Hz
        )
        
        filtered_tensor_rr = apply_temporal_bandpass(
            tensor_rr, LOW_RESP, HIGH_RESP, FPS, axis=0   # RR band: 0.2-0.8 Hz
        )
        
        # STEP 5: Signal amplification
        filtered_tensor_hr *= self.alpha_hr
        filtered_tensor_rr *= self.alpha_rr
        
        # STEP 6: Extract temporal signals
        # HR: Green channel (best SNR for pulse)
        signal_hr = extract_temporal_signal(filtered_tensor_hr, use_green_channel=True)
        
        # RR: All channels average
        signal_rr = extract_temporal_signal(filtered_tensor_rr, use_green_channel=True)
        
        return signal_hr, signal_rr