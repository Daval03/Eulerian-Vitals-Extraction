import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from src.config import LEVELS_RPI

def build_gaussian_pyramid(frame, levels=LEVELS_RPI):
    """
    Build Gaussian pyramid optimized for RP4 processing.
    
    Args:
        frame: Input frame (BGR format)
        levels: Number of pyramid levels
    
    Returns:
        list: Gaussian pyramid (list of downsampled frames)
    """
    pyramid = []
    try:
        current = frame.astype(np.float32)
        pyramid.append(current)
        
        for _ in range(levels):
            current = cv2.pyrDown(current)
            pyramid.append(current)
    
    except Exception as e:
        print(f"[PYRAMID] Error building Gaussian pyramid: {e}")
        # Return whatever we have built so far
        if not pyramid:
            pyramid = [frame.astype(np.float32)]
    
    return pyramid


def build_laplacian_pyramid(gaussian_pyramid):
    """
    Build Laplacian pyramid from Gaussian pyramid.
    
    Args:
        gaussian_pyramid: Pre-built Gaussian pyramid
    
    Returns:
        list: Laplacian pyramid
    """
    laplacian_pyramid = []
    if not gaussian_pyramid:
        return laplacian_pyramid
    
    try:
        for i in range(len(gaussian_pyramid) - 1):
            size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
            laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
            laplacian_pyramid.append(laplacian)
        
        # The last level of Laplacian pyramid is the same as the last Gaussian level
        laplacian_pyramid.append(gaussian_pyramid[-1])
    
    except Exception as e:
        print(f"[PYRAMID] Error building Laplacian pyramid: {e}")
        # Return empty pyramid on error
        laplacian_pyramid = []
    
    return laplacian_pyramid


def collapse_laplacian_pyramid(laplacian_pyramid):
    """
    Reconstruct image from Laplacian pyramid.
    
    Args:
        laplacian_pyramid: Laplacian pyramid
    
    Returns:
        np.ndarray: Reconstructed image
    """
    if not laplacian_pyramid:
        return np.array([], dtype=np.float32)
    try:
        current = laplacian_pyramid[-1]
        for i in range(len(laplacian_pyramid) - 2, -1, -1):
            size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
            expanded = cv2.pyrUp(current, dstsize=size)
            current = cv2.add(expanded, laplacian_pyramid[i])
        
        return current
    
    except Exception as e:
        print(f"[PYRAMID] Error collapsing pyramid: {e}")
        # Return a zero array with appropriate shape
        if laplacian_pyramid and len(laplacian_pyramid) > 0:
            return np.zeros_like(laplacian_pyramid[0])
        return np.array([], dtype=np.float32)


def build_video_pyramid_stack(video_frames, levels=LEVELS_RPI):
    """
    Build Laplacian pyramids for all video frames.
    
    Args:
        video_frames: List of video frames
        levels: Number of pyramid levels
    
    Returns:
        list: List of Laplacian pyramids (one per frame)
    """
    laplacian_pyramids = []
    
    for frame_idx, frame in enumerate(video_frames):
        try:
            gaussian_pyr = build_gaussian_pyramid(frame, levels=levels)
            laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
            
            if laplacian_pyr:
                laplacian_pyramids.append(laplacian_pyr)
            else:
                # If pyramid building fails, add a single-level pyramid
                laplacian_pyramids.append([frame.astype(np.float32)])
        
        except Exception as e:
            print(f"[PYRAMID] Error processing frame {frame_idx}: {e}")
            laplacian_pyramids.append([frame.astype(np.float32)])
    
    return laplacian_pyramids


def extract_pyramid_level(pyramid_stack, level):
    """
    Extract specific level from all pyramids and normalize dimensions.
    
    Args:
        pyramid_stack: Stack of Laplacian pyramids
        level: Pyramid level to extract
    
    Returns:
        np.ndarray: Tensor (T x H x W x C) of specified pyramid level
    """
    if not pyramid_stack:
        return np.array([], dtype=np.float32)
    
    try:
        level_frames = []
        
        for pyr in pyramid_stack:
            if level < len(pyr):
                level_frames.append(pyr[level])
            else:
                # If level doesn't exist, use the last available level
                level_frames.append(pyr[-1])
        
        # Determine target shape (most common shape)
        shapes = [frame.shape for frame in level_frames]
        unique_shapes = {}
        for shape in shapes:
            unique_shapes[shape] = unique_shapes.get(shape, 0) + 1
        
        if unique_shapes:
            target_shape = max(unique_shapes.items(), key=lambda x: x[1])[0]
        else:
            target_shape = level_frames[0].shape
        
        level_resized = []
        for frame in level_frames:
            if frame.shape != target_shape:
                frame = cv2.resize(frame, (target_shape[1], target_shape[0]))
            level_resized.append(frame)
        
        return np.array(level_resized, dtype=np.float32)
    
    except Exception as e:
        print(f"[PYRAMID] Error extracting level {level}: {e}")
        return np.array([], dtype=np.float32)