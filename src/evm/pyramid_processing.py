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
    pyramid = [frame.astype(np.float32)]
    current = frame.astype(np.float32)
    
    for _ in range(levels):
        current = cv2.pyrDown(current)
        pyramid.append(current)
    
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
    
    for i in range(len(gaussian_pyramid) - 1):
        size = (gaussian_pyramid[i].shape[1], gaussian_pyramid[i].shape[0])
        expanded = cv2.pyrUp(gaussian_pyramid[i + 1], dstsize=size)
        laplacian = cv2.subtract(gaussian_pyramid[i], expanded)
        laplacian_pyramid.append(laplacian)
    
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def collapse_laplacian_pyramid(laplacian_pyramid):
    """
    Reconstruct image from Laplacian pyramid.
    
    Args:
        laplacian_pyramid: Laplacian pyramid
    
    Returns:
        np.ndarray: Reconstructed image
    """
    current = laplacian_pyramid[-1]
    
    for i in range(len(laplacian_pyramid) - 2, -1, -1):
        size = (laplacian_pyramid[i].shape[1], laplacian_pyramid[i].shape[0])
        expanded = cv2.pyrUp(current, dstsize=size)
        current = cv2.add(expanded, laplacian_pyramid[i])
    
    return current


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
    
    for frame in video_frames:
        gaussian_pyr = build_gaussian_pyramid(frame, levels=levels)
        laplacian_pyr = build_laplacian_pyramid(gaussian_pyr)
        laplacian_pyramids.append(laplacian_pyr)
    
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
    level_frames = [pyr[level] for pyr in pyramid_stack]
    target_shape = level_frames[0].shape
    
    level_resized = []
    for frame in level_frames:
        if frame.shape != target_shape:
            frame = cv2.resize(frame, (target_shape[1], target_shape[0]))
        level_resized.append(frame)
    
    return np.array(level_resized, dtype=np.float32)