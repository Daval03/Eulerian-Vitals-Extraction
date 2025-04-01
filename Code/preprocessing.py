import cv2
import numpy as np

def preprocess_frame(frame):
    """
    Preprocesa un frame usando transformación HSV y ecualización de histograma.
    
    Args:
        frame (numpy.ndarray): Frame de entrada en formato BGR
    
    Returns:
        numpy.ndarray: Frame preprocesado
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    return cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

def postprocess_frame(frame):
    """
    Extrae el canal Y de la transformación YCrCb.
    
    Args:
        frame (numpy.ndarray): Frame de entrada
    
    Returns:
        numpy.ndarray: Canal Y como float32
    """
    frame_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    y_channel = frame_ycrcb[:, :, 0].astype(np.float32)
    return y_channel