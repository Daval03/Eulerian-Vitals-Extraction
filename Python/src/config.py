
# ==================== HEART RATE (HR) PARAMETERS ====================
ALPHA_HR = 30        # Amplification factor for heart rate magnification
LOW_HEART = 0.83     # Lower frequency bound for heart rate bandpass filter (Hz)
HIGH_HEART = 3.0     # Upper frequency bound for heart rate bandpass filter (Hz)
MAX_HEART_BPM = 250  # Maximum physically plausible heart rate (beats per minute)
MIN_HEART_BPM = 40   # Minimum physically plausible heart rate (beats per minute)

# ==================== RESPIRATORY RATE (RR) PARAMETERS ====================
ALPHA_RR = 50        # Amplification factor for respiratory rate magnification
LOW_RESP = 0.18      # Lower frequency bound for respiratory rate bandpass filter (Hz)
HIGH_RESP = 0.5      # Upper frequency bound for respiratory rate bandpass filter (Hz)
MAX_RESP_BPM = 35    # Maximum physically plausible respiratory rate (breaths per minute)
MIN_RESP_BPM = 8     # Minimum physically plausible respiratory rate (breaths per minute)

# ==================== ROI (Region of Interest) PARAMETERS ====================
LEVELS_RPI = 3       # Number of levels for ROI pyramid processing
ROI_PADDING = 10     # Padding around detected face for ROI extraction

# ==================== VIDEO CAPTURE CONFIGURATION ====================
FPS = 30             # Frames per second (assumed video frame rate)
# TARGET_ROI_SIZE = (640, 480)  # Alternative ROI size
TARGET_ROI_SIZE = (320, 240)  # Target size for ROI processing (width, height)

# ==================== YOLO MODEL PATHS ====================
YOLO_MODELS = {
    "yolov8n": "src/weights_models/yolov8n-face.pt",   # YOLOv8 nano face detection model
    "yolov12n": "src/weights_models/yolov12n-face.pt"  # YOLOv12 nano face detection model
}

# ==================== ROI STABILIZATION PARAMETERS ====================
ROI_CHANGE_THRESHOLD = 20  # Threshold for significant ROI position change (pixels)
ROI_WEIGHTS = [0.1, 0.15, 0.2, 0.25, 0.3]  # Weights for ROI smoothing filter