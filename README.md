# EVM Vital Signs Monitor

A non-invasive vital signs monitoring system using **Eulerian Video Magnification (EVM)** and AI-based facial detection. Estimates heart rate (HR) and respiratory rate (RR) from video by amplifying subtle color and motion changes in facial regions.

## Overview

This system processes video frames to detect physiological signals without physical contact. Optimized for **Raspberry Pi 4**, it achieves real-time performance through efficient dual-band signal processing.

## Features

- **Non-invasive monitoring**: Extracts HR and RR from video feed
- **Multiple face detection models**: YOLOv8, MediaPipe, Haar Cascade, MTCNN
- **Optimized EVM pipeline**: Single-pass dual-band processing for simultaneous HR/RR extraction
- **Real-time capable**: Runs on Raspberry Pi 4 with ~30+ FPS detection
- **Comprehensive benchmarking**: Performance and accuracy metrics included

## How It Works

1. **Face Detection**: Locates facial ROI using selected AI model
2. **Pyramid Decomposition**: Builds Laplacian pyramids from video frames
3. **Temporal Filtering**: Applies bandpass filters for HR (0.8-3 Hz) and RR (0.2-0.8 Hz)
4. **Signal Amplification**: Magnifies subtle color variations
5. **Frequency Analysis**: Extracts dominant frequencies via FFT

## Requirements

```
opencv-python==4.10.0.84
numpy>=1.24.0,<2.0.0
scipy>=1.10.0,<2.0.0
mediapipe==0.10.21
ultralytics==8.3.235
mtcnn==1.0.0
tensorflow==2.16.1
```

See `requirements.txt` for complete dependencies.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

```

## Face Detection Models

Choose your preferred model in the code:

- **MediaPipe**: Fast, accurate, recommended for Raspberry Pi
- **YOLOv8n & YOLOv12n**: High accuracy, higher computational cost
- **Haar Cascade**: Lightweight, less accurate
- **MTCNN**: Robust to pose variations

## Performance

On Raspberry Pi 4:
- **Detection**: ~30-40 FPS (MediaPipe)
- **EVM Processing**: ~1-2 seconds per 200-frame chunk
- **End-to-End**: ~3-4 seconds per measurement
- **Accuracy**: MAE < 5 BPM for HR under optimal conditions

## Dataset & Ground Truth

This project uses the **UBFC-RPPG Dataset** for validation. The dataset includes:
- Video recordings at 30 fps (640x480 resolution)
- Ground truth PPG data from CMS50E pulse oximeter
- Varying indoor lighting conditions

The `ground_truth_handler` is an auxiliary function that extracts reference heart rate values from the UBFC-RPPG Dataset 2 for accuracy benchmarking.

**Dataset source**: [UBFC-RPPG Database](https://sites.google.com/view/ybenezeth/ubfcrppg)


## Configuration

Key parameters in `src/config.py`:

- `FPS`: Video frame rate (default: 30)
- `BUFFER_SIZE`: Frames per measurement (default: 200)
- `ALPHA_HR`: HR amplification factor (default: 20)
- `ALPHA_RR`: RR amplification factor (default: 10)
- `LEVELS_RPI`: Pyramid levels for Raspberry Pi (default: 3)

## Limitations

- Requires stable lighting conditions
- Subject must remain relatively still
- Performance degrades with rapid head movements
- RR detection less reliable than HR

## License

This project is for research and educational purposes.

## Acknowledgments

Based on Eulerian Video Magnification research and optimized for embedded systems deployment.