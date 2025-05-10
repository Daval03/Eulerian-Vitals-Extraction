# Eulerian-Vitals-Extraction

# **Vital Signs Estimation using Eulerian Video Magnification (EVM)**  

## **📌 Description**  
This project implements a **non-invasive system** to estimate **heart rate (HR)** and **respiratory rate (RR)** using a **Raspberry Pi 4** and its camera. The system applies **Eulerian Video Magnification (EVM)** to amplify subtle facial color changes and movements, extracts physiological signals, and computes vital signs through signal processing.

The system consists of:  
✅ **Client-Server Architecture**: A Flask backend on the Raspberry Pi for video processing and a Tkinter GUI for remote visualization.  
✅ **Face Detection**: Uses **MediaPipe** to locate and stabilize the Region of Interest (ROI).  
✅ **Signal Processing**: Implements **EVM, ICA/PCA, and bandpass filtering** to extract heart and respiratory rates.  
✅ **Graphical Interface**: Displays real-time vital signs with start/stop controls.  

## **🎯 Objective**  
Develop a **low-cost, portable, and contactless** system for remote vital signs monitoring, suitable for **telemedicine, digital health, and patient monitoring applications**.  

---  

## **🛠️ Technologies & Hardware**  
### **🔹 Hardware**  
- **Raspberry Pi 4** (4GB RAM recommended)  
- **Raspberry Pi Camera (or compatible USB webcam)**  
- **Python 3.7+**  

### **🔹 Key Libraries**  
- **OpenCV** → Video processing and face detection.  
- **MediaPipe** → Face detection and tracking.  
- **Scipy & NumPy** → Signal processing and filtering.  
- **Flask** → Backend API server.  
- **Tkinter** → Client GUI.  

---  

## **⚙️ System Workflow**  

### **1️⃣ Video Capture**  
- The Raspberry Pi captures real-time video.  
- Each frame is preprocessed for contrast enhancement and noise reduction.  

### **2️⃣ Face Detection & ROI Stabilization**  
- **MediaPipe** detects the face and extracts the **Region of Interest (ROI)**.  
- The ROI is stabilized to minimize motion artifacts.  

### **3️⃣ Eulerian Video Magnification (EVM)**  
- **Laplacian pyramid decomposition** amplifies subtle color/motion changes.  
- **ICA (Independent Component Analysis)** and **PCA (Principal Component Analysis)** separate physiological signals from noise.  

### **4️⃣ Frequency Estimation**  
- Bandpass filtering isolates:  
  - **Heart rate (0.83 - 3.0 Hz ≈ 50 - 180 BPM)**  
  - **Respiratory rate (0.18 - 0.5 Hz ≈ 10 - 30 RPM)**  
- **FFT (Fast Fourier Transform)** extracts dominant frequencies.  

### **5️⃣ Real-Time Visualization**  
- The GUI displays:  
  - Heart rate (BPM) ❤️  
  - Respiratory rate (RPM) 🫁  
  - Optional live video stream.  

---  

## **🚀 Installation & Usage**  
### **Prerequisites**  
```bash
pip install opencv-python mediapipe scipy numpy flask flask-cors pillow
