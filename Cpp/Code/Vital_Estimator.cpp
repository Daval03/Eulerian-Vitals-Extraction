
#include "Config.h"
#include "Signal_Processing.cpp"
#include "Face_Detector.cpp"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
using namespace cv;
using namespace std;
using namespace dnn;

void procesarBloqueFrames(const vector<Mat>& bloqueFrames) {

    cout << "Procesando bloque de " << bloqueFrames.size() << " frames" << endl;
}

int main() {
    // Crear el detector de caras
    Face_Detector faceDetector;
    
    // Abrir video
    VideoCapture cap(Config::VIDEO_OUT);
    if (!cap.isOpened()) {
        cerr << "Error al abrir el video!" << endl;
        return -1;
    }

    vector<Mat> framesArray;
    framesArray.reserve(Config::MAX_FRAMES);
    pair<double, double> estimation;

    Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            cerr << "Fin del video o error de lectura!" << endl;
            break;
        }

        // Usar el detector de caras para obtener ROI
        Mat roi_frame = faceDetector.Get_ROI(frame);
        
        // Solo agregar frames con ROI válida (tamaño > 0)
        if (!roi_frame.empty()) {
            framesArray.push_back(roi_frame.clone());
            imshow("Face Detection", roi_frame);
        }

        // Procesar cuando se alcanza el máximo de frames
        if (framesArray.size() >= Config::MAX_FRAMES) {

            estimation = processBufferEVM(framesArray);
            cout<<"HR: " << estimation.first << "RR: " << estimation.second;

            framesArray.clear();
            cout << "Array vaciado. Comenzando nuevo bloque..." << endl;
        }

        if (waitKey(1) == 27) break;  // Salir con ESC
    }

    // Procesar los frames restantes si los hay
    if (!framesArray.empty()) {
        procesarBloqueFrames(framesArray);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}