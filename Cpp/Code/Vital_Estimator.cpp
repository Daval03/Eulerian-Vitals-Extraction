#include "Config.h"
#include "Signal_Processing.h"
#include "Face_Detector.h"
#include "Utils_Process_Data.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <deque>

using namespace cv;
using namespace std;
using namespace dnn;

int main() {

    // Crear el detector de caras
    Face_Detector face_detector;

    // Abrir video
    VideoCapture cap(Config::VIDEO_OUT);
    if (!cap.isOpened()) {
        cerr << "Error al abrir el video!" << endl;
        return -1;
    }

    vector<Mat> framesArray;
    framesArray.reserve(Config::MAX_FRAMES);
    pair<double, double> estimation, estimation_filter;

    Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            cerr << "Fin del video o error de lectura!" << endl;
            break;
        }
        // Usar el detector de caras para obtener ROI
        Mat roi_frame = face_detector.Get_ROI(frame);
        
        // Solo agregar frames con ROI válida (tamaño > 0)
        if (!roi_frame.empty()) {
            framesArray.push_back(roi_frame.clone());
            imshow("Face Detection", roi_frame);
        }

        // Procesar cuando se alcanza el máximo de frames
        if (framesArray.size() >= Config::MAX_FRAMES) {

            estimation = processBufferEVM(framesArray);
            estimation_filter = applyStatisticalFilter(estimation.first, estimation.second);
            cout<<"|------- \n";
            cout<<"HR: " << estimation.first << " RR: " << estimation.second << "\n";
            cout<<"F- HR: " << estimation_filter.first << "F- RR: " << estimation_filter.second << "\n";
            cout<<"|------- \n";

            framesArray.clear();

        }

        if (waitKey(1) == 27) break;  // Salir con ESC
    }
    cap.release();
    face_detector.Release();
    destroyAllWindows();
    return 0;
}