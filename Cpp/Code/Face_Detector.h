#ifndef FACE_DETECTOR_H
#define FACE_DETECTOR_H

#include "Config.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace std;
using namespace dnn;

class Face_Detector {
private:
    // Model
    Net net;
    // Detección
    Mat blob;
    Mat detections;
    Mat roi_frame;

public:
    Face_Detector();
    ~Face_Detector();
    
    Mat Get_ROI(Mat &frame);
    void Release();  // Función para liberar los datos
};

#endif // FACE_DETECTOR_H