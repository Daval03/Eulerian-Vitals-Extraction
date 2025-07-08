#include "Config.h"
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
using namespace cv;
using namespace std;
using namespace dnn;


class Face_Detector{
    private:
    //Modell
    Net net;
    //Deteksjon
    Mat blob;
    Mat detections;
    Mat roi_frame;

    public:
    Face_Detector(){
        this->net = readNetFromCaffe(
            Config::DEPLOY_MODEL, 
            Config::CAFFE_MODEL
        );

    }
    Mat Get_ROI(Mat &frame){
        this->blob = blobFromImage(frame, 1.0, Size(Config::WINDOW_WIDTH, Config::WINDOW_HEIGHT), Scalar(104, 177, 123));
        this->net.setInput(blob);
        this->detections = net.forward();
        const int* detections_size = detections.size.p;
        Mat detection_mat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

        for (int i = 0; i < detection_mat.rows; i++) {
            float confidence = detection_mat.at<float>(i, 2);
            if (confidence > 0.5) {
                int x1 = static_cast<int>(detection_mat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detection_mat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detection_mat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detection_mat.at<float>(i, 6) * frame.rows);
                
                rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(0, 255, 0), 2);
                putText(frame, format("%.2f", confidence), Point(x1, y1 - 5), 
                           FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1); // Viser konfidensnivå på bildet
                Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
                this->roi_frame = frame(roi_rect).clone();
            }
        }
        return this->roi_frame;
    }
};