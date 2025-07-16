#include "Face_Detector.h"

Face_Detector::Face_Detector() {
    this->net = readNetFromCaffe(
        Config::DEPLOY_MODEL, 
        Config::CAFFE_MODEL
    );
}

Face_Detector::~Face_Detector() {
    Release();  // Llama a Release() en el destructor
}

Mat Face_Detector::Get_ROI(Mat &frame) {
    this->blob = blobFromImage(frame, 1.0, Size(), Scalar(104, 177, 123));
    this->net.setInput(blob);
    this->detections = net.forward();
    const int* detections_size = detections.size.p;
    Mat detection_mat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

    for (int i = 0; i < detection_mat.rows; i++) {
        float confidence = detection_mat.at<float>(i, 2);
        if (confidence > 0.8) {
            int x1 = static_cast<int>(detection_mat.at<float>(i, 3) * frame.cols);
            int y1 = static_cast<int>(detection_mat.at<float>(i, 4) * frame.rows);
            int x2 = static_cast<int>(detection_mat.at<float>(i, 5) * frame.cols);
            int y2 = static_cast<int>(detection_mat.at<float>(i, 6) * frame.rows);
            
            rectangle(frame, Point(x1, y1), Point(x2, y2), Scalar(680, 480, 0), 2);
            putText(frame, format("%.2f", confidence), Point(x1, y1 - 5), 
                       FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 1);
            Rect roi_rect(x1, y1, x2 - x1, y2 - y1);
            Mat roi = frame(roi_rect).clone();
            resize(roi, this->roi_frame, Size(Config::WINDOW_WIDTH, Config::WINDOW_HEIGHT));
            return this->roi_frame;
        }
    }
    return Mat();
}

void Face_Detector::Release() {
    // Libera las matrices
    this->blob.release();
    this->detections.release();
    this->roi_frame.release();
}