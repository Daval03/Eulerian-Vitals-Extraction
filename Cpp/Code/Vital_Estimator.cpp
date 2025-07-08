
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

    cout << "Prosesserer blokk med " << bloqueFrames.size() << " rammer" << endl;
}

int main() {
    // Opprett ansiktsdetektoren
    Face_Detector faceDetector;
    
    // Åpne video
    VideoCapture cap(Config::VIDEO_OUT);
    if (!cap.isOpened()) {
        cerr << "Feil ved åpning av video!" << endl;
        return -1;
    }

    vector<Mat> framesArray;
    framesArray.reserve(Config::MAX_FRAMES);
    pair<double, double> estimation;

    Mat frame;
    while (true) {
        if (!cap.read(frame)) {
            cerr << "Slutt på video eller lesefeil!" << endl;
            break;
        }

        // Bruk ansiktsdetektoren for å få ROI
        Mat roi_frame = faceDetector.Get_ROI(frame);
        
        // Legg bare til rammer med gyldig ROI (størrelse > 0)
        if (!roi_frame.empty()) {
            framesArray.push_back(roi_frame.clone());
            imshow("Ansiktsgjenkjenning", roi_frame); // Viser ansiktsgjenkjenning
        }

        // Prosesser når maksimalt antall rammer er nådd
        if (framesArray.size() >= Config::MAX_FRAMES) {

            estimation = processBufferEVM(framesArray);
            cout<<"HR: " << estimation.first << " RR: " << estimation.second; // HR: Hjertefrekvens, RR: Respirasjonsfrekvens

            framesArray.clear();
            cout << "Array tømt. Starter ny blokk..." << endl;
        }

        if (waitKey(1) == 27) break;  // Avslutt med ESC
    }

    // Prosesser gjenværende rammer hvis det finnes noen
    if (!framesArray.empty()) {
        procesarBloqueFrames(framesArray);
    }

    cap.release();
    destroyAllWindows();
    return 0;
}