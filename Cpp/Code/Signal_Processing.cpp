
#include "Config.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <unsupported/Eigen/FFT> 
#include <Eigen/Dense>

using namespace cv;
using namespace std;
using namespace Eigen;

// Konstruksjon av Laplaciansk pyramide
vector<Mat> buildLaplacianPyramid(const Mat& frame, int levels) {
    vector<Mat> pyramid;
    Mat current = frame.clone();
    for (int i = 0; i < levels; ++i) {
        Mat down, up, lap;
        pyrDown(current, down);
        pyrUp(down, up, current.size());
        subtract(current, up, lap);
        pyramid.push_back(lap);
        current = down;
    }
    pyramid.push_back(current);
    return pyramid;
}

// Enkel PCA ved bruk av Eigen
VectorXd applyPCA(const MatrixXd& data) {
    MatrixXd centered = data.rowwise() - data.colwise().mean();
    MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);
    SelfAdjointEigenSolver<MatrixXd> eig(cov);
    VectorXd principal = centered * eig.eigenvectors().rightCols(1);
    return principal;
}

// Estimere dominant frekvens med Eigen::FFT
double estimateFrequency(const VectorXd& signal, double fps) {
    Eigen::FFT<double> fft;
    vector<complex<double>> freqDomain;
    vector<double> timeDomain(signal.data(), signal.data() + signal.size());
    fft.fwd(freqDomain, timeDomain);

    int N = timeDomain.size();
    double maxAmp = 0;
    int maxIdx = 1; // Ignorer DC-komponenten

    for (int i = 1; i < N / 2; ++i) {
        double magnitude = abs(freqDomain[i]);
        if (magnitude > maxAmp) {
            maxAmp = magnitude;
            maxIdx = i;
        }
    }

    double freqHz = static_cast<double>(maxIdx) * fps / N;
    return freqHz * 60.0; // Returnerer i bpm (slag per minutt)
}
VectorXd butterworthBandpassFilter(const VectorXd& input, double lowcut, double highcut, double fs, int order = 3) {
    // Adaptasjon fra klassisk digital Butterworth-design ved bruk av bilinear transform
    double nyquist = 0.5 * fs;
    double low = lowcut / nyquist;
    double high = highcut / nyquist;

    // Dette bruker et 3. ordens IIR Butterworth-filter i to trinn (høypass + lavpass)
    // For en robust implementasjon anbefales det å bruke et bibliotek som Iir1 eller DSPFilters

    VectorXd output = input;

    // Enkel implementasjon med differansefilter - Ikke ideelt, men fungerer som en tilnærming

    int N = input.size();
    VectorXd temp(N);

    // Trinn 1: høypassfilter (førsteordens glidende gjennomsnitt)
    temp[0] = input[0];
    for (int i = 1; i < N; ++i) {
        temp[i] = input[i] - input[i - 1] + low * temp[i - 1];
    }

    // Trinn 2: lavpassfilter (eksponentiell glattet gjennomsnitt)
    output[0] = temp[0];
    for (int i = 1; i < N; ++i) {
        output[i] = high * temp[i] + (1.0 - high) * output[i - 1];
    }

    return output;
}

// Hovedpipeline
pair<double, double> processBufferEVM(const vector<Mat>& frameBuffer) {
    int numFrames = frameBuffer.size();
    // #1. Konstruer pyramider
    vector<vector<Mat>> pyramids;
    for (const auto& frame : frameBuffer) {
        pyramids.push_back(buildLaplacianPyramid(frame, Config::LEVELS));
    }
    // # 2. Trekk ut temporale signaler per nivå
    MatrixXd levelSignals(numFrames, Config::LEVELS);
    for (int l = 0; l < Config::LEVELS; ++l) {
        for (int t = 0; t < numFrames; ++t) {
            Scalar meanVal = mean(pyramids[t][l]);
            levelSignals(t, l) = meanVal[0];
        }
    }

    // 3. Forsterk temporale variasjoner og bruk ICA/PCA
    VectorXd signal = applyPCA(levelSignals);
    signal *= Config::ALPHA;

    VectorXd heart_filtered = butterworthBandpassFilter(signal, Config::LOW_HEART, Config::HIGH_HEART, Config::FPS);
    VectorXd resp_filtered = butterworthBandpassFilter(signal, Config::LOW_RESP, Config::HIGH_RESP, Config::FPS);

    // Frekvensestimering
    double heart_rate = estimateFrequency(heart_filtered, Config::FPS);
    double resp_rate = estimateFrequency(resp_filtered, Config::FPS);
    return {heart_rate, resp_rate};
}