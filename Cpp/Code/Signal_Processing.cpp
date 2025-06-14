
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

// Construcción de pirámide Laplaciana
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

// PCA simple usando Eigen
VectorXd applyPCA(const MatrixXd& data) {
    MatrixXd centered = data.rowwise() - data.colwise().mean();
    MatrixXd cov = (centered.adjoint() * centered) / double(data.rows() - 1);
    SelfAdjointEigenSolver<MatrixXd> eig(cov);
    VectorXd principal = centered * eig.eigenvectors().rightCols(1);
    return principal;
}

// Estimar frecuencia dominante con Eigen::FFT
double estimateFrequency(const VectorXd& signal, double fps) {
    Eigen::FFT<double> fft;
    vector<complex<double>> freqDomain;
    vector<double> timeDomain(signal.data(), signal.data() + signal.size());
    fft.fwd(freqDomain, timeDomain);

    int N = timeDomain.size();
    double maxAmp = 0;
    int maxIdx = 1; // Ignora la componente DC

    for (int i = 1; i < N / 2; ++i) {
        double magnitude = abs(freqDomain[i]);
        if (magnitude > maxAmp) {
            maxAmp = magnitude;
            maxIdx = i;
        }
    }

    double freqHz = static_cast<double>(maxIdx) * fps / N;
    return freqHz * 60.0; // Retorna en bpm
}
VectorXd butterworthBandpassFilter(const VectorXd& input, double lowcut, double highcut, double fs, int order = 3) {
    // Adaptado de diseño digital clásico Butterworth usando bilinear transform
    double nyquist = 0.5 * fs;
    double low = lowcut / nyquist;
    double high = highcut / nyquist;

    // Esto usa un filtro IIR Butterworth de orden 3 en dos pasos (paso alto + paso bajo)
    // Para una implementación robusta se recomienda usar una librería como Iir1 o DSPFilters

    VectorXd output = input;

    // Simple implementación con filtro de diferencia - No es ideal pero funciona como aproximación

    int N = input.size();
    VectorXd temp(N);

    // Paso 1: filtro pasa alto (media móvil de primer orden)
    temp[0] = input[0];
    for (int i = 1; i < N; ++i) {
        temp[i] = input[i] - input[i - 1] + low * temp[i - 1];
    }

    // Paso 2: filtro pasa bajo (media exponencial suavizada)
    output[0] = temp[0];
    for (int i = 1; i < N; ++i) {
        output[i] = high * temp[i] + (1.0 - high) * output[i - 1];
    }

    return output;
}

// Pipeline principal
pair<double, double> processBufferEVM(const vector<Mat>& frameBuffer) {
    int numFrames = frameBuffer.size();
    // #1. Construir pirámides
    vector<vector<Mat>> pyramids;
    for (const auto& frame : frameBuffer) {
        pyramids.push_back(buildLaplacianPyramid(frame, Config::LEVELS));
    }
    // # 2. Extraer señales temporales por nivel
    MatrixXd levelSignals(numFrames, Config::LEVELS);
    for (int l = 0; l < Config::LEVELS; ++l) {
        for (int t = 0; t < numFrames; ++t) {
            Scalar meanVal = mean(pyramids[t][l]);
            levelSignals(t, l) = meanVal[0];
        }
    }

    // 3. Amplificar las variaciones temporales y aplicar ICA/PCA 
    VectorXd signal = applyPCA(levelSignals);
    signal *= Config::ALPHA;

    VectorXd heart_filtered = butterworthBandpassFilter(signal, Config::LOW_HEART, Config::HIGH_HEART, Config::FPS);
    VectorXd resp_filtered = butterworthBandpassFilter(signal, Config::LOW_RESP, Config::HIGH_RESP, Config::FPS);

    // Estimación de frecuencia
    double heart_rate = estimateFrequency(heart_filtered, Config::FPS);
    double resp_rate = estimateFrequency(resp_filtered, Config::FPS);
    return {heart_rate, resp_rate};
}