#include "Signal_Processing.h"

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

// Estimar frecuencia dominante usando FFTW3
double estimateFrequency(const VectorXd& signal, double fps) {
    int N = signal.size();
    double* in = fftw_alloc_real(N);
    fftw_complex* out = fftw_alloc_complex(N / 2 + 1);

    for (int i = 0; i < N; ++i) {
        in[i] = signal(i);
    }

    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    double maxAmp = 0.0;
    int maxIdx = 1; // Ignorar componente DC

    for (int i = 1; i < N / 2 + 1; ++i) {
        double magnitude = sqrt(out[i][0]*out[i][0] + out[i][1]*out[i][1]);
        if (magnitude > maxAmp) {
            maxAmp = magnitude;
            maxIdx = i;
        }
    }

    double freqHz = static_cast<double>(maxIdx) * fps / N;

    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return freqHz * 60.0; // BPM
}

// Filtro bandpass tipo Butterworth (simplificado)
VectorXd butterworthBandpassFilter(const VectorXd& input, double lowcut, double highcut, double fs, int order) {
    double nyquist = 0.5 * fs;
    double low = lowcut / nyquist;
    double high = highcut / nyquist;

    VectorXd output = input;
    int N = input.size();
    VectorXd temp(N);

    // Paso alto: media móvil de primer orden
    temp[0] = input[0];
    for (int i = 1; i < N; ++i) {
        temp[i] = input[i] - input[i - 1] + low * temp[i - 1];
    }

    // Paso bajo: suavizado exponencial
    output[0] = temp[0];
    for (int i = 1; i < N; ++i) {
        output[i] = high * temp[i] + (1.0 - high) * output[i - 1];
    }

    return output;
}

// Pipeline principal
pair<double, double> processBufferEVM(const vector<Mat>& frameBuffer) {
    int numFrames = frameBuffer.size();

    // 1. Construir pirámides Laplacianas
    vector<vector<Mat>> pyramids;
    for (const auto& frame : frameBuffer) {
        pyramids.push_back(buildLaplacianPyramid(frame, Config::LEVELS));
    }

    // 2. Extraer señales temporales
    MatrixXd levelSignals(numFrames, Config::LEVELS);
    for (int l = 0; l < Config::LEVELS; ++l) {
        for (int t = 0; t < numFrames; ++t) {
            Scalar meanVal = mean(pyramids[t][l]);
            levelSignals(t, l) = meanVal[0];
        }
    }

    // 3. PCA y amplificación
    VectorXd signal = applyPCA(levelSignals);
    signal *= Config::ALPHA;

    // 4. Filtros de banda para corazón y respiración
    VectorXd heart_filtered = butterworthBandpassFilter(signal, Config::LOW_HEART, Config::HIGH_HEART, Config::FPS);
    VectorXd resp_filtered = butterworthBandpassFilter(signal, Config::LOW_RESP, Config::HIGH_RESP, Config::FPS);

    // 5. Estimación de frecuencia
    double heart_rate = estimateFrequency(heart_filtered, Config::FPS);
    double resp_rate = estimateFrequency(resp_filtered, Config::FPS);

    return {heart_rate, resp_rate};
}
