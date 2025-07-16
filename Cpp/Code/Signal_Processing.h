#ifndef SIGNAL_PROCESSING_H
#define SIGNAL_PROCESSING_H

#include "Config.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <fftw3.h> 

using namespace cv;
using namespace std;
using namespace Eigen;

vector<Mat> buildLaplacianPyramid(const Mat& frame, int levels);
VectorXd applyPCA(const MatrixXd& data);
VectorXd butterworthBandpassFilter(const VectorXd& input, double lowcut, double highcut, double fs, int order = 3);
double estimateFrequency(const VectorXd& signal, double fps);
pair<double, double> processBufferEVM(const vector<Mat>& frameBuffer);

#endif // SIGNAL_PROCESSING_H
