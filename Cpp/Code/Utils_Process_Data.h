#ifndef UTILS_PROCESS_DATA_H
#define UTILS_PROCESS_DATA_H

#include <deque>
#include <utility>
#include <algorithm>
#include <cmath>
#include "Config.h"

using namespace std;

// Solo DECLARACIÓN (con extern)
extern deque<double> heart_rate_history;
extern deque<double> resp_rate_history;

pair<double, double> applyStatisticalFilter(double hr, double rr);
double calculateMedian(deque<double>& data);

#endif // UTILS_PROCESS_DATA_H