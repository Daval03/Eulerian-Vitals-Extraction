#include "Utils_Process_Data.h"

deque<double> heart_rate_history;
deque<double> resp_rate_history;

double calculateMedian(deque<double>& data) {
    if (data.empty()) {
        return NAN; 
    }
    deque<double> sorted_data = data;
    sort(sorted_data.begin(), sorted_data.end());
    
    size_t size = sorted_data.size();
    if (size % 2 == 0) {
        return (sorted_data[size/2 - 1] + sorted_data[size/2]) / 2.0f;
    } else {
        return sorted_data[size/2];
    }
}

pair<double, double> applyStatisticalFilter(double hr, double rr) {
    // Agregar nuevos valores si no son NaN
    if (!isnan(hr)) {
        heart_rate_history.push_back(hr);
        // Mantener el tamaño máximo
        if (heart_rate_history.size() > Config::LIST_SIZE) {
            heart_rate_history.pop_front();
        }
    }
    
    if (!isnan(rr)) {
        resp_rate_history.push_back(rr);
        // Mantener el tamaño máximo
        if (resp_rate_history.size() > Config::LIST_SIZE) {
            resp_rate_history.pop_front();
        }
    }
    
    // Calcular medianas
    double filtered_hr = calculateMedian(heart_rate_history);
    double filtered_rr = calculateMedian(resp_rate_history);
    
    return make_pair(filtered_hr, filtered_rr);
}