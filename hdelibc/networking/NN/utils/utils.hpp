#ifndef NN_UTILS_HPP
#define NN_UTILS_HPP

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

namespace NNUtils
{

    void initializeWeights(std::vector<std::vector<double>> &weights, double min_val, double max_val);
    void initializeBiases(std::vector<double> &biases, double initial_value = 0.0);

    namespace Activations
    {
        std::vector<double> softmax(const std::vector<double> &logits);
        double relu(double x);
        double reluDerivative(double x);
    }

}

#endif