#include "utils.hpp"
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

void NNUtils::initializeWeights(std::vector<std::vector<double>> &weights, double min_val, double max_val)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min_val, max_val);
    for (auto &row : weights)
    {
        for (auto &w : row)
        {
            w = dist(gen);
        }
    }
}

void NNUtils::initializeBiases(std::vector<double> &biases, double initial_value)
{
    std::fill(biases.begin(), biases.end(), initial_value);
}

double NNUtils::Activations::relu(double x)
{
    return std::max(0.0, x);
}

double NNUtils::Activations::reluDerivative(double x)
{
    return (x > 0.0) ? 1.0 : 0.0;
}

std::vector<double> NNUtils::Activations::softmax(const std::vector<double> &logits)
{
    double max_val = *std::max_element(logits.begin(), logits.end());
    std::vector<double> exp_values(logits.size());
    double sum_exp = 0.0;
    for (size_t i = 0; i < logits.size(); ++i)
    {
        exp_values[i] = exp(logits[i] - max_val);
        sum_exp += exp_values[i];
    }
    for (size_t i = 0; i < logits.size(); ++i)
    {
        exp_values[i] /= sum_exp;
    }
    return exp_values;
}
