#include "utils.hpp"
#include <random>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std;

void NNUtils::initializeWeights(vector<vector<double>> &weights, double min_val, double max_val)
{
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<double> dist(min_val, max_val);
    for (auto &row : weights)
    {
        for (auto &w : row)
        {
            w = dist(gen);
        }
    }
}

void NNUtils::initializeBiases(vector<double> &biases, double initial_value)
{
    fill(biases.begin(), biases.end(), initial_value);
}

static double NNUtils::randomDouble(double min_val, double max_val) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(min_val, max_val);
    return dist(gen);
}

double NNUtils::ActivationFunctions::relu(double x)
{
    return max(0.0, x);
}

double NNUtils::ActivationFunctions::reluDerivative(double x)
{
    return (x > 0.0) ? 1.0 : 0.0;
}

std::vector<double> NNUtils::ActivationFunctions::softmax(const std::vector<double> &logits)
{
    double max_val = *std::max_element(logits.begin(), logits.end()); 
    std::vector<double> exp_values(logits.size());
    double sum_exp = 0.0;

    for (size_t i = 0; i < logits.size(); ++i)
    {
        exp_values[i] = std::exp(logits[i] - max_val); 
        sum_exp += exp_values[i];
    }

    for (size_t i = 0; i < logits.size(); ++i)
    {
        exp_values[i] /= sum_exp;
    }

    return exp_values;
}