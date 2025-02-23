#ifndef FF_NEURAL_NET_HPP
#define FF_NEURAL_NET_HPP

#include <vector>
#include <functional>
#include "../utils/utils.hpp"

class FFNeuralNet
{
    std::vector<std::vector<double>> inputToHiddenLayerWeights;
    std::vector<std::vector<double>> hiddenToOutputLayerWeights;
    std::vector<double> hiddenLayerBiases;
    std::vector<double> outputLayerBiases;

    std::vector<double> computeLayerActivation(
        const std::vector<double> &input,
        const std::vector<std::vector<double>> &weights,
        std::vector<double> &biases,
        std::function<double(double)> activationFunction);

    void applyBackpropagation(
        const std::vector<double> &inputNormalized,
        const std::vector<double> &hiddenToOutputLayerActivation,
        const std::vector<double> &outputLayerProbability,
        int actualLabel,
        double learningRate);

    std::vector<double> extractNetworkParameters() const;

public:
    FFNeuralNet(int inputSize, int hiddenSize, int outputSize);
    std::vector<double> performForwardPass(const std::vector<uint8_t> &input);

    void train(
        const std::vector<std::vector<uint8_t>> &images,
        const std::vector<uint8_t> &labels,
        int epochs, double learningRate);

    void saveFinalWeights(const std::string &fileName);
    void loadPretrainedWeights(const std::string &filename);
};

#endif