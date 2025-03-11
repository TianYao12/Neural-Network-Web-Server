#ifndef CONV_NEURAL_NET_HPP
#define CONV_NEURAL_NET_HPP

#include <vector>
#include <cstdint>
#include <string>
#include <utility>

class CNN
{
    int inputWidth;
    int inputHeight;

    struct ConvLayer
    {
        int numFilters;
        int kernelSize;
        int stride;
        int outputWidth;
        int outputHeight;

        std::vector<std::vector<std::vector<double>>> filters; // [filter index][row][col]
        std::vector<double> biases;
    };
    std::vector<ConvLayer> convLayers;

    struct PoolingLayer
    {
        int poolSize;
        int stride;
        int outputWidth;
        int outputHeight;
    };
    std::vector<PoolingLayer> poolLayers;

    int fcInputSize; // # inputs to fully connected layer (flattened convolutional outputs)
    int outputSize;
    std::vector<std::vector<double>> fcWeights;
    std::vector<double> fcBiases;

    std::vector<std::vector<double>> convolve2D(const std::vector<std::vector<double>> &image,
                                                const std::vector<std::vector<double>> &filter,
                                                double bias, int stride);

    std::vector<std::vector<double>> maxPool2D(const std::vector<std::vector<double>> &featureMap,
                                               int poolSize, int stride);

    std::vector<double> flatten(const std::vector<std::vector<std::vector<double>>> &convOutputs);

    std::vector<double> computeFCActivation(const std::vector<double> &input);

public:
    CNN(int inputWidth,
        int inputHeight,
        const std::vector<std::pair<int, int>> &convSpecs, // {numFilters, kernelSize} per layer
        const std::vector<int> &poolSizes,                 // Pool size per layer
        int outputSize,
        int convStride = 1,
        int poolStride = 2);

    std::vector<double> performForwardPass(const std::vector<uint8_t> &input);

    void train(const std::vector<std::vector<uint8_t>> &images,
               const std::vector<uint8_t> &labels,
               int epochs,
               double learningRate,
               double momentum = 0.9);

    void saveFinalWeights(const std::string &filename);

    void loadPretrainedWeights(const std::string &filename);
};

#endif 
