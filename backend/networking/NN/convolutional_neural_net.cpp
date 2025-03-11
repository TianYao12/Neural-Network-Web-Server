#include "convolutional_neural_net.hpp"
#include "./utils/utils.hpp"
#include <vector>
#include <cstdint>
#include <iostream>
#include <fstream>
#include <cmath>
#include <random>
#include <algorithm>
#include <limits>

/**
 * @brief Constructs CNN with specified input dimensions and layer configs
 *
 * @param inputWidth Input Int: input width
 * @param inputHeight Input Int: input height
 * @param filterSpecs Layer Vector of pairs { numFilters, kernelSize }: filter specs for each convolutional layer
 * @param poolSizes Layer Vector: pool size for each convolutional layer
 * @param outputSize Layer Int: # output neurons in fully connected layer.
 * @param convStride Layer Int: Stride used in convolution operations (default 1)
 * @param poolStride Layer Int: Stride used in pooling operations (default 2)
 */
CNN::CNN(int inputWidth, int inputHeight,
         const std::vector<std::pair<int, int>> &filterSpecs, const std::vector<int> &poolSizes,
         int outputSize, int convStride, int poolStride)
    : inputWidth(inputWidth), inputHeight(inputHeight), outputSize(outputSize)
{
    int currentWidth = inputWidth;
    int currentHeight = inputHeight;

    for (size_t i = 0; i < filterSpecs.size(); ++i)
    {
        int numFilters = filterSpecs[i].first;
        int kernelSize = filterSpecs[i].second;

        ConvLayer convLayer;
        convLayer.numFilters = numFilters;
        convLayer.kernelSize = kernelSize;
        convLayer.stride = convStride;
        convLayer.outputWidth = (currentWidth - kernelSize) / convStride + 1;
        convLayer.outputHeight = (currentHeight - kernelSize) / convStride + 1;
        convLayer.filters.resize(numFilters, std::vector<std::vector<double>>(kernelSize, std::vector<double>(kernelSize)));
        convLayer.biases.resize(numFilters, 0.0);

        // He initialization for filters.
        double fanIn = kernelSize * kernelSize;
        double fanOut = numFilters * convLayer.outputWidth * convLayer.outputHeight;
        double limit = std::sqrt(2.0 / fanIn);
        for (auto &filter : convLayer.filters)
        {
            NNUtils::initializeWeights(filter, -limit, limit);
        }
        convLayers.push_back(std::move(convLayer));

        // Update current dimensions.
        currentWidth = convLayers.back().outputWidth;
        currentHeight = convLayers.back().outputHeight;

        if (i < poolSizes.size())
        {
            int poolSize = poolSizes[i];
            PoolingLayer poolLayer;
            poolLayer.poolSize = poolSize;
            poolLayer.stride = poolStride;
            poolLayer.outputWidth = (currentWidth - poolSize) / poolStride + 1;
            poolLayer.outputHeight = (currentHeight - poolSize) / poolStride + 1;
            poolLayers.push_back(poolLayer);
            currentWidth = poolLayer.outputWidth;
            currentHeight = poolLayer.outputHeight;
        }
    }

    fcInputSize = convLayers.back().numFilters * currentWidth * currentHeight;
    fcWeights.resize(outputSize, std::vector<double>(fcInputSize));

    double fcLimit = std::sqrt(6.0 / (fcInputSize + outputSize));
    NNUtils::initializeWeights(fcWeights, -fcLimit, fcLimit);
    fcBiases.resize(outputSize, 0.0);
}

/**
 * @brief 2D convolution on image
 *
 * @param image 2D Vector of Doubles: Input image
 * @param filter 2D Vector of Doubles: Convolution filter
 * @param bias Double: Bias to add
 * @param stride Int: Convolution stride
 * @return 2D Vector of Doubles: convolved feature map after applying ReLU activation
 */
std::vector<std::vector<double>> CNN::convolve2D(const std::vector<std::vector<double>> &image,
                                                 const std::vector<std::vector<double>> &filter,
                                                 double bias, int stride)
{
    int filterSize = filter.size();
    int outputHeight = (image.size() - filterSize) / stride + 1;
    int outputWidth = (image[0].size() - filter[0].size()) / stride + 1;
    std::vector<std::vector<double>> output(outputHeight, std::vector<double>(outputWidth, 0.0));

    for (int i = 0; i < outputHeight; ++i)
    {
        for (int j = 0; j < outputWidth; ++j)
        {
            double sum = 0.0;
            for (int m = 0; m < filterSize; ++m)
            {
                for (int n = 0; n < filterSize; ++n)
                {
                    sum += image[i * stride + m][j * stride + n] * filter[m][n];
                }
            }
            output[i][j] = NNUtils::ActivationFunctions::relu(sum + bias);
        }
    }
    return output;
}

/**
 * @brief 2D max pooling on a feature map
 *
 * @param featureMap 2D Vector of Doubles: Input feature map
 * @param poolSize Int: size of pooling window
 * @param stride Int: stride for pooling
 * @return A 2D vector containing the pooled feature map
 */
std::vector<std::vector<double>> CNN::maxPool2D(const std::vector<std::vector<double>> &featureMap,
                                                int poolSize, int stride)
{
    int outputHeight = (featureMap.size() - poolSize) / stride + 1;
    int outputWidth = (featureMap[0].size() - poolSize) / stride + 1;
    std::vector<std::vector<double>> output(outputHeight, std::vector<double>(outputWidth, 0.0));

    for (int i = 0; i < outputHeight; ++i)
    {
        for (int j = 0; j < outputWidth; ++j)
        {
            double maxVal = -std::numeric_limits<double>::infinity();
            for (int m = 0; m < poolSize; ++m)
            {
                for (int n = 0; n < poolSize; ++n)
                {
                    double val = featureMap[i * stride + m][j * stride + n];
                    if (val > maxVal)
                        maxVal = val;
                }
            }
            output[i][j] = maxVal;
        }
    }
    return output;
}

/**
 * @brief Flattens a set of 2D feature maps into a 1D vector.
 *
 * @param convOutputs 3D vector: convolutional layer outputs (channel, row, col)
 * @return Vector of Doubles: 1D vector containing all values
 */
std::vector<double> CNN::flatten(const std::vector<std::vector<std::vector<double>>> &convOutputs)
{
    std::vector<double> flattened;
    for (const auto &featureMap : convOutputs)
    {
        for (const auto &row : featureMap)
        {
            flattened.insert(flattened.end(), row.begin(), row.end());
        }
    }
    return flattened;
}

/**
 * @brief Computes the activation of the fully connected layer.
 *
 * @param input Vector of Doubles: flattened input vector
 * @return Vector of Doubles: A vector of raw output values (logits) for each neuron
 */
std::vector<double> CNN::computeFCActivation(const std::vector<double> &input)
{
    std::vector<double> output(outputSize, 0.0);
    for (int i = 0; i < outputSize; ++i)
    {
        for (int j = 0; j < fcInputSize; ++j)
        {
            output[i] += fcWeights[i][j] * input[j];
        }
        output[i] += fcBiases[i];
    }
    return output;
}

/**
 * @brief Performs a forward pass through the CNN
 *
 * @param input Vector:  input image (flattened)
 * @return Vector: output probabilities after the softmax activation
 */
std::vector<double> CNN::performForwardPass(const std::vector<uint8_t> &input)
{
    if (input.size() != static_cast<size_t>(inputWidth * inputHeight))
    {
        std::cerr << "Input size mismatch." << std::endl;
        return {};
    }

    // Normalize input and reshape into a 2D image.
    std::vector<std::vector<double>> image(inputHeight, std::vector<double>(inputWidth));
    for (int i = 0; i < inputHeight; ++i)
    {
        for (int j = 0; j < inputWidth; ++j)
        {
            image[i][j] = static_cast<double>(input[i * inputWidth + j]) / 255.0;
        }
    }

    // Forward pass through convolutional and pooling layers.
    std::vector<std::vector<std::vector<double>>> currentOutput = {image};
    for (size_t layerIdx = 0; layerIdx < convLayers.size(); ++layerIdx)
    {
        const auto &convLayer = convLayers[layerIdx];
        std::vector<std::vector<std::vector<double>>> convOutput;

        // Convolve with each filter in current convolutional layer.
        for (int f = 0; f < convLayer.numFilters; ++f)
        {
            auto featureMap = convolve2D(currentOutput[0], convLayer.filters[f],
                                         convLayer.biases[f], convLayer.stride);
            convOutput.push_back(featureMap);
        }
        currentOutput = convOutput;

        // Apply max pooling if a pooling layer is defined.
        if (layerIdx < poolLayers.size())
        {
            const auto &poolLayer = poolLayers[layerIdx];
            for (auto &featureMap : currentOutput)
            {
                featureMap = maxPool2D(featureMap, poolLayer.poolSize, poolLayer.stride);
            }
        }
    }

    std::vector<double> flattened = flatten(currentOutput);
    std::vector<double> fcOutput = computeFCActivation(flattened);

    return NNUtils::ActivationFunctions::softmax(fcOutput);
}

/**
 * @brief Trains the CNN
 *
 * @param images Vector: input images (each as a flattened vector).
 * @param labels Vector: labels corresponding to the images.
 * @param epochs Int: # training epochs
 * @param learningRate Int: Learning rate for weight updates
 * @param momentum Int: Momentum factor for gradient updates (default is 0.9)
 */
void CNN::train(const std::vector<std::vector<uint8_t>> &images,
                const std::vector<uint8_t> &labels,
                int epochs,
                double learningRate,
                double momentum)
{
    size_t numSamples = images.size();

    // Initialize velocity vectors for momentum-based gradient descent
    std::vector<std::vector<std::vector<std::vector<double>>>> filterVelocities(convLayers.size());
    std::vector<std::vector<double>> convBiasVelocities(convLayers.size()); // Bias velocities
    std::vector<std::vector<double>> fcWeightVelocities(outputSize, std::vector<double>(fcInputSize, 0.0));
    std::vector<double> fcBiasVelocities(outputSize, 0.0);

    // Initialize velocities for each convolutional filter
    for (size_t l = 0; l < convLayers.size(); ++l)
    {
        auto &layer = convLayers[l];
        filterVelocities[l].resize(layer.numFilters, std::vector<std::vector<double>>(layer.kernelSize, std::vector<double>(layer.kernelSize, 0.0)));
        convBiasVelocities[l].resize(layer.numFilters, 0.0);
    }

    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double totalLoss = 0.0;
        for (size_t sample = 0; sample < numSamples; ++sample)
        {
            // Normalize and reshape the current input image.
            std::vector<std::vector<double>> image(inputHeight, std::vector<double>(inputWidth));
            for (int i = 0; i < inputHeight; ++i)
            {
                for (int j = 0; j < inputWidth; ++j)
                {
                    image[i][j] = static_cast<double>(images[sample][i * inputWidth + j]) / 255.0;
                }
            }

            std::vector<std::vector<std::vector<double>>> currentOutput = {image};
            for (size_t layerIdx = 0; layerIdx < convLayers.size(); ++layerIdx)
            {
                const auto &convLayer = convLayers[layerIdx];
                std::vector<std::vector<std::vector<double>>> convOutput;

                for (int f = 0; f < convLayer.numFilters; ++f)
                {
                    auto featureMap = convolve2D(currentOutput[0], convLayer.filters[f],
                                                 convLayer.biases[f], convLayer.stride);
                    convOutput.push_back(featureMap);
                }
                currentOutput = convOutput;

                if (layerIdx < poolLayers.size())
                {
                    const auto &poolLayer = poolLayers[layerIdx];
                    for (auto &featureMap : currentOutput)
                    {
                        featureMap = maxPool2D(featureMap, poolLayer.poolSize, poolLayer.stride);
                    }
                }
            }

            std::vector<double> fcInput = flatten(currentOutput);
            std::vector<double> fcOutput = computeFCActivation(fcInput);
            std::vector<double> probabilities = NNUtils::ActivationFunctions::softmax(fcOutput);

            int actualLabel = labels[sample];
            totalLoss += -std::log(probabilities[actualLabel] + 1e-8);

            std::vector<double> outputError(outputSize);
            for (int i = 0; i < outputSize; ++i)
            {
                double target = (i == actualLabel) ? 1.0 : 0.0;
                outputError[i] = probabilities[i] - target;
            }

            // Update fully connected layer weights and biases
            for (int i = 0; i < outputSize; ++i)
            {
                for (int j = 0; j < fcInputSize; ++j)
                {
                    double grad = outputError[i] * fcInput[j];
                    fcWeightVelocities[i][j] = momentum * fcWeightVelocities[i][j] - learningRate * grad;
                    fcWeights[i][j] += fcWeightVelocities[i][j];
                }
                fcBiasVelocities[i] = momentum * fcBiasVelocities[i] - learningRate * outputError[i];
                fcBiases[i] += fcBiasVelocities[i];
            }

            // Update convolutional layer biases
            for (size_t layerIdx = 0; layerIdx < convLayers.size(); ++layerIdx)
            {
                auto &convLayer = convLayers[layerIdx];

                for (int f = 0; f < convLayer.numFilters; ++f)
                {
                    double biasGrad = 0.0; // Placeholder for actual gradient computation
                    convBiasVelocities[layerIdx][f] = momentum * convBiasVelocities[layerIdx][f] - learningRate * biasGrad;
                    convLayer.biases[f] += convBiasVelocities[layerIdx][f];
                }
            }

            // Backpropagation for convolutional layers is a placeholder.
            // Full implementation would require computing gradients for the filters and biases -> too complex lol
        }
        std::cout << "Epoch " << epoch + 1 << " - Loss: " << totalLoss / numSamples << std::endl;
    }
}

void CNN::saveFinalWeights(const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (const auto &layer : convLayers)
    {
        for (const auto &filter : layer.filters)
        {
            for (const auto &row : filter)
            {
                file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
            }
        }
        file.write(reinterpret_cast<const char *>(layer.biases.data()), layer.biases.size() * sizeof(double));
    }
    for (const auto &row : fcWeights)
    {
        file.write(reinterpret_cast<const char *>(row.data()), row.size() * sizeof(double));
    }
    file.write(reinterpret_cast<const char *>(fcBiases.data()), fcBiases.size() * sizeof(double));

    file.close();
}

void CNN::loadPretrainedWeights(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    for (auto &layer : convLayers)
    {
        for (auto &filter : layer.filters)
        {
            for (auto &row : filter)
            {
                file.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(double));
            }
        }
        file.read(reinterpret_cast<char *>(layer.biases.data()), layer.biases.size() * sizeof(double));
    }
    for (auto &row : fcWeights)
    {
        file.read(reinterpret_cast<char *>(row.data()), row.size() * sizeof(double));
    }
    file.read(reinterpret_cast<char *>(fcBiases.data()), fcBiases.size() * sizeof(double));

    file.close();
}
