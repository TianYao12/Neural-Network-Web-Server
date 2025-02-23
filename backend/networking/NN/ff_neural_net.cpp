#include "ff_neural_net.hpp"
#include "../Database/Database.hpp"
#include <vector>
#include <cmath>
#include <functional>
#include <fstream>
#include <iostream>

using namespace std;

/**
 * @brief Forward pass for a single layer
 *
 * @param input                 Input vector from the previous layer (or input layer)
 * @param weights               Weight matrix with dimensions: (current layer size x previous layer size)
 * @param biases                Bias vector for the current layer
 * @param activationFunction    Activation function to be applied to the weighted sum + bias for each row of the weight matrix
 *
 * @return Output vector for the current layer
 */
vector<double> FFNeuralNet::computeLayerActivation(
    const vector<double> &input,
    const vector<vector<double>> &weights,
    vector<double> &biases,
    function<double(double)> activationFunction)
{
    vector<double> output(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i)
    {
        for (size_t j = 0; j < input.size(); ++j)
        {
            output[i] += weights[i][j] * input[j];
        }
        output[i] += biases[i];
        output[i] = activationFunction(output[i]);
    }
    return output;
}

/**
 * @brief Backpropagation algorithm to update weights and biases
 *
 * @param inputNormalized      Normalized input vector used in the forward pass
 * @param hiddenLayerOutput    Output vector of the hidden layer from the forward pass
 * @param outputLayerOutput    Output probability vector from the output layer (after softmax) from the forward pass
 * @param actualLabel            Correct class label for the input
 * @param learningRate         Control magnitude of weight and bias updates
 */
void FFNeuralNet::applyBackpropagation(
    const vector<double> &inputNormalized,
    const vector<double> &hiddenLayerOutput,
    const vector<double> &outputLayerOutput,
    int actualLabel,
    double learningRate)
{
    vector<double> output_error(outputLayerOutput.size());

    for (size_t j = 0; j < outputLayerOutput.size(); ++j)
    {
        // derivative of the loss function (cross-entropy loss with softmax output)
        // output error for class j = predicted probability for class j - true label (correct 1, incorrect 0)
        output_error[j] = outputLayerOutput[j] - (j == actualLabel ? 1.0 : 0.0);
    }

    // Gradients for hidden-to-output weights and output biases
    for (size_t j = 0; j < hiddenToOutputLayerWeights.size(); ++j)
    {
        for (size_t k = 0; k < hiddenLayerOutput.size(); ++k)
        {
            hiddenToOutputLayerWeights[j][k] -= learningRate * output_error[j] * hiddenLayerOutput[k];
        }
        outputLayerBiases[j] -= learningRate * output_error[j];
    }

    vector<double> hidden_error(hiddenLayerOutput.size(), 0.0);
    for (size_t j = 0; j < hiddenLayerOutput.size(); ++j)
    {
        hidden_error[j] = 0.0; 

        for (size_t k = 0; k < outputLayerOutput.size(); ++k)
        {
            hidden_error[j] += output_error[k] * hiddenToOutputLayerWeights[k][j];
        }
        hidden_error[j] *= NNUtils::ActivationFunctions::reluDerivative(hiddenLayerOutput[j]);
    }

    // Gradients for input-to-hidden weights and hidden biases
    for (size_t j = 0; j < inputToHiddenLayerWeights.size(); ++j)
    {
        for (size_t k = 0; k < inputNormalized.size(); ++k)
        {
            inputToHiddenLayerWeights[j][k] -= learningRate * hidden_error[j] * inputNormalized[k];
        }
        hiddenLayerBiases[j] -= learningRate * hidden_error[j];
    }
}

/**
 * @brief Constructor that initializes the neural network architecture and parameters
 *
 * @param inputSize  Number of neurons in the input layer
 * @param hiddenSize Number of neurons in the hidden layer
 * @param outputSize Number of neurons in the output layer / number of output classes
 */
FFNeuralNet::FFNeuralNet(int inputSize, int hiddenSize, int outputSize) : inputToHiddenLayerWeights(hiddenSize, vector<double>(inputSize)),
                                                                          hiddenToOutputLayerWeights(outputSize, vector<double>(hiddenSize)),
                                                                          hiddenLayerBiases(hiddenSize),
                                                                          outputLayerBiases(outputSize)
{
    NNUtils::initializeWeights(inputToHiddenLayerWeights, -0.5, 0.5);
    NNUtils::initializeWeights(hiddenToOutputLayerWeights, -0.5, 0.5);
    NNUtils::initializeBiases(hiddenLayerBiases);
    NNUtils::initializeBiases(outputLayerBiases);
}

/**
 * @brief Complete forward pass for inference
 *
 * @param input_bytes Input data as a vector of unsigned 8-bit integers (bytes) representing pixel values of image
 *
 * @return Output vector containing probabilities for each class after softmax activation (vector size = # output neurons/classes)
 */
vector<double> FFNeuralNet::performForwardPass(const vector<uint8_t> &input_bytes)
{
    vector<double> inputNormalized(input_bytes.size());
    for (size_t i = 0; i < input_bytes.size(); ++i)
    {
        inputNormalized[i] = static_cast<double>(input_bytes[i]) / 255.0;
    }

    vector<double> hiddenLayerOutput = computeLayerActivation(
        inputNormalized,
        inputToHiddenLayerWeights,
        hiddenLayerBiases, NNUtils::ActivationFunctions::relu);

    vector<double> output_layer_logits = computeLayerActivation(
        hiddenLayerOutput,
        hiddenToOutputLayerWeights,
        outputLayerBiases,
        [](double x)
        { return x; });

    return NNUtils::ActivationFunctions::softmax(output_layer_logits);
}

/**
 * @brief Helper function to flatten network parameters (weights and biases) into a single vector.
 *
 * @return A vector of doubles containing all weights and biases of the network.
 */
vector<double> FFNeuralNet::extractNetworkParameters() const
{
    vector<double> params;
    for (const auto &row : inputToHiddenLayerWeights)
    {
        params.insert(params.end(), row.begin(), row.end());
    }
    for (const auto &row : hiddenToOutputLayerWeights)
    {
        params.insert(params.end(), row.begin(), row.end());
    }
    params.insert(params.end(), hiddenLayerBiases.begin(), hiddenLayerBiases.end());
    params.insert(params.end(), outputLayerBiases.begin(), outputLayerBiases.end());
    return params;
}

/**
 * @brief Using training images and labels to train NN using gradient descent/backpropagation
 *
 * @param images        2D vector of unsigned 8-bit integers representing training images, where each inner vector is a flattened image
 * @param labels        Vector of unsigned 8-bit integers representing labels for the training images.
 * @param epochs        Number of training epochs
 * @param learningRate Controls step size of weight and bias updates in gradient descent
 */
void FFNeuralNet::train(const vector<vector<uint8_t>> &images,
                        const vector<uint8_t> &labels,
                        int epochs, double learningRate)
{
    TrainingDatabase db("mnist/data/training_data.dat", "mnist/data/probabilities.dat");

    size_t numSamples = images.size();
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double totalLoss = 0.0;

        for (size_t i = 0; i < numSamples; ++i)
        {
            vector<double> inputNormalized(images[i].size());
            for (size_t j = 0; j < images[i].size(); ++j)
            {
                inputNormalized[j] = static_cast<double>(images[i][j]) / 255.0;
            }

            vector<double> hiddenLayerOutput = computeLayerActivation(
                inputNormalized,
                inputToHiddenLayerWeights,
                hiddenLayerBiases, NNUtils::ActivationFunctions::relu);

            vector<double> outputLayerOutput = NNUtils::ActivationFunctions::softmax(computeLayerActivation(
                hiddenLayerOutput,
                hiddenToOutputLayerWeights,
                outputLayerBiases,
                [](double x)
                { return x; }));

            int actualLabel = labels[i];
            double loss = -log(outputLayerOutput[actualLabel]);
            totalLoss += loss;

            applyBackpropagation(inputNormalized, hiddenLayerOutput, outputLayerOutput, actualLabel, learningRate);
        }

        double averageLoss = totalLoss / numSamples;
        cout << "Epoch " << epoch + 1 << " - Loss: " << averageLoss << endl;

        vector<double> currentParams = extractNetworkParameters();
        db.saveTrainingData(epoch + 1, averageLoss, currentParams);
    }
}

void FFNeuralNet::saveFinalWeights(const string &filename)
{
    ofstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening file for saving weights: " << filename << endl;
        return;
    }
    try
    {
        for (const auto &row : inputToHiddenLayerWeights)
        {
            for (double w : row)
            {
                file.write(reinterpret_cast<const char *>(&w), sizeof(double));
            }
        }
        for (const auto &row : hiddenToOutputLayerWeights)
        {
            for (double w : row)
            {
                file.write(reinterpret_cast<const char *>(&w), sizeof(double));
            }
        }
        for (double b : hiddenLayerBiases)
        {
            file.write(reinterpret_cast<const char *>(&b), sizeof(double));
        }
        for (double b : outputLayerBiases)
        {
            file.write(reinterpret_cast<const char *>(&b), sizeof(double));
        }
    }
    catch (const exception &e)
    {
        cerr << "Error writing weights to file: " << e.what() << endl;
    }
    file.close();
}

void FFNeuralNet::loadPretrainedWeights(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening weight file for loading: " << filename << endl;
        return;
    }
    try
    {
        for (auto &row : inputToHiddenLayerWeights)
        {
            for (double &w : row)
            {
                file.read(reinterpret_cast<char *>(&w), sizeof(double));
            }
        }
        for (auto &row : hiddenToOutputLayerWeights)
        {
            for (double &w : row)
            {
                file.read(reinterpret_cast<char *>(&w), sizeof(double));
            }
        }
        for (double &b : hiddenLayerBiases)
        {
            file.read(reinterpret_cast<char *>(&b), sizeof(double));
        }
        for (double &b : outputLayerBiases)
        {
            file.read(reinterpret_cast<char *>(&b), sizeof(double));
        }
    }
    catch (const exception &e)
    {
        cerr << "Error reading weights from file: " << e.what() << endl;
    }
    file.close();
}