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
 * @param activation_function   Activation function to be applied to the layer's output
 *
 * @return Output vector for the current layer
 */
vector<double> FFNeuralNet::layerForward(
    const vector<double> &input,
    const vector<vector<double>> &weights,
    vector<double> &biases,
    function<double(double)> activation_function)
{
    vector<double> output(weights.size(), 0.0);
    for (size_t i = 0; i < weights.size(); ++i)
    {
        for (size_t j = 0; j < input.size(); ++j)
        {
            output[i] += weights[i][j] * input[j];
        }
        output[i] += biases[i];
        output[i] = activation_function(output[i]);
    }
    return output;
}

/**
 * @brief Backpropagation algorithm to update weights and biases.
 *
 * @param input_normalized      Normalized input vector used in the forward pass.
 * @param hidden_layer_output   Output vector of the hidden layer from the forward pass.
 * @param output_layer_output   Output probability vector from the output layer (after softmax) from the forward pass.
 * @param true_label            Correct class label for the input
 * @param learning_rate         Control magnitude of weight and bias updates
 */
void FFNeuralNet::backpropagate(
    const vector<double> &input_normalized,
    const vector<double> &hidden_layer_output,
    const vector<double> &output_layer_output,
    int true_label,
    double learning_rate)
{
    vector<double> output_error(output_layer_output.size());
    for (size_t j = 0; j < output_layer_output.size(); ++j)
    {
        output_error[j] = output_layer_output[j] - (j == true_label ? 1.0 : 0.0);
    }

    // Gradients for hidden-to-output weights and output biases
    for (size_t j = 0; j < hidden_to_output_weights.size(); ++j)
    {
        for (size_t k = 0; k < hidden_layer_output.size(); ++k)
        {
            hidden_to_output_weights[j][k] -= learning_rate * output_error[j] * hidden_layer_output[k];
        }
        output_biases[j] -= learning_rate * output_error[j];
    }

    vector<double> hidden_error(hidden_layer_output.size(), 0.0);
    for (size_t j = 0; j < hidden_layer_output.size(); ++j)
    {
        for (size_t k = 0; k < output_layer_output.size(); ++k)
        {
            hidden_error[j] += output_error[k] * hidden_to_output_weights[k][j];
        }
    }

    for (size_t j = 0; j < hidden_error.size(); ++j)
    {
        hidden_error[j] *= NNUtils::ActivationFunctions::reluDerivative(hidden_layer_output[j]);
    }

    // Gradients for input-to-hidden weights and hidden biases
    for (size_t j = 0; j < input_to_hidden_weights.size(); ++j)
    {
        for (size_t k = 0; k < input_normalized.size(); ++k)
        {
            input_to_hidden_weights[j][k] -= learning_rate * hidden_error[j] * input_normalized[k];
        }
        hidden_biases[j] -= learning_rate * hidden_error[j];
    }
}

/**
 * @brief Constructor that initializes the neural network architecture and parameters
 *
 * @param input_size  Number of neurons in the input layer
 * @param hidden_size Number of neurons in the hidden layer
 * @param output_size Number of neurons in the output layer / number of output classes
 */
FFNeuralNet::FFNeuralNet(int input_size, int hidden_size, int output_size) : input_to_hidden_weights(hidden_size, vector<double>(input_size)),
                                                                             hidden_to_output_weights(output_size, vector<double>(hidden_size)),
                                                                             hidden_biases(hidden_size),
                                                                             output_biases(output_size)
{
    NNUtils::initializeWeights(input_to_hidden_weights, -0.5, 0.5);
    NNUtils::initializeWeights(hidden_to_output_weights, -0.5, 0.5);
    NNUtils::initializeBiases(hidden_biases);
    NNUtils::initializeBiases(output_biases);
}

/**
 * @brief Complete forward pass for inference
 *
 * @param input_bytes Input data as a vector of unsigned 8-bit integers (bytes) representing pixel values of image
 *
 * @return Output vector containing probabilities for each class after softmax activation (vector size = # output neurons/classes)
 */
vector<double> FFNeuralNet::forward(const vector<uint8_t> &input_bytes)
{
    vector<double> input_normalized(input_bytes.size());
    for (size_t i = 0; i < input_bytes.size(); ++i)
    {
        input_normalized[i] = static_cast<double>(input_bytes[i]) / 255.0;
    }

    vector<double> hidden_layer_output = layerForward(
        input_normalized,
        input_to_hidden_weights,
        hidden_biases, NNUtils::ActivationFunctions::relu);

    vector<double> output_layer_logits = layerForward(
        hidden_layer_output,
        hidden_to_output_weights,
        output_biases,
        [](double x)
        { return x; });

    return NNUtils::ActivationFunctions::softmax(output_layer_logits);
}

/**
 * @brief Helper function to flatten network parameters (weights and biases) into a single vector.
 *
 * @return A vector of doubles containing all weights and biases of the network.
 */
vector<double> FFNeuralNet::getParamsAsVector() const
{
    vector<double> params;
    for (const auto &row : input_to_hidden_weights)
    {
        params.insert(params.end(), row.begin(), row.end());
    }
    for (const auto &row : hidden_to_output_weights)
    {
        params.insert(params.end(), row.begin(), row.end());
    }
    params.insert(params.end(), hidden_biases.begin(), hidden_biases.end());
    params.insert(params.end(), output_biases.begin(), output_biases.end());
    return params;
}

/**
 * @brief Using training images and labels to train NN using gradient descent/backpropagation
 *
 * @param images        2D vector of unsigned 8-bit integers representing training images, where each inner vector is a flattened image
 * @param labels        Vector of unsigned 8-bit integers representing labels for the training images.
 * @param epochs        Number of training epochs
 * @param learning_rate Controls step size of weight and bias updates in gradient descent
 */
void FFNeuralNet::train(const vector<vector<uint8_t>> &images,
                        const vector<uint8_t> &labels,
                        int epochs, double learning_rate)
{
    TrainingDatabase db("training_data.db", "probabilities.dat");

    size_t num_samples = images.size();
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        double total_loss = 0.0;

        for (size_t i = 0; i < num_samples; ++i)
        {
            vector<double> input_normalized(images[i].size());
            for (size_t j = 0; j < images[i].size(); ++j)
            {
                input_normalized[j] = static_cast<double>(images[i][j]) / 255.0;
            }

            vector<double> hidden_layer_output = layerForward(
                input_normalized,
                input_to_hidden_weights,
                hidden_biases, NNUtils::ActivationFunctions::relu);

            vector<double> output_layer_output = NNUtils::ActivationFunctions::softmax(layerForward(
                hidden_layer_output,
                hidden_to_output_weights,
                output_biases,
                [](double x)
                { return x; }));

            int true_label = labels[i];
            double loss = -log(output_layer_output[true_label]);
            total_loss += loss;

            backpropagate(input_normalized, hidden_layer_output, output_layer_output, true_label, learning_rate);
        }

        double average_loss = total_loss / num_samples;
        cout << "Epoch " << epoch + 1 << " - Loss: " << average_loss << endl;

        vector<double> current_params = getParamsAsVector();
        db.saveTrainingData(epoch + 1, average_loss, current_params);
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
        for (const auto &row : input_to_hidden_weights)
        {
            for (double w : row)
            {
                file.write(reinterpret_cast<const char *>(&w), sizeof(double));
            }
        }
        for (const auto &row : hidden_to_output_weights)
        {
            for (double w : row)
            {
                file.write(reinterpret_cast<const char *>(&w), sizeof(double));
            }
        }
        for (double b : hidden_biases)
        {
            file.write(reinterpret_cast<const char *>(&b), sizeof(double));
        }
        for (double b : output_biases)
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

void FFNeuralNet::loadWeights(const string &filename)
{
    ifstream file(filename, ios::binary);
    if (!file.is_open())
    {
        cerr << "Error opening weight file for loading: " << filename << endl;
        return;
    }
    try
    {
        for (auto &row : input_to_hidden_weights)
        {
            for (double &w : row)
            {
                file.read(reinterpret_cast<char *>(&w), sizeof(double));
            }
        }
        for (auto &row : hidden_to_output_weights)
        {
            for (double &w : row)
            {
                file.read(reinterpret_cast<char *>(&w), sizeof(double));
            }
        }
        for (double &b : hidden_biases)
        {
            file.read(reinterpret_cast<char *>(&b), sizeof(double));
        }
        for (double &b : output_biases)
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