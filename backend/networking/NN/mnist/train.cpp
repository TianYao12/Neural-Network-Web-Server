#include "mnist_loader.hpp"
#include "../ff_neural_net.hpp"
#include "../convolutional_neural_net.hpp"

const std::string MNIST_TRAIN_IMAGES_PATH = "../../../data/mnist/train-images.idx3-ubyte";
const std::string MNIST_TRAIN_LABELS_PATH = "../../../data/mnist/train-labels.idx1-ubyte";
const int NUM_TRAINING_IMAGES = 1000;
const int INPUT_LAYER_ONE_DIMENSION_SIZE = 28;
const int INPUT_LAYER_SIZE = 28 * 28;
const int HIDDEN_LAYER_SIZE = 128;
const int OUTPUT_LAYER_SIZE = 10; // (0-9)
const int NUM_EPOCHS = 10;
const double LEARNING_RATE = 0.1; // 0.001
const std::string FINAL_WEIGHTS_FILE = "mnist/data/weights.dat";

// CNN config
const std::vector<std::pair<int, int>> CONV_SPECS = {{8, 3}, {16, 3}}; // Example: 2 conv layers (8 filters of 3x3, then 16 filters of 3x3)
const std::vector<int> POOL_SIZES = {2, 2};                            // Max pooling after each convolution
const double MOMENTUM = 0.9;

int main()
{
    std::vector<std::vector<uint8_t>> images = loadMNISTImages(MNIST_TRAIN_IMAGES_PATH, NUM_TRAINING_IMAGES);
    std::vector<uint8_t> labels = loadMNISTLabels(MNIST_TRAIN_LABELS_PATH, NUM_TRAINING_IMAGES);

    // FFNeuralNet net(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
    // net.train(images, labels, NUM_EPOCHS, LEARNING_RATE);
    // net.saveFinalWeights(FINAL_WEIGHTS_FILE);

    CNN cnnet(INPUT_LAYER_ONE_DIMENSION_SIZE, INPUT_LAYER_ONE_DIMENSION_SIZE, CONV_SPECS, POOL_SIZES, OUTPUT_LAYER_SIZE);
    cnnet.train(images, labels, NUM_EPOCHS, LEARNING_RATE);
    cnnet.saveFinalWeights(FINAL_WEIGHTS_FILE);
    return 0;
}