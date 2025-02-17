#include "mnist_loader.hpp"
#include "../ff_neural_net.hpp"

const std::string MNIST_TRAIN_IMAGES_PATH = "../../../data/mnist/train-images.idx3-ubyte";
const std::string MNIST_TRAIN_LABELS_PATH = "../../../data/mnist/train-labels.idx1-ubyte";
const int NUM_TRAINING_IMAGES = 1000;
const int INPUT_LAYER_SIZE = 28 * 28; 
const int HIDDEN_LAYER_SIZE = 128;
const int OUTPUT_LAYER_SIZE = 10; // (0-9)
const int NUM_EPOCHS = 10;
const double LEARNING_RATE = 0.001;
const std::string WEIGHTS_FILE = "weights.dat";

int main() {
    std::vector<std::vector<uint8_t>> images = loadMNISTImages(MNIST_TRAIN_IMAGES_PATH, NUM_TRAINING_IMAGES);
    std::vector<uint8_t> labels = loadMNISTLabels(MNIST_TRAIN_LABELS_PATH, NUM_TRAINING_IMAGES);

    FFNeuralNet net(INPUT_LAYER_SIZE, HIDDEN_LAYER_SIZE, OUTPUT_LAYER_SIZE);
    net.train(images, labels, NUM_EPOCHS, LEARNING_RATE);
    net.saveWeights(WEIGHTS_FILE);
    return 0;
}