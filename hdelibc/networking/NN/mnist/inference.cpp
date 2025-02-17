#include "mnist_loader.hpp"
#include "../ff_neural_net.hpp"
#include <iostream>

const int MNIST_IMAGE_ROWS = 28;
const int MNIST_IMAGE_COLS = 28;
const int MNIST_IMAGE_SIZE = MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS;
const int MNIST_POSSIBLE_DIGIT_OUTPUTS = 10;

int main()
{
    std::vector<std::vector<uint8_t>> test_images = loadMNISTImages("../../../data/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", 10);
    std::cout << "Number of images loaded: " << test_images.size() << std::endl;

    FFNeuralNet net(MNIST_IMAGE_SIZE, 128, MNIST_POSSIBLE_DIGIT_OUTPUTS);
    net.loadWeights("weights.dat");

    for (size_t i = 0; i < test_images.size(); ++i)
    {
        if (test_images[i].size() != MNIST_IMAGE_SIZE)
        {
            std::cerr << "Invalid image size at index " << i << ": " << test_images[i].size() << std::endl;
            continue;
        }

        std::vector<double> output = net.forward(test_images[i]);

        std::cout << "Output probabilities: ";
        std::cout.precision(10);
        for (double prob : output)
        {
            std::cout << prob << " ";
        }
        std::cout << "\n\n";
    }
    return 0;
}