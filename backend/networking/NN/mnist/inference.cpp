#include "mnist_loader.hpp"
#include "../ff_neural_net.hpp"
#include <iostream>
#include <fstream>

using namespace std;

const int MNIST_IMAGE_ROWS = 28;
const int MNIST_IMAGE_COLS = 28;
const int MNIST_IMAGE_SIZE = MNIST_IMAGE_ROWS * MNIST_IMAGE_COLS;
const int MNIST_POSSIBLE_DIGIT_OUTPUTS = 10;

int main()
{
    std::vector<std::vector<uint8_t>> test_images = loadMNISTImages("../../../data/mnist/t10k-images-idx3-ubyte/t10k-images-idx3-ubyte", 10);
    std::cout << "Number of images loaded: " << test_images.size() << std::endl;

    FFNeuralNet net(MNIST_IMAGE_SIZE, 128, MNIST_POSSIBLE_DIGIT_OUTPUTS);
    net.loadPretrainedWeights("mnist/data/weights.dat");

    std::ofstream prob_file("mnist/data/probabilities.dat", std::ios::binary);
    if (!prob_file.is_open())
    {
        std::cerr << "Error opening probabilities.dat for writing!" << std::endl;
        return 1;
    }

    for (size_t i = 0; i < test_images.size(); ++i)
    {
        if (test_images[i].size() != MNIST_IMAGE_SIZE)
        {
            std::cerr << "Invalid image size at index " << i << ": " << test_images[i].size() << std::endl;
            continue;
        }

        std::vector<double> probabilityOutputs = net.performForwardPass(test_images[i]);
        for (int i = 0; i < probabilityOutputs.size(); ++i)
        {
            cout << probabilityOutputs.data()[i];
        }
        prob_file.write(reinterpret_cast<const char *>(probabilityOutputs.data()), probabilityOutputs.size() * sizeof(double));
    }

    prob_file.close();
    return 0;
}
