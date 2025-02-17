#ifndef FF_NEURAL_NET_HPP
#define FF_NEURAL_NET_HPP

#include <vector>
#include <functional>
#include "../utils/utils.hpp" 

class FFNeuralNet {
    std::vector<std::vector<double>> input_to_hidden_weights;
    std::vector<std::vector<double>> hidden_to_output_weights;
    std::vector<double> hidden_biases;
    std::vector<double> output_biases;

    std::vector<double> layerForward(
        const std::vector<double>& input,
        const std::vector<std::vector<double>>& weights,
        std::vector<double>& biases,
        std::function<double(double)> activation_function
    );

    void backpropagate(
        const std::vector<double>& input_normalized,
        const std::vector<double>& hidden_layer_output,
        const std::vector<double>& output_layer_output,
        int true_label,
        double learning_rate
    );
    std::vector<double> getParamsAsVector() const;


public:
    FFNeuralNet(int input_size, int hidden_size, int output_size);
    std::vector<double> forward(const std::vector<uint8_t> &input);

    void train(
        const std::vector<std::vector<uint8_t>> &images,
        const std::vector<uint8_t> &labels,
        int epochs, double learning_rate);

    void saveFinalWeights(const std::string &fileName);
    void loadWeights(const std::string &filename);
};

#endif 