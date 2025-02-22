#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>

class TrainingDatabase {
    std::string fileName;
    std::string probabilityFileName;
    static constexpr int MNIST_POSSIBLE_DIGIT_OUTPUTS = 10;

    struct TrainingRecord {
        int epoch;
        double loss;
        std::vector<double> weights;
    };

public:
    TrainingDatabase(const std::string& file, const std::string &probFile); 
    bool saveTrainingData(int epoch, double loss, const std::vector<double>& weights);
    std::vector<TrainingRecord> loadTrainingResults();
    std::vector<std::vector<double>> loadProbabilitiesFromInference();
    std::pair<std::vector<TrainingRecord>, std::vector<std::vector<double>>> loadAllTrainingData();
};

#endif