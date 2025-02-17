#ifndef DATABASE_HPP
#define DATABASE_HPP

#include <iostream>
#include <fstream>
#include <vector>

class TrainingDatabase {
    std::string fileName;

    public:
    TrainingDatabase(const std::string& file) : fileName{file} {}

    void saveTrainingData(int epoch, double loss, double accuracy, const std::vector<double>& weights);
    std::vector<std::vector<double>> loadTrainingData();
};

#endif