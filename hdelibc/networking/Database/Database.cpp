#include "Database.hpp"
#include <fstream>
#include <iostream>

using namespace std;

bool TrainingDatabase::saveTrainingData(int epoch, double loss, const std::vector<double>& weights) {
    std::ofstream file(fileName, std::ios::binary | std::ios::app);
    if (!file) {
        std::cerr << "saveTrainingData: Error opening file for writing: " << fileName << std::endl;
        return false;
    }

    int numWeights = weights.size();
    
    std::cout << "Writing record - Size in bytes: " 
              << (sizeof(epoch) + sizeof(loss) + sizeof(numWeights) + numWeights * sizeof(double))
              << std::endl;
    
    file.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
    file.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
    file.write(reinterpret_cast<const char*>(&numWeights), sizeof(numWeights));
    file.write(reinterpret_cast<const char*>(weights.data()), numWeights * sizeof(double));
    
    if (!file) {
        std::cerr << "saveTrainingData: Error writing data for epoch " << epoch << std::endl;
        return false;
    }

    std::cout << "Saved training data - Epoch: " << epoch 
              << ", Loss: " << loss 
              << ", Weights: " << numWeights << std::endl;
    
    file.close();
    return true;
}

std::vector<TrainingDatabase::TrainingRecord> TrainingDatabase::loadTrainingResults() {
    std::ifstream file(fileName, std::ios::binary);
    if (!file) {
        std::cerr << "loadTrainingResults: Error opening file: " << fileName << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    std::cout << "Training data file size: " << fileSize << " bytes" << std::endl;

    const std::streamsize EXPECTED_RECORD_SIZE = sizeof(int) + sizeof(double) + sizeof(int) + (101770 * sizeof(double));
    std::cout << "Expected record size: " << EXPECTED_RECORD_SIZE << " bytes" << std::endl;

    if (fileSize % EXPECTED_RECORD_SIZE != 0) {
        std::cerr << "Warning: File size is not a multiple of expected record size" << std::endl;
    }

    std::vector<TrainingRecord> records;
    std::streamsize bytesRead = 0;

    while (file && bytesRead < fileSize) {
        TrainingRecord record;
        
        if (!file.read(reinterpret_cast<char*>(&record.epoch), sizeof(record.epoch)) ||
            !file.read(reinterpret_cast<char*>(&record.loss), sizeof(record.loss))) {
            break;
        }

        int numWeights;
        if (!file.read(reinterpret_cast<char*>(&numWeights), sizeof(numWeights))) {
            break;
        }

        record.weights.resize(numWeights);
        if (!file.read(reinterpret_cast<char*>(record.weights.data()), 
                      numWeights * sizeof(double))) {
            std::cerr << "Error reading weights at epoch " << record.epoch << std::endl;
            break;
        }

        bytesRead = file.tellg();
        std::cout << "Loaded epoch " << record.epoch 
                  << ", loss " << record.loss 
                  << ", weights " << numWeights 
                  << " (bytes read: " << bytesRead << ")" << std::endl;

        if (!record.weights.empty()) {
            std::cout << "First 3 weights: " 
                     << record.weights[0] << " "
                     << record.weights[1] << " "
                     << record.weights[2] << std::endl;
        }

        records.push_back(record);
    }

    std::cout << "Loaded " << records.size() << " training records" << std::endl;
    file.close();
    return records;
}

vector<vector<double>> TrainingDatabase::loadProbabilitiesFromInference()
{
    ifstream probabilityDataFile(probabilityFileName, ios::binary);
    if (!probabilityDataFile)
    {
        cerr << "loadProbabilityData: Error opening probabilities file for reading: " << probabilityFileName << endl;
        return {};
    }

    vector<vector<double>> probabilityDataHistory; 

    while (!probabilityDataFile.eof())
    {
        vector<double> probabilities(MNIST_POSSIBLE_DIGIT_OUTPUTS);
        if (!probabilityDataFile.read(reinterpret_cast<char *>(probabilities.data()), MNIST_POSSIBLE_DIGIT_OUTPUTS * sizeof(double)))
            break;

        probabilityDataHistory.push_back(probabilities);
    }
    probabilityDataFile.close();
    cout << "loadProbabilityData: Successfully loaded probability data from " << probabilityFileName << endl;
    return probabilityDataHistory;
}

std::pair<std::vector<TrainingDatabase::TrainingRecord>, std::vector<std::vector<double>>> 
TrainingDatabase::loadAllTrainingData() {
    return {loadTrainingResults(), loadProbabilitiesFromInference()};
}