#include "Database.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace std;

TrainingDatabase::TrainingDatabase(const string& file, const string &probFile) 
    : fileName(file), probabilityFileName{probFile} 
{
    if (!filesystem::exists(fileName)) {
        ofstream file(fileName, ios::binary | ios::trunc);
        file.close();
    }
    
    if (!filesystem::exists(probabilityFileName)) {
        ofstream probFile(probabilityFileName, ios::binary | ios::trunc);
        probFile.close();
    }
}


bool TrainingDatabase::saveTrainingData(int epoch, double loss, const vector<double>& weights) {
    ofstream file(fileName, ios::binary | ios::app);
    if (!file) {
        cerr << "saveTrainingData: Error opening file for writing: " << fileName << endl;
        return false;
    }

    int numWeights = weights.size();
    
    cout << "Writing record - Size in bytes: " 
              << (sizeof(epoch) + sizeof(loss) + sizeof(numWeights) + numWeights * sizeof(double))
              << endl;
    
    file.write(reinterpret_cast<const char*>(&epoch), sizeof(epoch));
    file.write(reinterpret_cast<const char*>(&loss), sizeof(loss));
    file.write(reinterpret_cast<const char*>(&numWeights), sizeof(numWeights));
    file.write(reinterpret_cast<const char*>(weights.data()), numWeights * sizeof(double));
    
    if (!file) {
        cerr << "saveTrainingData: Error writing data for epoch " << epoch << endl;
        return false;
    }

    cout << "Saved training data - Epoch: " << epoch 
              << ", Loss: " << loss 
              << ", Weights: " << numWeights << endl;
    
    file.close();
    return true;
}

vector<TrainingDatabase::TrainingRecord> TrainingDatabase::loadTrainingResults() {
    ifstream file(fileName, ios::binary);
    if (!file) {
        cerr << "loadTrainingResults: Error opening file: " << fileName << endl;
        return {};
    }

    file.seekg(0, ios::end);
    streamsize fileSize = file.tellg();
    file.seekg(0, ios::beg);

    cout << "Training data file size: " << fileSize << " bytes" << endl;

    const streamsize EXPECTED_RECORD_SIZE = sizeof(int) + sizeof(double) + sizeof(int) + (101770 * sizeof(double));
    cout << "Expected record size: " << EXPECTED_RECORD_SIZE << " bytes" << endl;

    if (fileSize % EXPECTED_RECORD_SIZE != 0) {
        cerr << "Warning: File size is not a multiple of expected record size" << endl;
    }

    vector<TrainingRecord> records;
    streamsize bytesRead = 0;

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
            cerr << "Error reading weights at epoch " << record.epoch << endl;
            break;
        }

        bytesRead = file.tellg();
        cout << "Loaded epoch " << record.epoch 
                  << ", loss " << record.loss 
                  << ", weights " << numWeights 
                  << " (bytes read: " << bytesRead << ")" << endl;

        if (!record.weights.empty()) {
            cout << "First 3 weights: " 
                     << record.weights[0] << " "
                     << record.weights[1] << " "
                     << record.weights[2] << endl;
        }

        records.push_back(record);
    }

    cout << "Loaded " << records.size() << " training records" << endl;
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

pair<vector<TrainingDatabase::TrainingRecord>, vector<vector<double>>> 
TrainingDatabase::loadAllTrainingData() {
    return {loadTrainingResults(), loadProbabilitiesFromInference()};
}