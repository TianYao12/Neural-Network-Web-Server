#include "Database.hpp"

using namespace std;

void TrainingDatabase::saveTrainingData(int epoch, double loss, double accuracy, const vector<double> &weights)
{
    ofstream file(fileName, ios::binary | ios::app); // binary (not text) mode and append mode 

    if (!file)
    {
        cerr << "saveTrainingData: Error opening file for writing!" << endl;
        return;
    }

    int numWeights = weights.size();

    // use reinterpret_cast to treat as raw bytes because write requires a byte array here
    file.write(reinterpret_cast<const char *>(&epoch), sizeof(epoch));
    file.write(reinterpret_cast<const char *>(&loss), sizeof(loss));
    file.write(reinterpret_cast<const char *>(&accuracy), sizeof(accuracy));
    file.write(reinterpret_cast<const char *>(&numWeights), sizeof(numWeights));
    file.write(reinterpret_cast<const char *>(weights.data()), numWeights * sizeof(double));
    file.close();
}

vector<vector<double>> TrainingDatabase::loadTrainingData()
{
    const int MNIST_POSSIBLE_DIGIT_OUTPUTS = 10;
    ifstream file("./NN/probabilities.dat", ios::binary);
    if (!file)
    {
        cerr << "loadTrainingData: Error opening probabilities file for reading!";
        return {};
    }

    vector<vector<double>> probabilityHistory;
    while (!file.eof())
    {
        vector<double> probabilities(MNIST_POSSIBLE_DIGIT_OUTPUTS);
        if (!file.read(reinterpret_cast<char *>(probabilities.data()), MNIST_POSSIBLE_DIGIT_OUTPUTS * sizeof(double)))
            break;

        probabilityHistory.push_back(probabilities);
    }
    file.close();
    return probabilityHistory;
}