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
    ifstream file(fileName, ios::binary);
    if (!file)
    {
        cerr << "loadTrainingData: Error opening file for reading!";
        return {};
    }

    vector<vector<double>> trainingHistory;

    while (!file.eof())
    {
        int epoch, numWeights;
        double loss, accuracy;

        // continously read each field to keep file position correct 
        if (!file.read(reinterpret_cast<char *>(&epoch), sizeof(epoch)))
            break;
        if (!file.read(reinterpret_cast<char *>(&loss), sizeof(loss)))
            break;
        if (!file.read(reinterpret_cast<char *>(&accuracy), sizeof(accuracy)))
            break;
        if (!file.read(reinterpret_cast<char *>(&numWeights), sizeof(numWeights)))
            break;

        vector<double> weights(numWeights);
        if (!file.read(reinterpret_cast<char *>(weights.data()), numWeights * sizeof(double)))
            break;

        cout << "Epoch: " << epoch << " | Loss: " << loss << " | Accuracy: " << accuracy << endl;
        trainingHistory.push_back(weights);
    }
    file.close();
    return trainingHistory;
}

vector<double> TrainingDatabase::getTrainingEpoch(int searchEpoch)
{
    ifstream file(fileName, ios::binary);
    if (!file)
    {
        cerr << "getTrainingEpoch: Error opening file for reading!" << endl;
        return {};
    }
    while (!file.eof())
    {
        int epoch, numWeights;
        double loss, accuracy;
        if (!file.read(reinterpret_cast<char *>(&epoch), sizeof(epoch)))
            break;
        if (!file.read(reinterpret_cast<char *>(&loss), sizeof(loss)))
            break;
        if (!file.read(reinterpret_cast<char *>(&accuracy), sizeof(accuracy)))
            break;
        if (!file.read(reinterpret_cast<char *>(&numWeights), sizeof(numWeights)))
            break;

        vector<double> weights(numWeights);
        if (!file.read(reinterpret_cast<char *>(weights.data()), numWeights * sizeof(double)))
            break;

        if (epoch == searchEpoch)
        {
            cout << "Found Epoch: " << epoch << " | Loss: " << loss << " | Accuracy: " << accuracy << endl;
            return weights;
        }
    }

    cout << "Epoch " << searchEpoch << " not found!" << endl;
    return {};
}