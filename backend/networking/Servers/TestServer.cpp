#include "TestServer.hpp"
#include "../Database/Database.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>

#include <cstdlib>
#include <ctime>

using namespace std;

HDE::TestServer::TestServer() : SimpleServer{AF_INET, SOCK_STREAM, 0, 80, INADDR_ANY, 10}
{
    startServer();
}

void HDE::TestServer::acceptClientConnection()
{
    struct sockaddr_in address = getSocket()->getAddress();
    int arrLen = sizeof(address);

    newSocket = accept(getSocket()->getSock(), (struct sockaddr *)&address, (socklen_t *)&arrLen);
    if (newSocket < 0)
    {
        cerr << "Failed to accept connection!" << endl;
        return;
    }

    memset(buffer, 0, sizeof(buffer)); // Clear buffer
    read(newSocket, buffer, 30000);
}

void HDE::TestServer::processRequestAndRespond()
{
    string request(buffer);
    cout << "Received Request:\n"
         << request << endl;

    stringstream requestStream(request);
    string method, path, protocol;

    requestStream >> method >> path >> protocol;
    this->method = method;

    if (method == "GET")
    {
        cout << "Handling GET request" << endl;
        handleTrainingRequest(newSocket);
    }
    else if (method == "POST")
    {
        cout << "Handling POST request" << endl;
        handlePostRequest(request);
    }
    else
    {
        cout << "Unsupported request method: " << method << endl;
        sendErrorResponse();
    }
}

void HDE::TestServer::handleTrainingRequest(int clientSocket)
{
    TrainingDatabase db("./NN/mnist/data/training_data.dat", "./NN/mnist/data/probabilities.dat");
    auto [trainingRecords, probabilityData] = db.loadAllTrainingData();

    string jsonResponse = "{";

    jsonResponse += "\"probabilities\": [";
    for (size_t i = 0; i < probabilityData.size(); i++)
    {
        jsonResponse += "[";
        for (size_t j = 0; j < probabilityData[i].size(); j++)
        {
            jsonResponse += to_string(probabilityData[i][j]);
            if (j < probabilityData[i].size() - 1)
                jsonResponse += ",";
        }
        jsonResponse += "]";
        if (i < probabilityData.size() - 1)
            jsonResponse += ",";
    }
    jsonResponse += "],";

    jsonResponse += "\"trainingHistory\": [";
    for (size_t i = 0; i < trainingRecords.size(); ++i)
    {
        const auto &record = trainingRecords[i];

        jsonResponse += "{";
        jsonResponse += "\"epoch\": " + to_string(record.epoch) + ",";
        jsonResponse += "\"loss\": " + to_string(record.loss) + ",";
        jsonResponse += "\"weights\": [";

        for (size_t j = 0; j < record.weights.size(); ++j)
        {
            jsonResponse += to_string(record.weights[j]);
            if (j < record.weights.size() - 1)
                jsonResponse += ",";
        }

        jsonResponse += "]";
        jsonResponse += "}";

        if (i < trainingRecords.size() - 1)
            jsonResponse += ",";
    }
    jsonResponse += "]";
    jsonResponse += "}";

    string response =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: http://localhost:3000\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Content-Length: " +
        to_string(jsonResponse.length()) + "\r\n"
                                                "\r\n" +
        jsonResponse;

    send(clientSocket, response.c_str(), response.length(), 0);
}

void HDE::TestServer::handlePostRequest(const string &request)
{
    string response =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: http://localhost:3000\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Content-Length: 40\r\n"
        "\r\n"
        "{\"message\": \"Dummy training data saved!\"}";

    send(newSocket, response.c_str(), response.length(), 0);
}

void HDE::TestServer::sendErrorResponse()
{
    string response =
        "HTTP/1.1 400 Bad Request\r\n"
        "Content-Type: text/plain\r\n"
        "Access-Control-Allow-Origin: http://localhost:3000\r\n"
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Content-Length: 15\r\n"
        "\r\n"
        "Invalid Request";

    send(newSocket, response.c_str(), response.length(), 0);
}

void HDE::TestServer::closeConnection()
{
    close(newSocket);
}

void HDE::TestServer::startServer()
{
    while (true)
    {
        cout << "Waiting for client connections..." << endl;
        acceptClientConnection();          
        processRequestAndRespond(); 
        closeConnection();
        cout << "Finished handling request." << endl;
    }
}
