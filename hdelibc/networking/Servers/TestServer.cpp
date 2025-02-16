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
    start();
}

void HDE::TestServer::acceptor()
{
    struct sockaddr_in address = getSocket()->getAddress();
    int arrLen = sizeof(address);
    newSocket = accept(getSocket()->getSock(), (struct sockaddr *)&address, (socklen_t *)&arrLen);
    if (newSocket < 0) {
        cerr << "Failed to accept connection!" << endl;
        return;
    }
    memset(buffer, 0, sizeof(buffer)); // Clear buffer
    read(newSocket, buffer, 30000);
}

void HDE::TestServer::handler()
{
    string request(buffer);
    cout << "Received Request:\n" << request << endl;

    stringstream requestStream(request);
    string method, path, protocol;
    requestStream >> method >> path >> protocol;

    this->method = method;

    if (method == "GET") {
        cout << "Handling GET request" << endl;
        handleTrainingRequest(newSocket);
    } 
    else if (method == "POST") {
        cout << "Handling POST request" << endl;
        handlePostRequest(request);
    } 
    else if (method == "OPTIONS") { 
        cout << "Handling OPTIONS request" << endl;
        sendOptionsResponse(); 
    }
    else {
        cout << "Unsupported request method: " << method << endl;
        sendErrorResponse();
    }
}

void HDE::TestServer::sendOptionsResponse() {
    string response =
        "HTTP/1.1 204 No Content\r\n"
        "Access-Control-Allow-Origin: http://localhost:3000\r\n" 
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Content-Length: 0\r\n"
        "\r\n";

    send(newSocket, response.c_str(), response.length(), 0);
}

void HDE::TestServer::handleTrainingRequest(int clientSocket) {
    TrainingDatabase db("../training_data.db");
    vector<vector<double>> trainingData = db.loadTrainingData();
    string jsonResponse = "{ \"epochs\": [";

    for (size_t i = 0; i < trainingData.size(); i++) {
        jsonResponse += "[";
        for (size_t j = 0; j < trainingData[i].size(); j++) {
            jsonResponse += to_string(trainingData[i][j]);
            if (j < trainingData[i].size() - 1) jsonResponse += ",";
        }
        jsonResponse += "]";
        if (i < trainingData.size() - 1) jsonResponse += ",";
    }

    jsonResponse += "] }";

    string response =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: http://localhost:3000\r\n" 
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
        "Access-Control-Allow-Headers: Content-Type\r\n"
        "Content-Length: " + to_string(jsonResponse.length()) + "\r\n"
        "\r\n" +
        jsonResponse;

    send(clientSocket, response.c_str(), response.length(), 0);
}

void HDE::TestServer::handlePostRequest(const string &request)
{
    srand(time(0));

    int epoch = rand() % 100; // Random epoch between 0-99
    double loss = static_cast<double>(rand()) / RAND_MAX; // Random loss between 0.0 and 1.0
    double accuracy = static_cast<double>(rand() % 100); // Random accuracy between 0 and 100

    vector<double> weights;
    for (int i = 0; i < 10; i++) // Generate 10 random weights
    {
        weights.push_back(static_cast<double>(rand()) / RAND_MAX);
    }

    TrainingDatabase db("../training_data.db");
    db.saveTrainingData(epoch, loss, accuracy, weights);

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

void HDE::TestServer::responder()
{
    close(newSocket); 
}

void HDE::TestServer::start()
{
    while (true)
    {
        cout << "Waiting for client connections..." << endl;
        acceptor();  // Accepts a client connection
        handler();   // Process request
        responder(); // Close connection
        cout << "Finished handling request." << endl;
    }
}
