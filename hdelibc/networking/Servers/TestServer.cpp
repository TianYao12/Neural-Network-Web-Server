#include "TestServer.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
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
    read(newSocket, buffer, 30000);
}

void HDE::TestServer::handler()
{
    string request(buffer);
    cout << "Received Request:\n" << request << endl;

    stringstream requestStream(request);
    string method, path, protocol;
    requestStream >> method >> path >> protocol;

    if (method == "GET") {
        this->method = "GET";
        cout << "That was a GET request" << endl;
    } else if (method == "POST") {
        this->method = "POST";
        size_t bodyStart = request.find("\r\n\r\n"); // Locate start of body
        if (bodyStart != string::npos)
        {
            string body = request.substr(bodyStart + 4);
            cout << "POST Body:\n" << body << endl;


            // Here you can process JSON or form data
        }
    } else {
        cout << "That was a " << method << " request" << endl;
    }
}

void HDE::TestServer::responder(const string& method) {
    string body;
    
    if (method == "GET") {
        body = "{\"message\": \"GET request YOOOO\"}";
    } 
    else if (method == "POST") {
        body = "{\"message\": \"POST request YOOOO\"}";
    } 
    else {
        body = "Method Not Allowed! YOOOO";
    }

    string response = 
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: application/json\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Content-Length: " + to_string(body.length()) + "\r\n"
        "\r\n" +
        body; 

    write(newSocket, response.c_str(), response.length());
    close(newSocket);
}


// loops continuously to accept and handle client connections
void HDE::TestServer::start()
{
    while (true)
    {
        cout << "Waiting for client connections: " << endl;
        acceptor();  // accepts a client connection
        handler();   // process the received request
        responder(method); // send a response to client
        cout << "Finished" << endl;
    }
}