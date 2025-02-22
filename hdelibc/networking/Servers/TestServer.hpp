#ifndef TEST_SERVER_HPP
#define TEST_SERVER_HPP

#include <stdio.h>
#include <string.h>
#include "SimpleServer.hpp"
#include "../Database/Database.hpp"

namespace HDE
{
    class TestServer : public SimpleServer
    {
        int newSocket;
        std::string method;
        char buffer[30000] = {0};
        void acceptClientConnection() override;
        void processRequestAndRespond() override;
        void closeConnection() override;
        void handleTrainingRequest(int);
        void handlePostRequest(const std::string &);
        void sendErrorResponse();

    public:
        TestServer();
        void start() override;
        std::unordered_map<std::string, std::string> parseHeaders(const std::string &request);
        std::string jsonResponse(const std::string &message);
    };
};

#endif