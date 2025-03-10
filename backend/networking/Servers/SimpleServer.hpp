#ifndef SIMPLE_SERVER_HPP
#define SIMPLE_SERVER_HPP

#include <stdio.h>
#include <unistd.h>
#include "../backend-networking.hpp"

namespace HDE
{
    class SimpleServer
    {
        ListeningSocket *socket;
        virtual void acceptClientConnection() = 0;
        virtual void processRequestAndRespond() = 0;
        virtual void closeConnection() = 0;

    public:
        SimpleServer(int, int, int, int, u_long, int);
        virtual void startServer() = 0;

        // Below are gettors and settors
        ListeningSocket *getSocket();
    };
};

#endif