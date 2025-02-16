#ifndef SIMPLE_SERVER_HPP
#define SIMPLE_SERVER_HPP

#include <stdio.h>
#include <unistd.h>
#include "../hdelibc-networking.hpp"

namespace HDE {
    class SimpleServer {
        ListeningSocket *socket;
        virtual void acceptor() = 0;
        virtual void handler() = 0;
        virtual void responder() = 0;

        public:
        SimpleServer(int, int, int, int, u_long, int);
        virtual void start() = 0;

        // Below are gettors and settors
        ListeningSocket *getSocket();
    };
};

#endif