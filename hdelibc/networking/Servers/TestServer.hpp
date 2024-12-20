#ifndef TEST_SERVER_HPP
#define TEST_SERVER_HPP

#include <stdio.h>
#include "SimpleServer.hpp"

namespace HDE {
    class TestServer : public SimpleServer {
        int newSocket;
        char buffer[30000] = {0};
        void acceptor() override;
        void handler() override;
        void responder() override;

        public:
        TestServer();
        void start() override;
    };
};

#endif