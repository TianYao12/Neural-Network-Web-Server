#ifndef CONNECTING_SOCKET
#define CONNECTING_SOCKET
#include <stdio.h>
#include "SimpleSocket.hpp"

namespace HDE {
    class ConnectingSocket : public SimpleSocket {
        public:
        ConnectingSocket(int, int, int, int, u_long);
        int connectToNetwork(int, struct sockaddr_in) override;
    };
};

#endif