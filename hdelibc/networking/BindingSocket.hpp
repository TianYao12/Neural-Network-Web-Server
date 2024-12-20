#ifndef BINDING_SOCKET
#define BINDING_SOCKET

#include <stdio.h>
#include "SimpleSocket.hpp"

namespace HDE {
    class BindingSocket : public SimpleSocket {
        public:
        BindingSocket(int, int, int, int, u_long);
        int connectToNetwork(int, struct sockaddr_in) override;
    };
};

#endif