#ifndef LISTENING_SOCKET_HPP
#define LISTENING_SOCKET_HPP

#include <stdio.h>
#include "BindingSocket.hpp"

namespace HDE {
    class ListeningSocket : public BindingSocket {
        int backlog, listening;
        public:
        ListeningSocket(int, int, int, int, u_long, int);
        void startListening();
        int getListening();
        int getBacklog();
    };
};
#endif