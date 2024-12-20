#ifndef SIMPLE_SOCKET
#define SIMPLE_SOCKET

#include <stdio.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <iostream>

namespace HDE {
    class SimpleSocket {
        struct sockaddr_in address;
        int sock, connection;

        public:
        SimpleSocket(int, int, int, int, u_long);
        virtual int connectToNetwork(int, struct sockaddr_in) = 0;
        void testConnection(int);

        struct sockaddr_in getAddress();
        int getSock();
        int getConnection();
        void setConnection(int);
    };
};

#endif