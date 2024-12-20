#include "SimpleSocket.hpp"

HDE::SimpleSocket::SimpleSocket(int domain, int service, int protocol, int port, u_long interface) {
    address.sin_family = domain;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(interface);

    sock = socket(domain, service, protocol);
    testConnection(sock);
}

void HDE::SimpleSocket::testConnection(int sockOrConnection) {
    if (sockOrConnection < 0) {
        perror("Connection failed broski");
        exit(EXIT_FAILURE);
    }
}

struct sockaddr_in HDE::SimpleSocket::getAddress() {
    return address;
}

int HDE::SimpleSocket::getConnection() {
    return connection;
}

int HDE::SimpleSocket::getSock() {
    return sock;
}

void HDE::SimpleSocket::setConnection(int newConnection) {
    connection = newConnection;
}