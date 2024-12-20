#include "TestServer.hpp"

using namespace std;

HDE::TestServer::TestServer() : SimpleServer{AF_INET, SOCK_STREAM, 0, 80, INADDR_ANY, 10} {
    start();
}

void HDE::TestServer::acceptor() {
    struct sockaddr_in address = getSocket()->getAddress();
    int arrLen = sizeof(address);
    newSocket = accept(getSocket()->getSock(), (struct sockaddr *)&address, (socklen_t *)&arrLen);
    read(newSocket, buffer, 30000);
}

void HDE::TestServer::handler() {
    cout << buffer << endl;
}

void HDE::TestServer::responder() {
    char *hello = "hello world";
    write(newSocket, hello, strlen(hello));
    close(newSocket);
}

void HDE::TestServer::start() {
    while (true) {
        cout << "Waiting for u bruh: " << endl;
        acceptor();
        handler();
        responder();
        cout << "DOne broski" << endl;
    }
}