#include "ListeningSocket.hpp"

HDE::ListeningSocket::ListeningSocket::ListeningSocket(int domain, int service, int protocol, int port, u_long interface, int backlog) : 
                    BindingSocket{domain, service, protocol, port, interface}, backlog{backlog} {
                        startListening();
                        testConnection(listening);
                    }

void HDE::ListeningSocket::startListening() {
    listening = listen(getSock(), backlog);
}

int HDE::ListeningSocket::getListening() {
    return listening;
}

int HDE::ListeningSocket::getBacklog() {
    return backlog;
}