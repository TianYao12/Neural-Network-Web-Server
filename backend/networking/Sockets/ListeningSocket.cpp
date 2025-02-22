#include "ListeningSocket.hpp"

// Inherits from BindingSocket and enables listening for incoming connections (listen()).
// backlog specifies how many clients can wait in queue
HDE::ListeningSocket::ListeningSocket::ListeningSocket(int domain, int service, int protocol, int port, u_long interface, int backlog) : BindingSocket{domain, service, protocol, port, interface}, backlog{backlog}
{
    startListening();
    validateSocketOperation(listening);
}

void HDE::ListeningSocket::startListening()
{
    listening = listen(getSock(), backlog);
}

// Below are gettors and settors
int HDE::ListeningSocket::getListening()
{
    return listening;
}

int HDE::ListeningSocket::getBacklog()
{
    return backlog;
}