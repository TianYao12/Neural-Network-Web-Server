#include "BindingSocket.hpp"

// Inherits from SimpleSocket and binds the socket to an IP and port (bind())
HDE::BindingSocket::BindingSocket(int domain, int service, int protocol, int port, u_long interface) : SimpleSocket{domain, service, protocol, port, interface}
{
    setConnection(connectToNetwork(getSock(), getAddress()));
    validateSocketOperation(getConnection());
};

/**
 * @brief Binds the socket to the specified address and port.
 *
 * @param sock     The socket file descriptor
 * @param address  The sockaddr_in structure containing the IP address and port information
 *
 * @return int     Returns 0 on success, -1 on failure 
 */
int HDE::BindingSocket::connectToNetwork(int sock, struct sockaddr_in address)
{
    return bind(sock, (struct sockaddr *)&address, sizeof(address));
}
