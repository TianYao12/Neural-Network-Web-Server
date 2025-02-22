#include "SimpleSocket.hpp"

/**
 * @brief Creates a raw socket using socket()
 *
 * @param domain    The communication domain (address family) for the socket.
 *                  Examples: AF_INET:  IPv4: Internet protocols, AF_INET6: IPv6 Internet protocols, AF_UNIX:  Local communication (Unix domain)
 *
 * @param service   The socket type, determining the communication semantics.
 *                  Examples: SOCK_STREAM is usually used with TCP, SOCK_DGRAM is usually used with UDP
 *
 * @param protocol  The protocol to be used with the socket. 0 selects the default protocol based on the specified domain and service.
 *                  If IPv4 and SOCK_STREAM are selected, a protocol of 0 means IPPROTO_TCP, representing TCP
 *
 * @param port      Specific protocols can be specified, such as IPPROTO_TCP for TCP or IPPROTO_UDP for UDP.
 *                  The port number on which the socket will communicate.
 *
 * @param interface The IP address of the network interface to bind to, provided in host byte order and will be converted to network byute order using htonl()
 *                  Examples: INADDR_ANY binds to all available interfaces; specific IP addresses bind to a specific interface
 */
HDE::SimpleSocket::SimpleSocket(int domain, int service, int protocol, int port, u_long interface)
{
    address.sin_family = domain;
    address.sin_port = htons(port);
    address.sin_addr.s_addr = htonl(interface);

    sock = socket(domain, service, protocol);
    validateSocketOperation(sock);
}

void HDE::SimpleSocket::validateSocketOperation(int sockOrConnection)
{
    if (sockOrConnection < 0)
    {
        perror("Connection failed broski");
        exit(EXIT_FAILURE);
    }
}

// Below are gettors and settors
struct sockaddr_in HDE::SimpleSocket::getAddress()
{
    return address;
}

int HDE::SimpleSocket::getConnection()
{
    return connection;
}

int HDE::SimpleSocket::getSock()
{
    return sock;
}

void HDE::SimpleSocket::setConnection(int newConnection)
{
    connection = newConnection;
}