#ifndef SOCKET_H
#define SOCKET_H

#include <iostream>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <vector>
#include <cstring>
#include <string>

#define PORT_SEND 6000
#define PORT_RECV (PORT_SEND + 1)
#define LOCALHOST "127.0.0.1"

using namespace std;

int send_to_robot(const string &message);
int listen_once(vector<char> &data);
int init_send_conn(struct sockaddr_in &addr, int &sender_fd);
int init_recv_conn(struct sockaddr_in &addr, int &receiver_fd);
void close_sockets(const int &sender_fd, const int &receiver_fd);

// global socket information to use throughout the program
struct sockaddr_in sender_addr;
int sender_fd = 0;
struct sockaddr_in receiver_addr;
int receiver_fd = 0;

// global params (TODO: read from .ini file)
const bool verbose = false;
// number of times the sockets were connected
size_t num_connections = 0;

int conn_recv(const int conn_fd, vector<char> &data, const int buf_size = 128)
{
    int response_len = 0;
    // does not need to be cleared as we only read what is set
    char buffer[buf_size];
    data.clear();
    while (true)
    {
        int chunk_amnt = recv(conn_fd, buffer, sizeof(buffer), 0);
        if (chunk_amnt <= 0)
            break;
        response_len += chunk_amnt;
        // append newly received chunk to overall data
        for (size_t i = 0; i < chunk_amnt; i++)
            data.push_back(buffer[i]);
    }
    return response_len;
}

void close_sockets(const int &send_fd, const int &recv_fd)
{
    close(send_fd);
    close(recv_fd);
}
int send_to_robot(const string &message)
{
    // create the TCP/IP socket and connect to the server (robot)
    if (init_send_conn(sender_addr, sender_fd) < 0)
        return -1;
    const void *buf = message.c_str();
    const size_t buf_len = message.size();
    int amnt_sent;
    if ((amnt_sent = send(sender_fd, buf, buf_len, 0)) < 0)
    {
        perror("\nsend() error: ");
        return -1;
    }
    close(sender_fd);
    if (verbose)
        cout << "sent " << amnt_sent << " bytes: "
             << "\"" << message << "\"" << endl;
    return 0;
}
int listen_once(vector<char> &data)
{
    int client_fd;
    int addr_len = sizeof(receiver_addr);
    if ((client_fd = accept(receiver_fd, (struct sockaddr *)&receiver_addr,
                            (socklen_t *)&addr_len)) < 0)
    {
        cout << "\033[31m"
             << "Unable to accept connection\n"
             << "\033[00m" << endl;
        return -1;
    }
    int response_len = conn_recv(client_fd, data);
    close(client_fd);
    // TODO: add versbosity check
    if (verbose)
        cout << "\033[36m"
             << "Received " << response_len << " bytes from server"
             << "\033[00m" << endl;
    return 0;
}
int init_send_conn(struct sockaddr_in &robot_addr,
                   int &robot_sender_fd)
{
    // "client" connection
    if ((robot_sender_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("\nsocket() error: ");
        return -1;
    }
    // bind the host and port to the socket
    robot_addr.sin_family = AF_INET;
    robot_addr.sin_port = htons(PORT_SEND);
    // Convert localhost from text to binary form
    if (inet_pton(AF_INET, LOCALHOST, &robot_addr.sin_addr.s_addr) <= 0)
    {
        cout << "\nInvalid address/Address not supported \n";
        return -1;
    }
    if (connect(robot_sender_fd, (struct sockaddr *)&robot_addr,
                sizeof(robot_addr)) < 0)
    {
        cout << "\033[31m"
             << "Unable to connect to robot\n"
             << "\033[00m"
             << "Make sure you have a simulation instance running" << endl;
        return -1;
    }
    // success!
    if (verbose || num_connections < 1)
        cout << "\033[32m"
             << "Robot <-- Joystick (sender) connection established"
             << "\033[00m" << endl;
    return 0;
}
int init_recv_conn(struct sockaddr_in &robot_addr,
                   int &robot_receiver_fd)
{
    int client;
    int opt = 1;
    if ((robot_receiver_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("socket() error");
        exit(EXIT_FAILURE);
    }
    if (setsockopt(robot_receiver_fd, SOL_SOCKET,
                   SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0)
    {
        perror("setsockopt() error");
        exit(EXIT_FAILURE);
    }
    robot_addr.sin_family = AF_INET;
    robot_addr.sin_addr.s_addr = INADDR_ANY;
    robot_addr.sin_port = htons(PORT_RECV);
    if (bind(robot_receiver_fd, (struct sockaddr *)&robot_addr,
             sizeof(robot_addr)) < 0)
    {
        perror("bind() error");
        exit(EXIT_FAILURE);
    }
    if (listen(robot_receiver_fd, 1) < 0)
    {
        perror("listen() error");
        exit(EXIT_FAILURE);
    }
    int addr_len = sizeof(robot_receiver_fd);
    if ((client = accept(robot_receiver_fd, (struct sockaddr *)&robot_addr,
                         (socklen_t *)&addr_len)) < 0)
    {
        perror("accept() error");
        exit(EXIT_FAILURE);
    }
    // success!
    if (verbose || num_connections < 1)
        cout << "\033[32m"
             << "Robot --> Joystick (receiver) connection established"
             << "\033[00m" << endl;
    // update count of the number of times the sockets have been connected
    num_connections++;
    // client should always be nonnegative integer
    return client;
}

#endif
