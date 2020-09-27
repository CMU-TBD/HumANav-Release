#include <iostream>
#include <sys/socket.h>
#include <netdb.h>
#include <arpa/inet.h>

#define PORT_SEND 6000
#define PORT_RECV 6001

using namespace std;

void close_recv_socket();
void send_to_robot();
int establish_sender_connection();
int establish_receiver_connection();

int main(int argc, char *argv[])
{
    cout << "Demo Joystick Interface in C++ (Random planner)" << endl;
    /// TODO: add suport for reading .ini param files from C++
    cout << "Initiated joystick at localhost:" << PORT_SEND << endl;
    if (establish_sender_connection() < 0)
        return -1;
    if (establish_receiver_connection() < 0)
        return -1;
    return 0;
}

void close_recv_socket()
{
}
void send_to_robot()
{
    /// TODO: let the robot_sender/receiver_sockets be globals
    //        so they can be used throughout the program
}

int establish_sender_connection()
{
    // "client" connection
    int robot_sender_socket = 0;
    struct sockaddr_in robot_addr;
    // struct hostent *hent;
    if ((robot_sender_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        cout << "\033[31m"
             << "Failed socket creation\n"
             << "\033[00m" << endl;
        return -1;
    }
    // bind the host and port to the socket
    robot_addr.sin_family = AF_INET;
    robot_addr.sin_port = htons(PORT_SEND);
    // robot_addr.sin_addr.s_addr = "goosinator";
    // Convert localhost from text to binary form
    if (inet_pton(AF_INET, "127.0.0.1", &(robot_addr.sin_addr.s_addr)) <= 0)
    {
        cout << "\nInvalid address/Address not supported \n";
        return -1;
    }
    if (connect(robot_sender_socket, (struct sockaddr *)&robot_addr,
                sizeof(robot_addr)) < 0)
    {
        cout << "\033[31m"
             << "Unable to connect to robot\n"
             << "\033[00m"
             << "Make sure you have a simulation instance running" << endl;
        return -1;
    }
    // success!
    cout << "\033[32m"
         << "Joystick->Robot connection established"
         << "\033[00m" << endl;
    return 0;
}
int establish_receiver_connection()
{
    int robot_receiver_socket, client;
    struct sockaddr_in robot_addr;
    int opt = 1;
    if ((robot_receiver_socket = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        exit(EXIT_FAILURE);
    }
    if (setsockopt(robot_receiver_socket, SOL_SOCKET,
                   SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0)
    {
        exit(EXIT_FAILURE);
    }
    robot_addr.sin_family = AF_INET;
    robot_addr.sin_addr.s_addr = INADDR_ANY;
    robot_addr.sin_port = htons(PORT_RECV);

    if (bind(robot_receiver_socket, (struct sockaddr *)&robot_addr,
             sizeof(robot_addr)) < 0)
    {
        exit(EXIT_FAILURE);
    }
    if (listen(robot_receiver_socket, 1) < 0)
    {
        exit(EXIT_FAILURE);
    }
    int addrlen = sizeof(robot_receiver_socket);
    if ((client = accept(robot_receiver_socket, (struct sockaddr *)&robot_addr,
                         (socklen_t *)&addrlen)) < 0)
    {
        exit(EXIT_FAILURE);
    }
    // success!
    cout << "\033[32m"
         << "Robot---->Joystick connection established"
         << "\033[00m" << endl;
    return 0;
}
