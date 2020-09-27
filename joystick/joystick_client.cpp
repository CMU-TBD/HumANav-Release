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

using namespace std;

void get_all_episode_names(vector<string> &episodes);
void close_recv_socket();
void send_to_robot();
int listen_once(const struct sockaddr_in &addr,
                const int &receiver_fd);
int init_send_conn(struct sockaddr_in &robot_addr,
                   int &sender_fd);
int init_recv_conn(struct sockaddr_in &robot_addr,
                   int &receiver_fd);

int main(int argc, char *argv[])
{
    cout << "Demo Joystick Interface in C++ (Random planner)" << endl;
    /// TODO: add suport for reading .ini param files from C++
    cout << "Initiated joystick at localhost:" << PORT_SEND << endl;
    // establish socket that sends data to robot
    int sender_fd = 0;
    struct sockaddr_in sender_addr;
    if (init_send_conn(sender_addr, sender_fd) < 0)
        return -1;
    // establish socket that receives data from robot
    int receiver_fd = 0;
    struct sockaddr_in receiver_addr;
    if (init_recv_conn(receiver_addr, receiver_fd) < 0)
        return -1;
    listen_once(receiver_addr, receiver_fd);
    // vector<string> episode_names;
    // get_all_episode_names(&receiver_addr, &receiver_fd, &episode_names);
    // once completed all episodes, close socket connections
    close(sender_fd);
    close(receiver_fd);
    return 0;
}

int conn_recv(const int &client_fd, string &data,
              const int buffer_size = 128)
{
    int response_len = 0;
    char buffer[buffer_size];

    while (true)
    {
        memset(buffer, 0, buffer_size); // clear buffer
        int chunk_amnt = recv(client_fd, buffer, sizeof(buffer), 0);
        if (chunk_amnt < 0)
        {
            perror("recv() error");
            break;
        }
        response_len += chunk_amnt;
        // append newly received chunk to overall data
        data += string(buffer);
    }
    return response_len;
}

void get_all_episode_names(struct sockaddr_in &addr,
                           const int &receiver_fd,
                           vector<string> &episodes)
{
}

void close_recv_socket() {}
void send_to_robot() {}
int listen_once(const struct sockaddr_in &addr,
                const int &receiver_fd)
{
    int client_fd;
    int addr_len = sizeof(receiver_fd);
    if ((client_fd = accept(receiver_fd, (struct sockaddr *)&addr,
                            (socklen_t *)&addr_len)) < 0)
    {
        cout << "\033[31m"
             << "Unable to accept connection\n"
             << "\033[00m" << endl;
        return -1;
    }
    string data = ""; // incoming data
    int response_len = conn_recv(client_fd, data);
    cout << "Received " << response_len << " from server:" << endl
         << data << endl;
    return 0;
}
int init_send_conn(struct sockaddr_in &robot_addr,
                   int &robot_sender_fd)
{
    // "client" connection
    if ((robot_sender_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        cout << "\033[31m"
             << "Failed socket creation\n"
             << "\033[00m" << endl;
        return -1;
    }
    // bind the host and port to the socket
    robot_addr.sin_family = AF_INET;
    robot_addr.sin_port = htons(PORT_SEND);
    // Convert localhost from text to binary form
    if (inet_pton(AF_INET, "127.0.0.1", &robot_addr.sin_addr.s_addr) <= 0)
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
    cout << "\033[32m"
         << "Joystick->Robot connection established"
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
        exit(EXIT_FAILURE);
    }
    if (setsockopt(robot_receiver_fd, SOL_SOCKET,
                   SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt)) < 0)
    {
        exit(EXIT_FAILURE);
    }
    robot_addr.sin_family = AF_INET;
    robot_addr.sin_addr.s_addr = INADDR_ANY;
    robot_addr.sin_port = htons(PORT_RECV);
    if (bind(robot_receiver_fd, (struct sockaddr *)&robot_addr,
             sizeof(robot_addr)) < 0)
    {
        exit(EXIT_FAILURE);
    }
    if (listen(robot_receiver_fd, 1) < 0)
    {
        exit(EXIT_FAILURE);
    }
    int addr_len = sizeof(robot_receiver_fd);
    if ((client = accept(robot_receiver_fd, (struct sockaddr *)&robot_addr,
                         (socklen_t *)&addr_len)) < 0)
    {
        exit(EXIT_FAILURE);
    }
    // success!
    cout << "\033[32m"
         << "Robot---->Joystick connection established"
         << "\033[00m" << endl;
    return 0;
}
