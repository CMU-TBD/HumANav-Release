#include <iostream>
#include <vector>
#include <string>
#include "joystick_cpp/sockets.h"
#include "joystick_cpp/episode.h"
#include "joystick_cpp/agents.h"

using namespace std;

void get_all_episode_names(vector<string> &episodes);
void get_episode_metadata(vector<char> &metadata);

int main(int argc, char *argv[])
{
    cout << "Demo Joystick Interface in C++ (Random planner)" << endl;
    /// TODO: add suport for reading .ini param files from C++
    cout << "Initiated joystick at localhost:" << PORT_SEND << endl;
    // establish socket that sends data to robot
    if (init_send_conn(sender_addr, sender_fd) < 0)
        return -1;
    // establish socket that receives data from robot
    if (init_recv_conn(receiver_addr, receiver_fd) < 0)
        return -1;
    vector<string> episode_names;
    get_all_episode_names(episode_names);
    vector<char> episode_metadata;
    get_episode_metadata(episode_metadata);

    for (auto &ep : episode_names)
    {
        // init control pipeline
        // update_loop();
    }
    // once completed all episodes, close socket connections
    close_sockets(sender_fd, receiver_fd);
    return 0;
}

void get_all_episode_names(vector<string> &episodes)
{
    int ep_len;
    vector<char> raw_data;
    ep_len = listen_once(raw_data);
    // parse the episode names from the raw data
    // episodes = ...
}

void get_episode_metadata(vector<char> &metadata)
{
    int ep_len;
    vector<char> raw_data;
    ep_len = listen_once(raw_data);
    // parse the episode_names raw data from the connection
    // update_knowledge_from_episode(raw_data)
    // episodes = ...
    send_to_robot("ready");
}
