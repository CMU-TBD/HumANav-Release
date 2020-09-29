#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
// external json parser (https://github.com/nlohmann/json)
#include "joystick_cpp/json.hpp"
#include "joystick_cpp/sockets.hpp"
#include "joystick_cpp/episode.hpp"
#include "joystick_cpp/agents.hpp"

using namespace std;
using json = nlohmann::json; // for convenience

void get_all_episode_names(vector<string> &episodes);
void get_episode_metadata(Episode &ep);

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
    // run the episode loop on individual episodes
    for (auto &ep : episode_names)
    {
        Episode current_episode;
        get_episode_metadata(current_episode);
        current_episode.print();
        // would-be init control pipeline
        update_loop();
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
    json ep_data = json::parse(raw_data);
    cout << "Received episodes: [";
    for (auto &s : ep_data["episodes"])
    {
        cout << s;
        episodes.push_back(s);
    }
    cout << "]" << endl;
}

void get_episode_metadata(Episode &ep)
{
    int ep_len;
    vector<char> raw_data;
    ep_len = listen_once(raw_data);
    // parse the episode_names raw data from the connection
    json metadata = json::parse(raw_data);
    // gather data from json
    string title = metadata["episode_name"];
    auto &env = metadata["environment"];
    vector<vector<int>> map_trav = env["map_traversible"];
    vector<vector<int>> h_trav = {}; //  not being sent currently
    vector<float> room_center = env["room_center"];
    float dx_m = 0.05; // TODO: fix map_scale being string-json?
    // float dx_m = env["map_scale"];
    unordered_map<string, AgentState> agents =
        AgentState::construct_from_dict(metadata["pedestrians"]);
    float max_time = metadata["episode_max_time"];
    float sim_t = metadata["sim_t"];
    // NOTE there is an assumption that there is only one robot in the
    // simulator at once, and its *name* is "robot_agent"
    auto &robots = metadata["robots"];
    auto &robot = robots["robot_agent"];
    vector<float> r_start = robot["start_config"];
    vector<float> r_goal = robot["goal_config"];

    ep = Episode(title, map_trav, h_trav, room_center,
                 dx_m, agents, max_time, r_start, r_goal);
    // episodes = ...
    send_to_robot("ready");
}
void send_cmd(const float in_x, const float in_y)
{
    json message;
    message["j_input"] = {in_x, in_y};
    send_to_robot(message.dump());
}
