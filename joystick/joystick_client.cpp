#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include <cstdlib> // rand()
// external json parser (https://github.com/nlohmann/json)
#include "joystick_cpp/json.hpp"
#include "joystick_cpp/sockets.hpp"
#include "joystick_cpp/episode.hpp"
#include "joystick_cpp/agents.hpp"
#include "joystick_cpp/sim_state.hpp"

using namespace std;
using json = nlohmann::json; // for convenience

void get_all_episode_names(vector<string> &episodes);
void get_episode_metadata(Episode &ep);
void joystick_sense(bool &robot_on, unordered_map<float, SimState> &hist);
void joystick_plan(unordered_map<float, SimState> &hist);
void joystick_act();
void update_loop();

int main(int argc, char *argv[])
{
    srand(1); // seed random number generator
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
        // would-be init control pipeline
        update_loop();
    }
    // once completed all episodes, close socket connections
    close_sockets(sender_fd, receiver_fd);
    return 0;
}

void update_loop()
{
    unordered_map<float, SimState> sim_state_hist;
    bool robot_on = true;
    while (robot_on)
    {
        joystick_sense(robot_on, sim_state_hist);
        joystick_plan(sim_state_hist);
        joystick_act();
    }
}
void joystick_sense(bool &robot_on, unordered_map<float, SimState> &hist)
{
    vector<char> raw_data;
    listen_once(raw_data);
    // process the raw_data into a sim_state
    json sim_state_json = json::parse(raw_data);
    SimState new_state = SimState::construct_from_json(sim_state_json);
    // the new time from the simulator
    float current_time = new_state.get_sim_t();
    // update robot running status
    robot_on = new_state.get_robot_status();
    // add new sim_state to the history
    hist.insert({current_time, new_state});
}
void joystick_plan(unordered_map<float, SimState> &hist)
{
    // This is left as an exercise to the reader
    return;
}
void send_cmd(const float v, const float w)
{
    json message;
    // Recall, commands are sent as list of lists where inner lists
    // form the commands v & w, but the outer list contains these commands
    // in case multiple should be sent across a single update (ie. when
    // simulator dt and joystick dt don't match)
    message["j_input"] = {{v, w}};
    send_to_robot(message.dump());
}
void joystick_act()
{
    // Currently send random commands
    const float max_v = 1.2;
    const float max_w = 1.1;
    const int p = 100; // 2 decimal places of precision
    float rand_v = ((rand() % int(p * max_v)) / float(p));
    float rand_w = ((rand() % int(p * max_w)) / float(p));
    send_cmd(rand_v, rand_w);
}
void get_all_episode_names(vector<string> &episodes)
{
    int data_len;
    vector<char> raw_data;
    data_len = listen_once(raw_data);
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