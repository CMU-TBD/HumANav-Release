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
void get_episode_metadata(const string &title, Episode &ep);
void joystick_sense(bool &robot_on, float &sim_time,
                    unordered_map<float, SimState> &hist, const float max_time);
void joystick_plan(const bool robot_on, const float sim_t,
                   unordered_map<float, SimState> &hist);
void joystick_act(const bool robot_on, const float sim_t,
                  unordered_map<float, SimState> &hist);
void update_loop(Episode &ep);

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
    for (auto &title : episode_names)
    {
        Episode current_episode;
        get_episode_metadata(title, current_episode);
        // would-be init control pipeline
        update_loop(current_episode);
    }
    cout << "\033[32m"
         << "Finished all episodes"
         << "\033[00m" << endl;
    // once completed all episodes, close socket connections
    close_sockets(sender_fd, receiver_fd);
    return 0;
}

void update_loop(Episode &ep)
{
    unordered_map<float, SimState> sim_state_hist;
    bool robot_on = true;
    float sim_t = 0;
    const float max_t = ep.get_time_budget();
    cout << "\033[35m"
         << "Starting episode: " << ep.get_title() << "\033[00m" << endl;
    while (robot_on)
    {
        // gather information about the world state based off the simulator
        joystick_sense(robot_on, sim_t, sim_state_hist, max_t);
        // create a plan for the next steps of the trajectory
        joystick_plan(robot_on, sim_t, sim_state_hist);
        // send commands to the robot to execute
        joystick_act(robot_on, sim_t, sim_state_hist);
    }
    cout << "\n\033[32m"
         << "Finished episode: " << ep.get_title() << "\033[00m" << endl;
}

void joystick_sense(bool &robot_on, float &current_time,
                    unordered_map<float, SimState> &hist, const float max_time)
{
    vector<char> raw_data;
    // send keyword (trigger sense action) and await response
    if (send_to_robot("sense") >= 0 && listen_once(raw_data) >= 0)
    {
        // process the raw_data into a sim_state
        json sim_state_json = json::parse(raw_data);
        SimState new_state = SimState::construct_from_json(sim_state_json);
        // the new time from the simulator
        current_time = new_state.get_sim_t();
        // update robot running status
        robot_on = new_state.get_robot_status();
        // add print output:
        cout << "\033[36m"
             << "\33[2K" // clear old line
             << "Updated state of the world for time = " << current_time
             << " out of " << max_time << "\033[00m\r" << flush;
        // add new sim_state to the history
        hist.insert({current_time, new_state});
    }
    else
    {
        // connection failure, power off the robot
        robot_on = false;
        return;
    }
}
void joystick_plan(const bool robot_on, const float sim_t,
                   unordered_map<float, SimState> &hist)
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
void joystick_act(const bool robot_on, const float sim_t,
                  unordered_map<float, SimState> &hist)
{
    string termination_cause;
    if (hist.find(sim_t) != hist.end())
    {
        SimState *sim_state = &hist.at(sim_t); // pointer (not deepcopy)
        termination_cause = sim_state->get_termination_cause();
    }
    else
        termination_cause = "Disconnection";
    if (robot_on)
    {
        // Currently send random commands
        const float max_v = 1.2;
        const float max_w = 1.1;
        const int p = 100; // 2 decimal places of precision
        float rand_v = ((rand() % int(p * max_v)) / float(p));
        float rand_w = ((rand() % int(p * max_w)) / float(p));
        send_cmd(rand_v, rand_w);
    }
    else
    {
        cout << "\nPowering off joystick, robot terminated with: "
             << termination_cause << endl;
    }
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

void get_episode_metadata(const string &ep_name, Episode &ep)
{
    cout << "Waiting for episode: " << ep_name << endl;
    int ep_len;
    vector<char> raw_data;
    ep_len = listen_once(raw_data);
    // parse the episode_names raw data from the connection
    json metadata = json::parse(raw_data);
    // TODO: move to static Episode class
    ep = Episode::construct_from_json(metadata);
    // notify the robot that all the metadata has been obtained
    // to begin the simualtion
    send_to_robot("ready");
}