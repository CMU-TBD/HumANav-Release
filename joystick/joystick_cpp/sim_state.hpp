#ifndef SIMSTATE_H
#define SIMSTATE_H

#include <unordered_map>
#include "agents.hpp"

class SimState
{
public:
    SimState(AgentState &r, unordered_map<string, AgentState> &peds,
             float sim_time, string &episode_name, float time_budget,
             bool rob_on)
    {
        robot = r;
        robot_on = rob_on;
        running_episode = episode_name;
        pedestrians = peds;
        simulator_time = sim_time;
        max_time_s = time_budget;
    }
    const AgentState get_robot() const { return robot; }
    const bool get_robot_status() const { return robot_on; }
    const float get_sim_t() const { return simulator_time; }
    const float get_max_sim_t() const { return max_time_s; }
    const string get_episode_title() const { return running_episode; }
    const unordered_map<string, AgentState> get_pedestrians() const
    {
        return pedestrians;
    }
    static SimState construct_from_json(const json &json_data)
    {
        // not used for the SimStates for now
        auto &env = json_data["environment"];
        unordered_map<string, AgentState> peds =
            AgentState::construct_from_dict(json_data["pedestrians"]);
        float t_budget = json_data["episode_max_time"];
        float t = json_data["sim_t"];
        bool rob_on = json_data["robot_on"];
        string ep_name = json_data["episode_name"];
        // NOTE there is an assumption that there is only one robot in
        // the simulator at once, and its *name* is "robot_agent"
        auto &sim_robots = json_data["robots"];
        // currently only one robot is supported
        auto &robot_json = sim_robots["robot_agent"];
        AgentState r = AgentState::construct_from_json(robot_json);
        // construct the new SimState and return it
        return SimState(r, peds, t, ep_name, t_budget, rob_on);
    }

private:
    AgentState robot;
    bool robot_on;
    float simulator_time;
    unordered_map<string, AgentState> pedestrians;
    string running_episode;
    float max_time_s;
};

#endif
