#ifndef SIMSTATE_H
#define SIMSTATE_H

#include <unordered_map>
#include "agents.hpp"

class SimState
{
public:
    SimState(AgentState &r, bool rob_on, float sim_time,
             unordered_map<string, AgentState> &peds, string &term_cause)
    {
        robot = r;
        robot_on = rob_on;
        simulator_time = sim_time;
        pedestrians = peds;
        termination_cause = term_cause;
    }
    const AgentState get_robot() const { return robot; }
    const bool get_robot_status() const { return robot_on; }
    const float get_sim_t() const { return simulator_time; }
    const string get_termination_cause() const { return termination_cause; }
    const unordered_map<string, AgentState> get_pedestrians() const
    {
        return pedestrians;
    }
    static SimState construct_from_json(const json &json_data)
    {
        // first and foremost, every python sim_state has these variables:
        bool rob_on = json_data["robot_on"];
        float sim_t = json_data["sim_t"];
        // however, some variables may or may not be included
        string term_cause = ""; // only included if the robot has terminated
        // the remaining variables are included if the robot is still running
        AgentState rob;
        unordered_map<string, AgentState> peds;

        // not used for the SimStates for now
        // auto &env = json_data["environment"];
        if (rob_on)
        {
            peds = AgentState::construct_from_dict(json_data["pedestrians"]);
            // time of capture
            auto &sim_robots = json_data["robots"];
            // currently only one robot exists (and its name is "robot_agent")
            auto &robot_json = sim_robots["robot_agent"];
            rob = AgentState::construct_from_json(robot_json);
        }
        else
        {
            term_cause = json_data["termination_cause"];
        }
        // construct the new SimState and return it
        return SimState(rob, rob_on, sim_t, peds, term_cause);
    }

private:
    AgentState robot;
    bool robot_on;
    float simulator_time;
    unordered_map<string, AgentState> pedestrians;
    string termination_cause;
};

#endif
