#ifndef EPISODE_H
#define EPISODE_H

#include <string>
#include <vector>
#include <unordered_map>
#include "agents.h"

using namespace std;

struct env_t
{
    float dx_scale;
    vector<int> room_center;
    vector<vector<int>> building_traversible;
    vector<vector<int>> human_traversible;
    env_t()
    {
        dx_scale = 0;
        room_center = {0, 0, 0};
        building_traversible = {};
        human_traversible = {};
    }
    void update_environment(vector<vector<int>> &building_trav,
                            vector<vector<int>> &human_trav,
                            vector<int> &center, float scale)
    {
        dx_scale = scale;
        building_traversible = building_trav;
        human_traversible = human_trav;
        room_center = center;
    }
};

class Episode
{
private:
    Episode(string &t, vector<vector<int>> &building_trav,
            vector<vector<int>> &human_trav, vector<int> &center,
            float scale, unordered_map<string, AgentState> &a, float t_budget,
            vector<int> &r_start, vector<int> &r_goal)
    {
        title = t;
        env.update_environment(building_trav, human_trav, center, scale);
        agents = a;
        max_time = t_budget;
        robot_start = r_start;
        robot_goal = r_goal;
    }
    string title;
    env_t env;
    unordered_map<string, AgentState> agents;
    float max_time;
    vector<int> robot_start, robot_goal;

public:
    string get_title() const { return title; }
    vector<int> get_robot_start() const { return robot_start; }
    vector<int> get_robot_goal() const { return robot_goal; }
    unordered_map<string, AgentState> get_agents() const { return agents; }
    float get_time_budget() const { return max_time; }
    env_t get_environment() const { return env; }
};

#endif
