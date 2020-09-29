#ifndef EPISODE_H
#define EPISODE_H

#include <string>
#include <vector>
#include <unordered_map>
#include "agents.hpp"

using namespace std;

struct env_t
{
    float dx_scale;
    vector<float> room_center;
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
                            vector<float> &center, float scale)
    {
        dx_scale = scale;
        building_traversible = building_trav;
        human_traversible = human_trav;
        room_center = center;
    }
};

class Episode
{
public:
    Episode() {}
    Episode(string &t, vector<vector<int>> &building_trav,
            vector<vector<int>> &human_trav, vector<float> &center,
            float scale, unordered_map<string, AgentState> &a, float t_budget,
            vector<float> &r_start, vector<float> &r_goal)
    {
        title = t;
        env.update_environment(building_trav, human_trav, center, scale);
        agents = a;
        max_time_s = t_budget;
        robot_start = r_start;
        robot_goal = r_goal;
    }
    string get_title() const { return title; }
    vector<float> get_robot_start() const { return robot_start; }
    vector<float> get_robot_goal() const { return robot_goal; }
    unordered_map<string, AgentState> get_agents() const { return agents; }
    float get_time_budget() const { return max_time_s; }
    env_t get_environment() const { return env; }
    void print() const
    {
        cout << "Episode: " << get_title() << endl;
        cout << "Max time: " << get_time_budget() << endl;
        float start_x = get_robot_start()[0];
        float start_y = get_robot_start()[1];
        float start_theta = get_robot_start()[2];
        cout << "Robot start: " << start_x << ", " << start_y
             << ", " << start_theta << endl;
        float goal_x = get_robot_goal()[0];
        float goal_y = get_robot_goal()[1];
        float goal_theta = get_robot_goal()[2];
        cout << "Robot goal: " << goal_x << ", " << goal_y
             << ", " << goal_theta << endl;
        }

private:
    string title;
    env_t env;
    unordered_map<string, AgentState> agents;
    float max_time_s;
    vector<float> robot_start, robot_goal;
};

#endif
