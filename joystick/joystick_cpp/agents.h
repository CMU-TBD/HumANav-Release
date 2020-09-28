#ifndef AGENTS_H
#define AGENTS_H

#include <string>
#include <vector>

using namespace std;

/* PEDESTRIAN CLASS */

class AgentState
{
private:
    AgentState(string &n, vector<float> &current_pos3, float r)
    {
        name = n;
        current_config = current_pos3;
        radius = r;
    }
    string name;
    vector<float> current_config;
    float radius;

public:
    string get_name() const { return name; }
    float get_radius() const { return radius; }
    vector<float> get_current_config() const { return current_config; }
};

#endif
