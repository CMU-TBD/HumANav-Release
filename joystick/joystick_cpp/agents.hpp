#ifndef AGENTS_H
#define AGENTS_H

#include <string>
#include <vector>
// external json parser (https://github.com/nlohmann/json)
#include "json.hpp"

using namespace std;
using json = nlohmann::json; // for convenience

/* PEDESTRIAN CLASS */

class AgentState
{
public:
    AgentState() {}
    AgentState(string &n, vector<float> &current_pos3, float r)
    {
        name = n;
        current_config = current_pos3;
        radius = r;
    }
    string get_name() const { return name; }
    float get_radius() const { return radius; }
    vector<float> get_current_config() const { return current_config; }
    // construct an AgentState out of its serialized json form
    static AgentState construct_from_json(const json &a)
    {
        string name = a["name"];
        vector<float> current_pos3 = a["current_config"];
        float radius = a["radius"];
        return AgentState(name, current_pos3, radius);
    }
    // construct a hash map out of a serialized json dictionary
    static unordered_map<string, AgentState> construct_from_dict(const json &d)
    {
        unordered_map<string, AgentState> pedestrians;
        // traverses the dictionary and creates AgentState instances
        for (auto &ped : d.items())
        {
            string name = ped.key(); // indexed by name (string)
            auto new_agent = AgentState::construct_from_json(ped.value());
            pedestrians.insert({name, new_agent});
        }
        return pedestrians;
    }

private:
    string name;
    vector<float> current_config;
    float radius;
};

#endif
