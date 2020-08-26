import numpy as np
from copy import deepcopy
import json
from simulators.agent import Agent
from humans.human import Human
from utils.utils import *

""" These are smaller "wrapper" classes that are visible by other
gen_agents/humans and saved during state deepcopies
NOTE: they are all READ-ONLY (only getters)
"""


class AgentState():
    def __init__(self, a, deepcpy=False):
        self.name = a.get_name()
        self.start_config = a.get_start_config(deepcpy=deepcpy)
        self.goal_config = a.get_goal_config(deepcpy=deepcpy)
        self.current_config = a.get_current_config(deepcpy=deepcpy)
        self.vehicle_trajectory = a.get_trajectory(deepcpy=deepcpy)
        self.collided = a.get_collided()
        self.end_acting = a.end_acting
        self.radius = a.get_radius()
        self.color = a.get_color()

    def get_name(self):
        return self.name

    def get_current_config(self):
        return self.current_config

    def get_start_config(self):
        return self.start_config

    def get_goal_config(self):
        return self.goal_config

    def get_trajectory(self):
        return self.vehicle_trajectory

    def get_collided(self):
        return self.collided

    def get_radius(self):
        return self.radius

    def get_color(self):
        return self.color

    def get_pos3(self):
        return self.get_current_config().to_3D_numpy()

    def to_json(self, include_start_goal=False):
        name_json = SimState.to_json_type(deepcopy(self.name))
        # NOTE: the configs are just being serialized with their 3D positions
        if(include_start_goal):
            start_json = SimState.to_json_type(
                self.get_start_config().to_3D_numpy())
            goal_json = SimState.to_json_type(
                self.get_goal_config().to_3D_numpy())
        current_json = SimState.to_json_type(
            deepcopy(self.get_current_config().to_3D_numpy()))
        # SimState.to_json_type( self.get_trajectory().to_numpy_repr())
        # trajectory_json = "None"
        radius_json = deepcopy(self.radius)
        json_dict = {}
        json_dict['name'] = name_json
        # NOTE: the start and goal (of the robot) are only sent when the environment is sent
        if(include_start_goal):
            json_dict['start_config'] = start_json
            json_dict['goal_config'] = goal_json
        json_dict['current_config'] = current_json
        # json_dict['trajectory'] = trajectory_json
        json_dict['radius'] = radius_json
        # returns array (python list) to be json'd in_simstate
        return json_dict


class HumanState(AgentState):
    def __init__(self, human, deepcpy=False):
        self.appearance = human.get_appearance()
        # Initialize the agent state class
        super().__init__(human, deepcpy=deepcpy)

    def get_appearance(self):
        return self.appearance


class SimState():
    def __init__(self, environment, gen_agents, prerecs, robots, sim_t, wall_t, delta_t):
        self.environment = environment
        self.gen_agents = gen_agents
        self.prerecs = prerecs
        self.robots = robots
        self.sim_t = sim_t
        self.wall_t = wall_t
        self.delta_t = delta_t

    def to_json(self, robot_on=True, include_map=False):
        json_dict = {}
        json_dict['robot_on'] = deepcopy(robot_on)  # true or false
        if robot_on:  # only send the world if the robot is ON
            if include_map:
                environment_json = SimState.to_json_dict(
                    deepcopy(self.environment))
                json_dict['delta_t'] = deepcopy(self.delta_t)
            else:
                environment_json = {}  # empty dictionary
            # serialize all other fields
            agents_json = SimState.to_json_dict(deepcopy(self.gen_agents))
            prerecs_json = SimState.to_json_dict(deepcopy(self.prerecs))
            robots_json = SimState.to_json_dict(
                deepcopy(self.robots), include_start_goal=include_map)
            sim_t_json = deepcopy(self.sim_t)
            # append them to the json dictionary
            json_dict['environment'] = environment_json
            json_dict['gen_agents'] = agents_json
            json_dict['prerecs'] = prerecs_json
            json_dict['robots'] = robots_json
            json_dict['sim_t'] = sim_t_json
        return json.dumps(json_dict, indent=1)

    def get_environment(self):
        return self.environment

    def get_map(self):
        return self.environment["traversibles"][0]

    # TODO rename to get_genagents or get_gen_agents
    def get_agents(self):
        return self.gen_agents

    def get_gen_agents(self):
        return self.gen_agents

    def get_prerecs(self):
        return self.prerecs

    def get_robots(self):
        return self.robots

    def get_sim_t(self):
        return self.sim_t

    def get_wall_t(self):
        return self.wall_t

    def get_delta_t(self):
        return self.delta_t

    def get_all_agents(self):
        all_agents = {}
        all_agents.update(get_agent_type(self, "gen_agents"))
        all_agents.update(get_agent_type(self, "prerecs"))
        all_agents.update(get_agent_type(self, "robots"))
        return all_agents

    @ staticmethod
    def to_json_type(elem, include_start_goal=False):
        """ Converts an element to a json serializable type. """
        if isinstance(elem, np.int64) or isinstance(elem, np.int32):
            return int(elem)
        if isinstance(elem, np.ndarray):
            return elem.tolist()
        if isinstance(elem, dict):
            # recursive for dictionaries within dictionaries
            return SimState.to_json_dict(elem, include_start_goal=include_start_goal)
        if isinstance(elem, AgentState):
            return elem.to_json(include_start_goal=include_start_goal)
        if type(elem) is type:  # elem is a class
            return str(elem)
        else:
            return str(elem)

    @ staticmethod
    def to_json_dict(param_dict, include_start_goal=False):
        """ Converts params_dict to a json serializable dict."""
        for key in param_dict.keys():
            param_dict[key] = SimState.to_json_type(
                param_dict[key], include_start_goal=include_start_goal)
        return param_dict


"""BEGIN SimState utils"""


def get_agent_type(sim_state, agent_type: str):
    if callable(getattr(sim_state, 'get_' + agent_type, None)):
        getter_agent_type = getattr(sim_state, 'get_' + agent_type, None)
        return getter_agent_type()
    elif agent_type in sim_state.keys():
        return sim_state[agent_type]
    else:
        return {}  # empty dict


def get_sim_t(sim_state):
    if callable(getattr(sim_state, 'get_sim_t', None)):
        return sim_state.get_sim_t()
    return sim_state["sim_t"]


def compute_delta_t(sim_states: list):
    # need at least one (usually the first) to have a delta_t
    for i in range(len(sim_states)):
        if(callable(getattr(sim_states[i], 'get_delta_t', None))):
            return sim_states[i].get_delta_t()
        # optimized to only have delta_t on the FIRST SimState
        return sim_states[i]["delta_t"]
    # or computing it manually with two sim_states:
    # if(len(sim_states) <= 1):
    #     print("%sNeed at least two states to compute delta_t%s" %
    #           (color_red, color_reset))
    # else:
    #     delta_t = get_sim_t(sim_states[1]) - get_sim_t(sim_states[0])
    #     return delta_t


def get_pos3(agent):
    if callable(getattr(agent, "get_current_config", None)):
        return agent.get_current_config().to_3D_numpy()
    return agent["current_config"]


# TODO: check how the delta_t is treated - want to use inter value deltas rather than const delta
def compute_next_vel(sim_state_prev, sim_state_now, agent_name: str, delta_t: float):
    old_agent = sim_state_prev.get_all_agents()[agent_name]
    old_pos = get_pos3(old_agent)
    new_agent = sim_state_prev.get_all_agents()[agent_name]
    new_pos = get_pos3(new_agent)
    # calculate distance over time
    # TODO: add sign to distance (displacement) for velocity?
    return euclidean_dist2(old_pos, new_pos) / delta_t


def compute_agent_state_velocity(sim_states: list, agent_name: str):
    if(len(sim_states) > 1):  # need at least two to compute differences in positions
        if(agent_name in get_all_agents(sim_states[0]).keys()):
            agent_velocities = []
            delta_t = compute_delta_t(sim_states)
            for i, s in enumerate(sim_states):
                if(i > 0):
                    speed = compute_next_vel(
                        sim_states[i - 1], sim_states[i], agent_name, delta_t)
                    agent_velocities.append(speed)
                else:
                    agent_velocities.append(0)
            return agent_velocities
        else:
            print("%sAgent" % color_red, agent_name,
                  "is not in the SimStates%s" % color_reset)


def compute_agent_state_acceleration(sim_states: list, agent_name: str, velocities: list = None):
    if(len(sim_states) > 1):  # need at least two to compute differences in velocities
        # optionally compute velocities as well
        if(velocities is None):
            velocities = compute_agent_state_velocity(
                sim_states, agent_name)
        delta_t = compute_delta_t(sim_states)
        if(agent_name in get_all_agents(sim_states[0]).keys()):
            agent_accels = []
            for i, this_vel in enumerate(velocities):
                if(i > 0):
                    last_vel = velocities[i - 1]
                    # calculate speeds over time
                    accel = (this_vel - last_vel) / delta_t
                    agent_accels.append(accel)
                    if(i == len(sim_states) - 1):
                        # last element gets no acceleration
                        break
                        # record[j].append(0)
            return agent_accels
        else:
            print("%sAgent" % color_red, agent_name,
                  "is not in the SimStates%s" % color_reset)
    else:
        return []


def compute_all_velocities(sim_states: list):
    all_velocities = {}
    for agent_name in get_all_agents(sim_states[0]).keys():
        assert(isinstance(agent_name, str))  # keyed by name
        all_velocities[agent_name] = compute_agent_state_velocity(
            sim_states, agent_name)
    return all_velocities


def compute_all_accelerations(sim_states: list):
    all_accels = {}
    # TODO: add option of providing precomputed velocities list
    for agent_name in get_all_agents(sim_states[0]).keys():
        assert(isinstance(agent_name, str))  # keyed by name
        all_accels[agent_name] = compute_agent_state_acceleration(
            sim_states, agent_name)
    return all_accels
