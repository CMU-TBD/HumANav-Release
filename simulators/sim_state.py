import tensorflow as tf
import numpy as np
import copy
import json
from simulators.agent import Agent
from humans.human import Human

""" These are smaller "wrapper" classes that are visible by other
agents/humans and saved during state deepcopies
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

    def to_json(self):
        name_json = SimState.to_json_type(copy.deepcopy(self.name))
        start_json = SimState.to_json_type(
            self.get_start_config().to_numpy_repr())
        goal_json = SimState.to_json_type(
            self.get_start_config().to_numpy_repr())
        current_json = SimState.to_json_type(
            self.get_start_config().to_numpy_repr())
        trajectory_json = SimState.to_json_type(
            self.get_trajectory().to_numpy_repr())
        collided_json = SimState.to_json_type(
            copy.deepcopy(self.collided))
        end_acting_json = SimState.to_json_type(
            copy.deepcopy(self.end_acting))
        radius_json = SimState.to_json_type(
            copy.deepcopy(self.radius))
        json_type = ""
        json_type += json.dumps(name_json, indent=4)
        json_type += json.dumps(start_json, indent=4)
        json_type += json.dumps(goal_json, indent=4)
        json_type += json.dumps(current_json, indent=4)
        json_type += json.dumps(trajectory_json, indent=4)
        json_type += json.dumps(collided_json, indent=4)
        json_type += json.dumps(end_acting_json, indent=4)
        json_type += json.dumps(radius_json, indent=4)
        return json_type


class HumanState(AgentState):
    def __init__(self, human, deepcpy=False):
        self.appearance = human.get_appearance()
        self.name = human.get_name()
        # Initialize the agent state class
        super().__init__(human, deepcpy=deepcpy)

    def get_appearance(self):
        return self.appearance

    def to_json(self):
        appearance_json = SimState.to_json_type(self.appearance)
        agent_json = super().to_json()
        return json.dumps(appearance_json, indent=4) + agent_json


class SimState():
    def __init__(self, environment, agents, prerecs, robots, sim_time, wall_time):
        self.environment = environment
        self.agents = agents
        self.prerecs = prerecs
        self.robots = robots
        self.sim_t = sim_time
        self.wall_t = wall_time

    def convert_to_json(self):
        environment_json = SimState.to_json_dict(
            copy.deepcopy(self.environment))
        agents_json = SimState.to_json_dict(self.agents)
        prerecs_json = SimState.to_json_dict(self.prerecs)
        robots_json = SimState.to_json_dict(self.robots)
        sim_t_json = SimState.to_json_type(self.sim_t)
        wall_t_json = SimState.to_json_type(self.wall_t)
        json_type = ""
        json_type += json.dumps(environment_json, indent=4)
        json_type += json.dumps(agents_json, indent=4)
        json_type += json.dumps(robots_json, indent=4)
        json_type += json.dumps(prerecs_json, indent=4)
        json_type += json.dumps(sim_t_json, indent=4)
        json_type += json.dumps(wall_t_json, indent=4)
        return json_type

    def get_environment(self):
        return self.environment

    def get_map(self):
        return self.environment["traversibles"][0]

    def get_agents(self):
        return self.agents

    def get_prerecs(self):
        return self.prerecs

    def get_robots(self):
        return self.robots

    def get_sim_t(self):
        return self.sim_t

    def get_wall_t(self):
        return self.wall_t

    @ staticmethod
    def to_json_type(elem):
        """ Converts an element to a json serializable type. """
        if isinstance(elem, np.int64) or isinstance(elem, np.int32):
            return int(elem)
        if isinstance(elem, tf.Tensor):
            return elem.numpy().tolist()
        if isinstance(elem, np.ndarray):
            return elem.tolist()
        if isinstance(elem, dict):
            # recursive for dictionaries within dictionaries
            return SimState.to_json_dict(elem)
        if isinstance(elem, AgentState):
            return elem.to_json()
        if type(elem) is type:  # elem is a class
            return str(elem)
        else:
            return str(elem)

    @ staticmethod
    def to_json_dict(param_dict):
        """ Converts params_dict to a json serializable dict."""
        for key in param_dict.keys():
            param_dict[key] = SimState.to_json_type(param_dict[key])
        return param_dict
