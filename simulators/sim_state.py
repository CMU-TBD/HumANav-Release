import tensorflow as tf
import numpy as np
from copy import deepcopy
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
        name_json = SimState.to_json_type(deepcopy(self.name))
        # NOTE: the configs are just being serialized with their 3D positions
        # start_json = SimState.to_json_type(
        #     self.get_start_config().to_3D_numpy())
        # goal_json = SimState.to_json_type(
        #     self.get_goal_config().to_3D_numpy())
        current_json = SimState.to_json_type(
            deepcopy(self.get_current_config().to_3D_numpy()))
        # SimState.to_json_type( self.get_trajectory().to_numpy_repr())
        # trajectory_json = "None"
        collided_json = deepcopy(self.collided)
        end_acting_json = deepcopy(self.end_acting)
        radius_json = deepcopy(self.radius)
        json_dict = {}
        json_dict['name'] = name_json
        # NOTE: goal and start can perhaps be optimized to be only sent once
        # since they don't change over the course of the simulation.
        # json_dict['start_config'] = start_json
        # json_dict['goal_config'] = goal_json
        json_dict['current_config'] = current_json
        # json_dict['trajectory'] = trajectory_json
        json_dict['collided'] = collided_json
        json_dict['end_acting'] = end_acting_json
        json_dict['radius'] = radius_json
        # returns array (python list) to be json'd in_simstate
        return json_dict


class HumanState(AgentState):
    def __init__(self, human, deepcpy=False):
        self.appearance = human.get_appearance()
        self.name = human.get_name()
        # Initialize the agent state class
        super().__init__(human, deepcpy=deepcpy)

    def get_appearance(self):
        return self.appearance


class SimState():
    def __init__(self, environment, agents, prerecs, robots, sim_time, wall_time):
        self.environment = environment
        self.agents = agents
        self.prerecs = prerecs
        self.robots = robots
        self.sim_t = sim_time
        self.wall_t = wall_time

    def to_json(self, robot_on=True, include_map=False):
        json_dict = {}
        json_dict['robot_on'] = deepcopy(robot_on)  # true or false
        if(robot_on):  # only send the world if the robot is ON
            if(include_map):
                environment_json = SimState.to_json_dict(
                    deepcopy(self.environment))
            else:
                environment_json = {}  # empty dictionary
            # serialize all other fields
            agents_json = deepcopy(
                SimState.to_json_dict(deepcopy(self.agents)))
            prerecs_json = deepcopy(
                SimState.to_json_dict(deepcopy(self.prerecs)))
            robots_json = deepcopy(
                SimState.to_json_dict(deepcopy(self.robots)))
            sim_t_json = deepcopy(self.sim_t)
            wall_t_json = deepcopy(self.wall_t)
            # append them to the json dictionary
            json_dict['environment'] = environment_json
            json_dict['agents'] = agents_json
            json_dict['prerecs'] = prerecs_json
            json_dict['robots'] = robots_json
            json_dict['sim_t'] = sim_t_json
            json_dict['wall_t'] = wall_t_json
        return json.dumps(json_dict, indent=1)

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
