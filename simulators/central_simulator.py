import tensorflow as tf
import numpy as np
import copy
from trajectory.trajectory import SystemConfig, Trajectory
from simulators.simulator_helper import SimulatorHelper
from simulators.agent import Agent
from utils.fmm_map import FmmMap
from utils.utils import print_colors
import matplotlib

class CentralSimulator(SimulatorHelper):

    def __init__(self, params, renderer=None):
        self.params = params.simulator.parse_params(params)
        self.obstacle_map = self._init_obstacle_map(renderer)
        # theoretially all the agents can have their own system dynamics as well
        self.system_dynamics = self._init_system_dynamics()
        self.agents = []

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        # Parse the dependencies
        p.planner_params.planner.parse_params(p.planner_params)
        p.obstacle_map_params.obstacle_map.parse_params(p.obstacle_map_params)
        # Time discretization step
        dt = p.planner_params.control_pipeline_params.system_dynamics_params.dt
        # Updating horizons
        p.episode_horizon = int(np.ceil(p.episode_horizon_s / dt))
        p.control_horizon = int(np.ceil(p.control_horizon_s / dt))
        p.dt = dt

        return p

    def add_agent(self, agent):
        # have each agent potentially have their own planners
        agent.obj_fn = agent._init_obj_fn(self.params, self.obstacle_map)
        agent.planner = agent._init_planner(self.params)
        agent.vehicle_data = agent.planner.empty_data_dict()
        agent.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
        agent._update_fmm_map(self.params, self.obstacle_map)
        self.agents.append(agent)

    def exists_running_agent(self):
        for x in self.agents:
            # if there is even just a single agent running
            if (x.end_episode == False):
                return True
        return False

    def simulate(self):
        """ A function that simulates an entire episode. The agent starts
        at self.start_config, repeatedly calling _iterate to generate 
        subtrajectories. Generates a vehicle_trajectory for the episode, 
        calculates its objective value, and sets the episode_type 
        (timeout, collision, success) """
        print(print_colors()["blue"], "Running simulation on", len(
            self.agents), "agents", print_colors()["reset"])
        i = 0
        while self.exists_running_agent():
            for a in self.agents:
                if(i == 0 and self.params.verbose_printing):
                    print("start: ", a.start_config.position_nk2().numpy())
                    print("goal: ", a.goal_config.position_nk2().numpy())
                a.update(self.params, self.system_dynamics, self.obstacle_map)
            i = i + 1
        print(" Took", i, "iterations")
        for a in self.agents:
            a.update_final(self.params)

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config
        or pos_nk3.
        """
        return self.obstacle_map.get_observation(config=config, pos_n3=pos_n3, **kwargs)

    def get_observation_from_data_dict_and_model(self, data_dict, model):
        """
        Returns the robot's observation from the data inside data_dict,
        using parameters specified by the model.
        """
        if hasattr(model, 'occupancy_grid_positions_ego_1mk12'):
            kwargs = {'occupancy_grid_positions_ego_1mk12':
                      model.occupancy_grid_positions_ego_1mk12}
        else:
            kwargs = {}

        img_nmkd = self.get_observation(pos_n3=data_dict['vehicle_state_nk3'][:, 0],
                                        **kwargs)
        return img_nmkd

    def _reset_obstacle_map(self, rng):
        """
        For SBPD the obstacle map does not change
        between episodes.
        """
        return False

    def _init_obstacle_map(self, renderer=None):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p, renderer)

    def _init_system_dynamics(self):
        """
        If there is a control pipeline (i.e. model based method)
        return its system_dynamics. Else create a new system_dynamics
        instance.
        """
        try:
            return self.planner.control_pipeline.system_dynamics
        except AttributeError:
            p = self.params.planner_params.control_pipeline_params.system_dynamics_params
            return p.system(dt=p.dt, params=p)

    def _render_obstacle_map(self, ax, zoom=0):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(
                ax,
                margin0=p.avoid_obstacle_objective.obstacle_margin0,
                margin1=p.avoid_obstacle_objective.obstacle_margin1,
                zoom=zoom)