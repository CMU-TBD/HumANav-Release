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

    def __init__(self, params, renderer = None):
        self.params = params.simulator.parse_params(params)
        self.obstacle_map = self._init_obstacle_map(renderer)
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

        dt = p.planner_params.control_pipeline_params.system_dynamics_params.dt

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
        """ A function that simulates an entire episode. The agent starts at self.start_config, repeatedly
        calling _iterate to generate subtrajectories. Generates a vehicle_trajectory for the episode, calculates its
        objective value, and sets the episode_type (timeout, collision, success)"""
        print(print_colors()["blue"], "Running simulation on", len(self.agents), "agents", print_colors()["reset"])
        i = 0
        while self.exists_running_agent():
            for a in self.agents:
                if(i == 0 and self.params.verbose_printing):
                    print("start: ", a.start_config.position_nk2().numpy())
                    print("goal: ", a.goal_config.position_nk2().numpy())
                # vehicle_data = a.planner.empty_data_dict()
                if(not a.end_episode):
                    if(self.params.verbose_printing):
                        print(a.current_config.position_nk2().numpy())
                    trajectory_segment, next_config, trajectory_data, commanded_actions_1kf = self._iterate(a)
                    # Append to Vehicle Data
                    for key in a.vehicle_data.keys():
                        a.vehicle_data[key].append(trajectory_data[key])
                    a.vehicle_trajectory.append_along_time_axis(trajectory_segment)
                    a.commanded_actions_nkf.append(commanded_actions_1kf)
                    # update config
                    a.current_config = next_config
                    # overwrites vehicle data with last instance before termination
                    # vehicle_data_last = copy.copy(vehicle_data) #making a hardcopy
                    a._enforce_episode_termination_conditions(self.params, self.obstacle_map)
            i = i + 1
        print(" Took",i,"iterations")
        for a in self.agents:
            a.vehicle_trajectory = a.episode_data['vehicle_trajectory']
            a.vehicle_data = a.episode_data['vehicle_data']
            a.vehicle_data_last_step = a.episode_data['vehicle_data_last_step']
            a.last_step_data_valid = a.episode_data['last_step_data_valid']
            a.episode_type = a.episode_data['episode_type']
            a.valid_episode = a.episode_data['valid_episode']
            a.commanded_actions_1kf = a.episode_data['commanded_actions_1kf']
            a.obj_val = a._compute_objective_value(self.params)

    def _iterate(self, agent):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data"""
        agent.planner_data = agent.planner.optimize(agent.current_config, agent.goal_config)
        trajectory_segment, trajectory_data, commanded_actions_nkf = self._process_planner_data(agent, agent.planner_data)
        next_config = SystemConfig.init_config_from_trajectory_time_index(trajectory_segment, t=-1)
        return trajectory_segment, next_config, trajectory_data, commanded_actions_nkf

    def _process_planner_data(self, agent, planner_data):
        """
        Process the planners current plan. This could mean applying
        open loop control or LQR feedback control on a system.
        """
        start_config = agent.current_config
        # The 'plan' is open loop control
        if 'trajectory' not in planner_data.keys():
            trajectory, commanded_actions_nkf = self.apply_control_open_loop(start_config,
                                                                             planner_data['optimal_control_nk2'],
                                                                             T=self.params.control_horizon-1,
                                                                             sim_mode=self.system_dynamics.simulation_params.simulation_mode)
        # The 'plan' is LQR feedback control
        else:
            # If we are using ideal system dynamics the planned trajectory
            # is already dynamically feasible. Clip it to the control horizon
            if self.system_dynamics.simulation_params.simulation_mode == 'ideal':
                trajectory = Trajectory.new_traj_clip_along_time_axis(planner_data['trajectory'],
                                                                      self.params.control_horizon,
                                                                      repeat_second_to_last_speed=True)
                _, commanded_actions_nkf = self.system_dynamics.parse_trajectory(trajectory)
            elif self.system_dynamics.simulation_params.simulation_mode == 'realistic':
                trajectory, commanded_actions_nkf = self.apply_control_closed_loop(start_config,
                                                                                   planner_data['spline_trajectory'],
                                                                                   planner_data['k_nkf1'],
                                                                                   planner_data['K_nkfd'],
                                                                                   T=self.params.control_horizon-1,
                                                                                   sim_mode='realistic')
            else:
                assert(False)

        agent.planner.clip_data_along_time_axis(planner_data, self.params.control_horizon)
        return trajectory, planner_data, commanded_actions_nkf

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config or
        pos_nk3.
        """
        return [None]*config.n

    def get_observation_from_data_dict_and_model(self, data_dict, model):
        """
        Returns the robot's observation from the data inside data_dict,
        using parameters specified by the model.
        """
        raise NotImplementedError

    def _reset_obstacle_map(self, rng):
        raise NotImplementedError

    def _init_obstacle_map(self, renderer = None):
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
