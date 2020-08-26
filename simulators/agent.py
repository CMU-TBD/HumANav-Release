import numpy as np
import sys
import os
import copy
import time

from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance

from trajectory.trajectory import SystemConfig, Trajectory
from utils.fmm_map import FmmMap
from utils.utils import *
from planners.sampling_planner import SamplingPlanner
from simulators.agent_helper import AgentHelper
from params.central_params import create_agent_params


class Agent(AgentHelper):
    def __init__(self, start, goal, name=None, with_init=True):
        if name is None:
            self.name = generate_name(20)
        else:
            self.name = name
        self.start_config = start
        self.goal_config = goal
        # upon initialization, the current config of the agent is start
        self.current_config = copy.deepcopy(start)
        # path planning and acting fields
        self.end_episode = False
        self.end_acting = False
        # for collisions with other gen_agents
        self.has_collided = False
        if(with_init):
            self.init()
        # cosmetic items (for drawing the trajectories)
        possible_colors = ['b', 'g', 'r', 'c', 'm', 'y']  # not white or black
        self.color = random.choice(possible_colors)

    def init(self):
        self.planned_next_config = copy.deepcopy(self.current_config)
        # Dynamics and movement attributes
        self.fmm_map = None
        self.path_step = 0
        self.termination_cause = None
        # NOTE: JSON serialization is done within sim_state.py
        self.velocities = {}
        self.accelerations = {}
        self.sim_states = []

    # Getters for the Agent class
    def get_name(self):
        return self.name

    def get_config(self, config, deepcpy):
        if(deepcpy):
            return SystemConfig.copy(config)
        return config

    def get_start_config(self, deepcpy=False):
        return self.get_config(self.start_config, deepcpy)

    def set_start_config(self, start):
        self.start_config = start

    def get_goal_config(self, deepcpy=False):
        return self.get_config(self.goal_config, deepcpy)

    def set_goal_config(self, goal):
        self.goal_config = goal

    def get_current_config(self, deepcpy=False):
        return self.get_config(self.current_config, deepcpy)

    def set_current_config(self, current):
        self.current_config = current

    def get_trajectory(self, deepcpy=False):
        if(deepcpy):
            return Trajectory.copy(self.vehicle_trajectory, check_dimens=False)
        return self.vehicle_trajectory

    def get_collided(self):
        return self.has_collided

    def get_radius(self):
        return self.params.radius

    def get_color(self):
        return self.color

    def simulation_init(self, sim_map, with_planner=True):
        """ Initializes important fields for the CentralSimulator"""
        if(not hasattr(self, 'params')):
            self.params = create_agent_params(with_planner=with_planner)
        self.obstacle_map = sim_map
        self.obj_fn = Agent._init_obj_fn(self)
        # Initialize Fast-Marching-Method map for agent's pathfinding
        self.fmm_map = Agent._init_fmm_map(self)
        Agent._update_fmm_map(self)
        # Initialize system dynamics and planner fields
        if(with_planner):
            self.planner = Agent._init_planner(self)
            self.vehicle_data = self.planner.empty_data_dict()
        else:
            self.planner = None
            self.vehicle_data = None
        self.system_dynamics = Agent._init_system_dynamics(self)
        self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
        # the point in the trajectory where the agent collided
        self.collision_point_k = np.inf

    def update_final(self):
        self.vehicle_trajectory = self.episode_data['vehicle_trajectory']
        self.vehicle_data = self.episode_data['vehicle_data']
        self.vehicle_data_last_step = self.episode_data['vehicle_data_last_step']
        self.last_step_data_valid = self.episode_data['last_step_data_valid']
        self.episode_type = self.episode_data['episode_type']
        self.valid_episode = self.episode_data['valid_episode']
        self.commanded_actions_1kf = self.episode_data['commanded_actions_1kf']
        self.obj_val = self._compute_objective_value()

    def update(self, t, t_step, sim_state=None):
        """ Run the agent.plan() and agent.act() functions to generate a path and follow it """
        self.sim_states.append(sim_state)
        if(self.params.verbose_printing):
            print("start: ", self.get_start_config().position_nk2())
            print("goal: ", self.get_goal_config().position_nk2())

        # self.velocities[get_sim_t(sim_state)] = compute_all_velocities(self.sim_states)
        # self.accelerations[get_sim_t(sim_state)] = compute_all_accelerations(self.sim_states)

        # Generate the next trajectory segment, update next config, update actions/data
        self.plan()
        action_dt = int(np.floor(t_step / self.params.dt))
        self.act(action_dt, world_state=sim_state)

    def plan(self):
        """
        Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data
        """
        if not hasattr(self, 'commanded_actions_nkf'):
            # initialize commanded actions
            self.commanded_actions_nkf = []

        if not hasattr(self, 'planner'):
            # create planner if none exists
            self.planner = Agent._init_planner(self)

        if not self.end_episode and not self.end_acting:
            if self.params.verbose_printing:
                print("planned next:",
                      self.planned_next_config.position_nk2())

            self.planner_data = self.planner.optimize(
                self.planned_next_config, self.goal_config)
            traj_segment, trajectory_data, commands_1kf = self._process_planner_data()

            self.planned_next_config = \
                SystemConfig.init_config_from_trajectory_time_index(
                    traj_segment, t=-1)

            # Append to Vehicle Data
            for key in self.vehicle_data.keys():
                self.vehicle_data[key].append(trajectory_data[key])
            self.vehicle_trajectory.append_along_time_axis(
                traj_segment,
                track_trajectory_acceleration=self.params.planner_params.track_accel)
            self.commanded_actions_nkf.append(commands_1kf)
            self._enforce_episode_termination_conditions()

            if self.end_episode or self.end_acting:
                if self.params.verbose:
                    print("terminated plan for agent", self.get_name(),
                          "k=", self.vehicle_trajectory.k,
                          "total time=", self.vehicle_trajectory.k * self.vehicle_trajectory.dt)

    def _collision_in_group(self, own_pos: np.array, group: list):
        for a in group:
            othr_pos = a.get_current_config().to_3D_numpy()
            if(a.get_name() is not self.get_name() and
                    euclidean_dist2(own_pos, othr_pos) < self.get_radius() + a.get_radius()):
                # instantly collide and stop updating
                self.has_collided = True
                self.collision_point_k = self.vehicle_trajectory.k  # this instant
                self.end_acting = True

    def check_collisions(self, world_state, include_agents=True, include_prerecs=True, include_robots=True):
        if world_state is not None:
            own_pos = self.get_current_config().to_3D_numpy()
            if include_agents and self._collision_in_group(own_pos, world_state.get_agents().values()):
                return True
            if include_prerecs and self._collision_in_group(own_pos, world_state.get_prerecs().values()):
                return True
            if include_robots and self._collision_in_group(own_pos, world_state.get_robots().values()):
                return True
        return False

    def act(self, action_dt, instant_complete=False, world_state=None):
        """ A utility method to initialize a config object
        from a particular timestep of a given trajectory object"""
        if not self.end_acting:
            if instant_complete:
                # Complete the entire update of the current_config in one go
                self.current_config = \
                    SystemConfig.init_config_from_trajectory_time_index(
                        self.vehicle_trajectory, t=-1)
                # Instantly finished trajectory
                self.end_acting = True
            else:
                # Update through the path traversal incrementally

                # first check for collisions with any other gen_agents
                self.check_collisions(world_state)

                # then update the current config
                self.current_config = \
                    SystemConfig.init_config_from_trajectory_time_index(
                        self.vehicle_trajectory, t=self.path_step)

                # updating "next step" for agent path after traversing it
                self.path_step += action_dt
                if self.path_step >= self.vehicle_trajectory.k:
                    self.end_acting = True

                # considers a full on collision once the agent has passed its "collision point"
                if self.path_step >= self.collision_point_k:
                    self.has_collided = True
                    self.end_acting = True

                if self.end_acting or self.has_collided:
                    if self.params.verbose:
                        print("terminated act for agent", self.get_name())
                    # save memory by deleting control pipeline (very memory intensive)
                    del self.planner

        # NOTE: can use the following if want to update further tracked variables, but sometimes
        # this is buggy when the action is not fully completed, thus this should be a TODO: fix
        # else:
        #     self.update_final()

    # TODO: put most of the below functions in an agent_helper.py class
    # TODO: this should probably be static too
    def _process_planner_data(self):
        """
        Process the planners current plan. This could mean applying
        open loop control or LQR feedback control on a system.
        """
        start_config = self.current_config
        # The 'plan' is open loop control
        if 'trajectory' not in self.planner_data.keys():
            trajectory, commanded_actions_nkf = \
                Agent.apply_control_open_loop(self, start_config,
                                              self.planner_data['optimal_control_nk2'],
                                              T=self.params.control_horizon - 1,
                                              sim_mode=self.system_dynamics.simulation_params.simulation_mode)
        # The 'plan' is LQR feedback control
        else:
            # If we are using ideal system dynamics the planned trajectory
            # is already dynamically feasible. Clip it to the control horizon
            if self.system_dynamics.simulation_params.simulation_mode == 'ideal':
                trajectory = \
                    Trajectory.new_traj_clip_along_time_axis(self.planner_data['trajectory'],
                                                             self.params.control_horizon,
                                                             repeat_second_to_last_speed=True)
                _, commanded_actions_nkf = self.system_dynamics.parse_trajectory(
                    trajectory)
            elif self.system_dynamics.simulation_params.simulation_mode == 'realistic':
                trajectory, commanded_actions_nkf = \
                    self.apply_control_closed_loop(start_config,
                                                   self.planner_data['spline_trajectory'],
                                                   self.planner_data['k_nkf1'],
                                                   self.planner_data['K_nkfd'],
                                                   T=self.params.control_horizon - 1,
                                                   sim_mode='realistic')
            else:
                assert False

        self.planner.clip_data_along_time_axis(
            self.planner_data, self.params.control_horizon)
        return trajectory, self.planner_data, commanded_actions_nkf

    def _compute_objective_value(self):
        p = self.params.objective_fn_params
        if p.obj_type == 'valid_mean':
            self.vehicle_trajectory.update_valid_mask_nk()
        else:
            assert (p.obj_type in ['valid_mean', 'mean'])
        obj_val = np.squeeze(
            self.obj_fn.evaluate_function(self.vehicle_trajectory))
        return obj_val

    @staticmethod
    def _init_obj_fn(self, params=None):
        """
        Initialize the objective function given sim params
        """
        if params is None:
            params = self.params
        obstacle_map = self.obstacle_map
        obj_fn = ObjectiveFunction(params.objective_fn_params)
        if not params.avoid_obstacle_objective.empty():
            obj_fn.add_objective(
                ObstacleAvoidance(params=params.avoid_obstacle_objective,
                                  obstacle_map=obstacle_map))
        if not params.goal_distance_objective.empty():
            obj_fn.add_objective(
                GoalDistance(params=params.goal_distance_objective,
                             fmm_map=obstacle_map.fmm_map))
        if not params.goal_angle_objective.empty():
            obj_fn.add_objective(
                AngleDistance(params=params.goal_angle_objective,
                              fmm_map=obstacle_map.fmm_map))
        return obj_fn

    @staticmethod
    def _init_planner(self, params=None):
        if(params is None):
            params = self.params
        obj_fn = self.obj_fn
        return params.planner_params.planner(obj_fn=obj_fn,
                                             params=params.planner_params)

    @staticmethod
    def _init_fmm_map(self, goal_pos_n2=None, params=None):
        if(params is None):
            params = self.params
        obstacle_map = self.obstacle_map
        obstacle_occupancy_grid = obstacle_map.create_occupancy_grid_for_map()

        if goal_pos_n2 is None:
            goal_pos_n2 = self.goal_config.position_nk2()[0]

        return FmmMap.create_fmm_map_based_on_goal_position(
            goal_positions_n2=goal_pos_n2,
            map_size_2=np.array(self.obstacle_map.get_map_size_2()),
            dx=self.obstacle_map.get_dx(),
            map_origin_2=self.obstacle_map.get_map_origin_2(),
            mask_grid_mn=obstacle_occupancy_grid)

    @staticmethod
    def _init_system_dynamics(self, params=None):
        """
        If there is a control pipeline (i.e. model based method)
        return its system_dynamics. Else create a new system_dynamics
        instance.
        """
        if params is None:
            params = self.params
        try:
            planner = self.planner
            return planner.control_pipeline.system_dynamics
        except AttributeError:
            p = params.system_dynamics_params
            return p.system(dt=p.dt, params=p)

    @staticmethod
    def _update_obj_fn(self):
        """ 
        Update the objective function to use a new obstacle_map and fmm map
        """
        for objective in self.obj_fn.objectives:
            if isinstance(objective, ObstacleAvoidance):
                objective.obstacle_map = self.obstacle_map
            elif isinstance(objective, GoalDistance):
                objective.fmm_map = self.fmm_map
            elif isinstance(objective, AngleDistance):
                objective.fmm_map = self.fmm_map
            else:
                assert False

    @staticmethod
    def _update_fmm_map(self):
        """
        For SBPD the obstacle map does not change,
        so just update the goal position.
        """
        goal_pos_n2 = self.goal_config.position_nk2()[:, 0]
        assert(hasattr(self, 'fmm_map'))
        if self.fmm_map is not None:
            self.fmm_map.change_goal(goal_pos_n2)
        else:
            self.fmm_map = Agent._init_fmm_map(self)
        Agent._update_obj_fn(self)
