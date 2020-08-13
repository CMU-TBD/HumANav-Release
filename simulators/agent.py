import tensorflow as tf
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


class Agent(object):
    def __init__(self, start, goal, name=None):
        if name is None:
            self.name = generate_name(20)
        else:
            self.name = name
        self.start_config = start
        self.goal_config = goal
        # upon initialization, the current config of the agent is start
        self.current_config = copy.deepcopy(start)
        self.planned_next_config = copy.deepcopy(self.current_config)

        self.time = 0  # tie to track progress during an update
        self.radius = 0.2  # meters (10cm radius)

        # Dynamics and movement attributes
        self.fmm_map = None
        # path planning and acting fields
        self.end_episode = False
        self.end_acting = False
        self.path_step = 0
        self.termination_cause = None
        # for collisions with other agents
        self.has_collided = False
        # cosmetic items (for drawing the trajectories)
        possible_colors = ['b','g','r','c','m','y'] # not white or black
        self.color = random.choice(possible_colors)
        # NOTE: JSON serialization is done within sim_state.py

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
        return self.radius

    def get_color(self):
        return self.color

    def simulation_init(self, sim_params, sim_map, with_planner=True):
        """ Initializes important fields for the CentralSimulator"""
        self.params = sim_params
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
        # Motion fields
        self.max_v = self.params.planner_params.control_pipeline_params.system_dynamics_params.v_bounds[
            1]
        self.max_w = self.params.planner_params.control_pipeline_params.system_dynamics_params.w_bounds[
            1]

    def update_final(self):
        self.vehicle_trajectory = self.episode_data['vehicle_trajectory']
        self.vehicle_data = self.episode_data['vehicle_data']
        self.vehicle_data_last_step = self.episode_data['vehicle_data_last_step']
        self.last_step_data_valid = self.episode_data['last_step_data_valid']
        self.episode_type = self.episode_data['episode_type']
        self.valid_episode = self.episode_data['valid_episode']
        self.commanded_actions_1kf = self.episode_data['commanded_actions_1kf']
        self.obj_val = self._compute_objective_value()

    def update_time(self, t):
        self.time = t

    def update(self, t, t_step, sim_state=None):
        """ Run the agent.plan() and agent.act() functions to generate a path and follow it """
        # with lock:
        self.update_time(t)
        if(self.params.verbose_printing):
            print("start: ", self.get_start_config().position_nk2().numpy())
            print("goal: ", self.get_goal_config().position_nk2().numpy())

        # Generate the next trajectory segment, update next config, update actions/data

        self.plan()
        # NOTE: typically the current planner will have a large difference between lin/ang speed, thus
        #       there are very few situations where both are at a high point
        action_dt = int(np.floor(t_step / self.params.dt))
        # NOTE: instant_complete discards any animations and finishes the trajectory segment instantly
        self.act(action_dt, instant_complete=False, world_state=sim_state)

    def plan(self):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data"""
        if not hasattr(self, 'commanded_actions_nkf'):
            # initialize commanded actions
            self.commanded_actions_nkf = []
        if(not hasattr(self, 'planner')):
            # create planner if none exists
            self.planner = Agent._init_planner(self)
        if(not self.end_episode and not self.end_acting):
            if(self.params.verbose_printing):
                print("planned next:",
                      self.planned_next_config.position_nk2().numpy())
            self.planner_data = self.planner.optimize(
                self.planned_next_config, self.goal_config)
            traj_segment, trajectory_data, commands_1kf = self._process_planner_data()
            self.planned_next_config = \
                SystemConfig.init_config_from_trajectory_time_index(
                    traj_segment,
                    t=-1
                )
            # Append to Vehicle Data
            for key in self.vehicle_data.keys():
                self.vehicle_data[key].append(trajectory_data[key])
            self.vehicle_trajectory.append_along_time_axis(traj_segment)
            self.commanded_actions_nkf.append(commands_1kf)
            self._enforce_episode_termination_conditions()
            if(self.end_episode or self.end_acting):
                if(self.params.verbose):
                    print("terminated plan for agent", self.get_name(),
                          "at t =", self.time, "k=", self.vehicle_trajectory.k,
                          "total time=", self.vehicle_trajectory.k * self.vehicle_trajectory.dt)

    def _collision_in_group(self, own_pos:np.array, group:list):
        for a in group:
            othr_pos = a.get_current_config().to_3D_numpy()
            if(a.get_name() is not self.get_name() and
                    euclidean_dist2(own_pos, othr_pos) < self.get_radius() + a.get_radius()):
                # instantly collide and stop updating
                self.has_collided = True
                self.collision_point_k = self.vehicle_trajectory.k  # this instant
                self.end_acting = True

    def check_collisions(self, world_state, include_agents=True, include_prerecs=True, include_robots=True):
        if(world_state is not None):
            own_pos = self.get_current_config().to_3D_numpy()
            if(include_agents and self._collision_in_group(own_pos, world_state.get_agents().values())):
                return True
            if(include_prerecs and self._collision_in_group(own_pos, world_state.get_prerecs().values())):
                return True
            if(include_robots and self._collision_in_group(own_pos, world_state.get_robots().values())):
                return True
        return False
            
    def act(self, action_dt, instant_complete=False, world_state=None):
        """ A utility method to initialize a config object
        from a particular timestep of a given trajectory object"""
        if(not self.end_acting):
            if instant_complete:
                # Complete the entire update of the current_config in one go
                self.current_config = \
                    SystemConfig.init_config_from_trajectory_time_index(
                        self.vehicle_trajectory, t=-1)
                # Instantly finished trajectory
                self.end_acting = True
            else:
                # Update through the path traversal incrementally

                # first check for collisions with any other agents
                self.check_collisions(world_state)

                # then update the current config
                self.current_config = \
                    SystemConfig.init_config_from_trajectory_time_index(
                        self.vehicle_trajectory, t=self.path_step)

                # updating "next step" for agent path after traversing it
                self.path_step += action_dt
                if(self.path_step >= self.vehicle_trajectory.k):
                    self.end_acting = True

                # considers a full on collision once the agent has passed its "collision point"
                if(self.path_step >= self.collision_point_k):
                    self.has_collided = True
                    self.end_acting = True

                if(self.end_acting or self.has_collided):
                    if(self.params.verbose):
                        print("terminated act for agent",
                              self.get_name(), "at t =", self.time)
                    # save memory by deleting control pipeline (very memory intensive)
                    del(self.planner)

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
                self.apply_control_open_loop(start_config,
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
                assert(False)

        self.planner.clip_data_along_time_axis(
            self.planner_data, self.params.control_horizon)
        return trajectory, self.planner_data, commanded_actions_nkf

    def _compute_objective_value(self):
        p = self.params.objective_fn_params
        if p.obj_type == 'valid_mean':
            self.vehicle_trajectory.update_valid_mask_nk()
        else:
            assert (p.obj_type in ['valid_mean', 'mean'])
        obj_val = tf.squeeze(
            self.obj_fn.evaluate_function(self.vehicle_trajectory))
        return obj_val

    @staticmethod
    def _init_obj_fn(self):
        """
        Initialize the objective function given sim params
        """
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
    def _init_planner(self):
        p = self.params
        obj_fn = self.obj_fn
        return p.planner_params.planner(obj_fn=obj_fn,
                                        params=p.planner_params)

    @staticmethod
    def _init_fmm_map(self, goal_pos_n2 = None):
        p = self.params
        obstacle_map = self.obstacle_map
        obstacle_occupancy_grid = obstacle_map.create_occupancy_grid_for_map()

        if goal_pos_n2 is None:
            goal_pos_n2 = self.goal_config.position_nk2()[0]

        return FmmMap.create_fmm_map_based_on_goal_position(
            goal_positions_n2=goal_pos_n2,
            map_size_2=np.array(p.obstacle_map_params.map_size_2),
            dx=p.obstacle_map_params.dx,
            map_origin_2=p.obstacle_map_params.map_origin_2,
            mask_grid_mn=obstacle_occupancy_grid)

    @staticmethod
    def _init_system_dynamics(self):
        """
        If there is a control pipeline (i.e. model based method)
        return its system_dynamics. Else create a new system_dynamics
        instance.
        """
        params = self.params
        try:
            planner = self.planner
            return planner.control_pipeline.system_dynamics
        except AttributeError:
            p = params.planner_params.control_pipeline_params.system_dynamics_params
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
                assert (False)

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


    def _enforce_episode_termination_conditions(self):
        p = self.params
        time_idxs = []
        for condition in p.episode_termination_reasons:
            time_idxs.append(
                self._compute_time_idx_for_termination_condition(condition))
        try:
            idx = np.argmin(time_idxs)
        except ValueError:
            idx = np.argmin([time_idx.numpy() for time_idx in time_idxs])

        try:
            termination_time = time_idxs[idx].numpy()
        except ValueError:
            termination_time = time_idxs[idx]

        if termination_time != np.inf:
            end_episode = True
            for i, condition in enumerate(p.episode_termination_reasons):
                if (time_idxs[i].numpy() != np.inf):
                    color = "green"
                    if (condition is "Timeout"):
                        color = "blue"
                    elif (condition is "Collision"):
                        color = "red"
                    self.termination_cause = color
            # clipping the trajectory only ends it early, we want it to actually reach the goal
            # vehicle_trajectory.clip_along_time_axis(termination_time)
            if(self.planner is not None and self.planner_data is not None):
                self.planner_data, planner_data_last_step, last_step_data_valid = \
                    self.planner.mask_and_concat_data_along_batch_dim(
                        self.planner_data,
                        k=termination_time
                    )
                commanded_actions_1kf = tf.concat(self.commanded_actions_nkf,
                                                  axis=1)[:, :termination_time]

                # If all of the data was masked then
                # the episode simulated is not valid
                valid_episode = True
                if self.planner_data['system_config'] is None:
                    valid_episode = False
                episode_data = {
                    'vehicle_trajectory': self.vehicle_trajectory,
                    'vehicle_data': self.planner_data,
                    'vehicle_data_last_step': planner_data_last_step,
                    'last_step_data_valid': last_step_data_valid,
                    'episode_type': idx,
                    'valid_episode': valid_episode,
                    'commanded_actions_1kf': commanded_actions_1kf
                }
            else:
                episode_data = {}
        else:
            end_episode = False
            episode_data = {}
        self.end_episode = end_episode
        self.episode_data = episode_data

    def _compute_time_idx_for_termination_condition(self, condition):
        """
        For a given trajectory termination condition (i.e. timeout, collision, etc.)
        computes the earliest time index at which this condition is met. Returns
        infinity if a condition is not met.
        """
        if condition == 'Timeout':
            time_idx = self._compute_time_idx_for_timeout()
        elif condition == 'Collision':
            time_idx = self._compute_time_idx_for_collision()
        elif condition == 'Success':
            time_idx = self._compute_time_idx_for_success()
        else:
            raise NotImplementedError

        return time_idx

    def _compute_time_idx_for_timeout(self):
        """
        If vehicle_trajectory has exceeded episode_horizon,
        return episode_horizon, else return infinity.
        """
        if self.vehicle_trajectory.k >= self.params.episode_horizon:
            time_idx = tf.constant(self.params.episode_horizon)
        else:
            time_idx = tf.constant(np.inf)
        return time_idx

    def _compute_time_idx_for_collision(self, use_current_config=None):
        """
        Compute and return the earliest time index of collision in vehicle
        trajectory. If there is no collision return infinity.
        """
        if(use_current_config is None):
            pos_1k2 = self.vehicle_trajectory.position_nk2()
        else:
            pos_1k2 = self.get_current_config().position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        collisions = tf.where(tf.less(obstacle_dists_1k, 0.0))
        collision_idxs = collisions[:, 1]
        if tf.size(collision_idxs).numpy() != 0:
            time_idx = collision_idxs[0]
            self.collision_point_k = self.vehicle_trajectory.k
        else:
            time_idx = tf.constant(np.inf)
        return time_idx

    def _dist_to_goal(self, use_euclidean=False):
        """Calculate the FMM distance between
        each state in trajectory and the goal."""
        for objective in self.obj_fn.objectives:
            if isinstance(objective, GoalDistance):
                euclidean = 0
                # also compute euclidean distance as a heuristic
                if use_euclidean:
                    diff_x = self.vehicle_trajectory.position_nk2()[0][-1][0]
                    - self.goal_config.position_nk2()[0][0][0]
                    diff_y = self.vehicle_trajectory.position_nk2()[0][-1][1]
                    - self.goal_config.position_nk2()[0][0][1]
                    euclidean = np.sqrt(diff_x**2 + diff_y**2)
                dist_to_goal_nk = objective.compute_dist_to_goal_nk(
                    self.vehicle_trajectory) + euclidean
        return dist_to_goal_nk

    def _compute_time_idx_for_success(self):
        """
        Compute and return the earliest time index of success (reaching the goal region)
        in vehicle trajectory. If there is no collision return infinity.
        """
        dist_to_goal_1k = self._dist_to_goal(use_euclidean=False)
        successes = tf.where(
            tf.less(dist_to_goal_1k, self.params.goal_cutoff_dist))
        success_idxs = successes[:, 1]
        if tf.size(success_idxs).numpy() != 0:
            time_idx = success_idxs[0]
        else:
            time_idx = tf.constant(np.inf)
        return time_idx

    def apply_control_open_loop(self, start_config, control_nk2,
                                T, sim_mode='ideal'):
        """
        Apply control commands in control_nk2 in an open loop
        fashion to the system starting from start_config.
        """
        x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
        applied_actions = []
        states = [x0_n1d * 1.]
        x_next_n1d = x0_n1d * 1.
        for t in range(T):
            u_n1f = control_nk2[:, t:t + 1]
            x_next_n1d = self.system_dynamics.simulate(
                x_next_n1d, u_n1f, mode=sim_mode)

            # Append the applied action to the action list
            if sim_mode == 'ideal':
                applied_actions.append(u_n1f)
            elif sim_mode == 'realistic':
                # TODO: This line is intended for a real hardware setup.
                # If running this code on a real robot the user will need to
                # implement hardware.state_dx such that it reflects the current
                # sensor reading of the robot's applied actions
                applied_actions.append(
                    np.array(self.system_dynamics.hardware.state_dx * 1.)[None, None])
            else:
                assert(False)

            states.append(x_next_n1d)

        commanded_actions_nkf = tf.concat([control_nk2[:, :T], u_n1f], axis=1)
        u_nkf = tf.concat(applied_actions, axis=1)
        x_nkd = tf.concat(states, axis=1)
        trajectory = self.system_dynamics.assemble_trajectory(x_nkd,
                                                              u_nkf,
                                                              pad_mode='repeat')
        return trajectory, commanded_actions_nkf

    def apply_control_closed_loop(self, start_config, trajectory_ref,
                                  k_array_nTf1, K_array_nTfd, T,
                                  sim_mode='ideal'):
        """
        Apply LQR feedback control to the system to track trajectory_ref
        Here k_array_nTf1 and K_array_nTfd are tensors of dimension
        (n, self.T-1, f, 1) and (n, self.T-1, f, d) respectively.
        """
        with tf.name_scope('apply_control'):
            x0_n1d, _ = self.system_dynamics.parse_trajectory(start_config)
            assert(len(x0_n1d.shape) == 3)  # [n,1,x_dim]
            angle_dims = self.system_dynamics._angle_dims
            commanded_actions_nkf = []
            applied_actions = []
            states = [x0_n1d * 1.]
            x_ref_nkd, u_ref_nkf = self.system_dynamics.parse_trajectory(
                trajectory_ref)
            x_next_n1d = x0_n1d * 1.
            for t in range(T):
                x_ref_n1d, u_ref_n1f = x_ref_nkd[:,
                                                 t:t + 1], u_ref_nkf[:, t:t + 1]
                error_t_n1d = x_next_n1d - x_ref_n1d

                # TODO: Currently calling numpy() here as tfe.DEVICE_PLACEMENT_SILENT
                # is not working to place non-gpu ops (i.e. mod) on the cpu
                # turning tensors into numpy arrays is a hack around this.
                error_t_n1d = tf.concat([error_t_n1d[:, :, :angle_dims],
                                         angle_normalize(
                                             error_t_n1d[:, :, angle_dims:angle_dims + 1].numpy()),
                                         error_t_n1d[:, :, angle_dims + 1:]],
                                        axis=2)
                fdback_nf1 = tf.matmul(K_array_nTfd[:, t],
                                       tf.transpose(error_t_n1d, perm=[0, 2, 1]))
                u_n1f = u_ref_n1f + tf.transpose(k_array_nTf1[:, t] + fdback_nf1,
                                                 perm=[0, 2, 1])

                x_next_n1d = self.system_dynamics.simulate(
                    x_next_n1d, u_n1f, mode=sim_mode)

                commanded_actions_nkf.append(u_n1f)
                # Append the applied action to the action list
                if sim_mode == 'ideal':
                    applied_actions.append(u_n1f)
                elif sim_mode == 'realistic':
                    # TODO: This line is intended for a real hardware setup.
                    # If running this code on a real robot the user will need to
                    # implement hardware.state_dx such that it reflects the current
                    # sensor reading of the robot's applied actions
                    applied_actions.append(
                        np.array(self.system_dynamics.hardware.state_dx * 1.)[None, None])
                else:
                    assert(False)

                states.append(x_next_n1d)

            commanded_actions_nkf.append(u_n1f)
            commanded_actions_nkf = tf.concat(commanded_actions_nkf, axis=1)
            u_nkf = tf.concat(applied_actions, axis=1)
            x_nkd = tf.concat(states, axis=1)
            trajectory = self.system_dynamics.assemble_trajectory(x_nkd,
                                                                  u_nkf,
                                                                  pad_mode='repeat')
            return trajectory, commanded_actions_nkf
