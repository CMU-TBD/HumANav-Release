import tensorflow as tf
import copy
from objectives.objective_function import ObjectiveFunction
from objectives.angle_distance import AngleDistance
from objectives.goal_distance import GoalDistance
from objectives.obstacle_avoidance import ObstacleAvoidance

from humans.human import Human
from utils.fmm_map import FmmMap
from utils.utils import print_colors
import random
import string
import math
import numpy as np
import sys
import os
import pickle


class Agent():
    def __init__(self, start, goal, planner=None):
        self.start_config = start
        self.current_config = copy.copy(start)
        self.goal_config = goal

        self.planner = planner

        self.obj_fn = None  # Until called by simulator
        self.obj_val = None

        self.fmm_map = None

        self.end_episode = False
        self.termination_cause = None
        self.episode_data = None
        self.vehicle_trajectory = None
        self.vehicle_data = None
        self.planner_data = None
        self.last_step_data_valid = None
        self.episode_type = None
        self.valid_episode = None
        self.commanded_actions_1kf = None
        self.commanded_actions_nkf = []

    def human_to_agent(self, human):
        """
        Sample a new agent from a human with configs
        """
        return Agent(human.get_start_config(), human.get_goal_config())

    def _compute_objective_value(self, params):
        p = params.objective_fn_params
        if p.obj_type == 'valid_mean':
            self.vehicle_trajectory.update_valid_mask_nk()
        else:
            assert (p.obj_type in ['valid_mean', 'mean'])
        obj_val = tf.squeeze(
            self.obj_fn.evaluate_function(self.vehicle_trajectory))
        return obj_val

    def _init_obj_fn(self, p, obstacle_map):
        """
        Initialize the objective function given sim params
        """
        obj_fn = ObjectiveFunction(p.objective_fn_params)
        if not p.avoid_obstacle_objective.empty():
            obj_fn.add_objective(
                ObstacleAvoidance(params=p.avoid_obstacle_objective,
                                  obstacle_map=obstacle_map))
        if not p.goal_distance_objective.empty():
            obj_fn.add_objective(
                GoalDistance(params=p.goal_distance_objective,
                             fmm_map=obstacle_map.fmm_map))
        if not p.goal_angle_objective.empty():
            obj_fn.add_objective(
                AngleDistance(params=p.goal_angle_objective,
                              fmm_map=obstacle_map.fmm_map))
        return obj_fn

    def _init_planner(self, params):
        p = params
        return p.planner_params.planner(obj_fn=self.obj_fn,
                                        params=p.planner_params)

    def _update_fmm_map(self, params, obstacle_map):
        """
        For SBPD the obstacle map does not change,
        so just update the goal position.
        """
        goal_pos_n2 = self.goal_config.position_nk2()[:, 0]
        if self.fmm_map is not None:
            self.fmm_map.change_goal(goal_pos_n2)
        else:
            self.fmm_map = self._init_fmm_map(params, obstacle_map,
                                              goal_pos_n2)
        self._update_obj_fn(obstacle_map)

    def _init_fmm_map(self, params, obstacle_map, goal_pos_n2=None):
        p = params
        self.obstacle_occupancy_grid = \
            obstacle_map.create_occupancy_grid_for_map()

        if goal_pos_n2 is None:
            goal_pos_n2 = self.goal_config.position_nk2()[0]

        return FmmMap.create_fmm_map_based_on_goal_position(
            goal_positions_n2=goal_pos_n2,
            map_size_2=np.array(p.obstacle_map_params.map_size_2),
            dx=p.obstacle_map_params.dx,
            map_origin_2=p.obstacle_map_params.map_origin_2,
            mask_grid_mn=self.obstacle_occupancy_grid)

    def _update_obj_fn(self, obstacle_map):

        # Update the objective function to use a new
        # obstacle_map and fmm map
        # PROBABLY never going to use this

        for objective in self.obj_fn.objectives:
            if isinstance(objective, ObstacleAvoidance):
                objective.obstacle_map = obstacle_map
            elif isinstance(objective, GoalDistance):
                objective.fmm_map = self.fmm_map
            elif isinstance(objective, AngleDistance):
                objective.fmm_map = self.fmm_map
            else:
                assert (False)

    def _enforce_episode_termination_conditions(self, params, obstacle_map):
        p = params
        time_idxs = []
        for condition in p.episode_termination_reasons:
            time_idxs.append(
                self._compute_time_idx_for_termination_condition(
                    params, obstacle_map, condition))
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
                    print(print_colors()[color], "Terminated due to",
                          condition,
                          print_colors()["reset"])
                    if (condition is "Timeout"):
                        print(print_colors()["blue"], "Max time:",
                              p.episode_horizon,
                              print_colors()["reset"])
            # clipping the trajectory only ends it early, we want it to actually reach the goal
            # vehicle_trajectory.clip_along_time_axis(termination_time)
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
            end_episode = False
            episode_data = {}
        self.end_episode = end_episode
        self.episode_data = episode_data

    def _compute_time_idx_for_termination_condition(self, params, obstacle_map,
                                                    condition):
        """
        For a given trajectory termination condition (i.e. timeout, collision, etc.)
        computes the earliest time index at which this condition is met. Returns
        infinity if a condition is not met.
        """
        if condition == 'Timeout':
            time_idx = self._compute_time_idx_for_timeout(params)
        elif condition == 'Collision':
            time_idx = self._compute_time_idx_for_collision(
                obstacle_map, params)
        elif condition == 'Success':
            time_idx = self._compute_time_idx_for_success(params)
        else:
            raise NotImplementedError

        return time_idx

    def _compute_time_idx_for_timeout(self, params):
        """
        If vehicle_trajectory has exceeded episode_horizon,
        return episode_horizon, else return infinity.
        """
        if self.vehicle_trajectory.k >= params.episode_horizon:
            time_idx = tf.constant(params.episode_horizon)
        else:
            time_idx = tf.constant(np.inf)
        return time_idx

    def _compute_time_idx_for_collision(self, obstacle_map, params):
        """
        Compute and return the earliest time index of collision in vehicle
        trajectory. If there is no collision return infinity.
        """
        pos_1k2 = self.vehicle_trajectory.position_nk2()
        obstacle_dists_1k = obstacle_map.dist_to_nearest_obs(pos_1k2)
        collisions = tf.where(tf.less(obstacle_dists_1k, 0.0))
        collision_idxs = collisions[:, 1]
        if tf.size(collision_idxs).numpy() != 0:
            time_idx = collision_idxs[0]
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
                    diff_x = self.vehicle_trajectory.position_nk2(
                    )[0][-1][0] - self.goal_config.position_nk2()[0][0][0]
                    diff_y = self.vehicle_trajectory.position_nk2(
                    )[0][-1][1] - self.goal_config.position_nk2()[0][0][1]
                    euclidean = np.sqrt(diff_x**2 + diff_y**2)
                dist_to_goal_nk = objective.compute_dist_to_goal_nk(
                    self.vehicle_trajectory) + euclidean
        return dist_to_goal_nk

    def _compute_time_idx_for_success(self, params):
        """
        Compute and return the earliest time index of success (reaching the goal region)
        in vehicle trajectory. If there is no collision return infinity.
        """
        dist_to_goal_1k = self._dist_to_goal(use_euclidean=False)
        successes = tf.where(tf.less(dist_to_goal_1k, params.goal_cutoff_dist))
        success_idxs = successes[:, 1]
        if tf.size(success_idxs).numpy() != 0:
            time_idx = success_idxs[0]
        else:
            time_idx = tf.constant(np.inf)
        return time_idx
