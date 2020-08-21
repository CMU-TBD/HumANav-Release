import numpy as np
import sys
import os
import copy
import time
from objectives.goal_distance import GoalDistance


class AgentHelper(object):
    def _enforce_episode_termination_conditions(self):
        p = self.params
        time_idxs = []
        for condition in p.episode_termination_reasons:
            time_idxs.append(
                self._compute_time_idx_for_termination_condition(condition))
        try:
            idx = np.argmin(time_idxs)
        except ValueError:
            idx = np.argmin([time_idx for time_idx in time_idxs])

        try:
            termination_time = time_idxs[idx]
        except ValueError:
            termination_time = time_idxs[idx]

        if termination_time != np.inf:
            end_episode = True
            for i, condition in enumerate(p.episode_termination_reasons):
                if (time_idxs[i] != np.inf):
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
                commanded_actions_1kf = np.concatenate(self.commanded_actions_nkf,
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
        if(self.planner is not None):
            if self.vehicle_trajectory.k >= self.params.episode_horizon:
                time_idx = np.array(self.params.episode_horizon)
            else:
                time_idx = np.array(np.inf)
        else:
            time_idx = np.array(np.inf)
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
        collisions = np.where(np.less(obstacle_dists_1k, 0.0))
        collision_idxs = collisions[1]
        if np.size(collision_idxs) != 0:
            time_idx = collision_idxs[0]
            self.collision_point_k = self.vehicle_trajectory.k
        else:
            time_idx = np.array(np.inf)
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
        successes = np.where(
            np.less(dist_to_goal_1k, self.params.goal_cutoff_dist))
        success_idxs = successes[1]
        if np.size(success_idxs) != 0:
            time_idx = success_idxs[0]
        else:
            time_idx = np.array(np.inf)
        return time_idx

    @staticmethod
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

        commanded_actions_nkf = np.concatenate(
            [control_nk2[:, :T], u_n1f], axis=1)
        u_nkf = np.concatenate(applied_actions, axis=1)
        x_nkd = np.concatenate(states, axis=1)
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
        with np.name_scope('apply_control'):
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
                error_t_n1d = np.concatenate([error_t_n1d[:, :, :angle_dims],
                                              angle_normalize(
                                             error_t_n1d[:, :, angle_dims:angle_dims + 1]),
                    error_t_n1d[:, :, angle_dims + 1:]],
                    axis=2)
                fdback_nf1 = np.matmul(K_array_nTfd[:, t],
                                       np.transpose(error_t_n1d, perm=[0, 2, 1]))
                u_n1f = u_ref_n1f + np.transpose(k_array_nTf1[:, t] + fdback_nf1,
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
            commanded_actions_nkf = np.concatenate(
                commanded_actions_nkf, axis=1)
            u_nkf = np.concatenate(applied_actions, axis=1)
            x_nkd = np.concatenate(states, axis=1)
            trajectory = self.system_dynamics.assemble_trajectory(x_nkd,
                                                                  u_nkf,
                                                                  pad_mode='repeat')
            return trajectory, commanded_actions_nkf
