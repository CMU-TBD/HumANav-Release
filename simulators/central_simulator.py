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

    def __init__(self, params):
        self.params = params.simulator.parse_params(params)
        self.obstacle_map = self._init_obstacle_map()
        # self.obj_fn = self._init_obj_fn()
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
                if(i == 0):
                    print("start: ", a.start_config.position_nk2().numpy())
                    print("goal: ", a.goal_config.position_nk2().numpy())
                # vehicle_data = a.planner.empty_data_dict()
                if(not a.end_episode):
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
        print("Took",i,"iterations")
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

    def _init_obstacle_map(self):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p)

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

    # Functions for computing relevant metrics
    # on robot trajectories

    def _calculate_min_obs_distances(self, vehicle_trajectory):
        """Returns an array of dimension 1k where each element is the distance to the closest
        obstacle at each time step."""
        pos_1k2 = vehicle_trajectory.position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        return obstacle_dists_1k

    def _calculate_trajectory_collisions(self, vehicle_trajectory):
        """Returns an array of dimension 1k where each element is a 1 if the robot collided with an
        obstacle at that time step or 0 otherwise. """
        pos_1k2 = vehicle_trajectory.position_nk2()
        obstacle_dists_1k = self.obstacle_map.dist_to_nearest_obs(pos_1k2)
        return tf.cast(obstacle_dists_1k < 0.0, tf.float32)

    def get_metrics(self):
        """After the episode is over, call the get_metrics function to get metrics
        per episode.  Returns a structure, lists of which are passed to accumulate
        metrics static function to generate summary statistics."""
        dists_1k = self._dist_to_goal(self.vehicle_trajectory)
        init_dist = dists_1k[0, 0].numpy()
        final_dist = dists_1k[0, -1].numpy()
        collisions_mu = np.mean(self._calculate_trajectory_collisions(self.vehicle_trajectory))
        min_obs_distances = self._calculate_min_obs_distances(self.vehicle_trajectory)
        return np.array([self.obj_val,
                         init_dist,
                         final_dist,
                         self.vehicle_trajectory.k,
                         collisions_mu,
                         np.min(min_obs_distances),
                         self.episode_type])

    @staticmethod
    def collect_metrics(ms, termination_reasons=['Timeout', 'Collision', 'Success']):
        ms = np.array(ms)
        if len(ms) == 0:
            return None, None
        obj_vals, init_dists, final_dists, episode_length, collisions, min_obs_distances, episode_types = ms.T
        keys = ['Objective Value', 'Initial Distance', 'Final Distance',
                'Episode Length', 'Collisions_Mu', 'Min Obstacle Distance']
        vals = [obj_vals, init_dists, final_dists,
                episode_length, collisions, min_obs_distances]

        # mean, 25 percentile, median, 75 percentile
        fns = [np.mean, lambda x: np.percentile(x, q=25), lambda x:
               np.percentile(x, q=50), lambda x: np.percentile(x, q=75)]
        fn_names = ['mu', '25', '50', '75']
        out_vals, out_keys = [], []
        for k, v in zip(keys, vals):
            for fn, name in zip(fns, fn_names):
                _ = fn(v)
                out_keys.append('{:s}_{:s}'.format(k, name))
                out_vals.append(_)

        # Log the number of episodes
        num_episodes = len(episode_types)
        out_keys.append('Number Episodes')
        out_vals.append(num_episodes)

        # Log Percet Collision, Timeout, Success, Etc.
        for i, reason in enumerate(termination_reasons):
            out_keys.append('Percent {:s}'.format(reason))
            out_vals.append(1.*np.sum(episode_types == i) / num_episodes)

            # Log the Mean Episode Length for Each Episode Type
            episode_idxs = np.where(episode_types == i)[0]
            episode_length_for_this_episode_type = episode_length[episode_idxs]
            if len(episode_length_for_this_episode_type) > 0:
                mean_episode_length_for_this_episode_type = np.mean(episode_length_for_this_episode_type)
                out_keys.append('Mean Episode Length for {:s} Episodes'.format(reason))
                out_vals.append(mean_episode_length_for_this_episode_type)

        return out_keys, out_vals

    def start_recording_video(self, video_number):
        """ By default the simulator does not support video capture."""
        return None

    def stop_recording_video(self, video_number, video_filename):
        """ By default the simulator does not support video capture."""
        return None

    def render(self, axs, freq=4, render_waypoints=False, render_velocities=False, prepend_title='', zoom=0, markersize=10):
        if type(axs) is list or type(axs) is np.ndarray:
            self._render_trajectory(axs[0], freq, render_waypoints)

            if render_velocities:
                self._render_velocities(axs[1], axs[2])
            [ax.set_title('{:s}{:s}'.format(prepend_title, ax.get_title())) for ax in axs]
        else:
            self._render_trajectory(axs, freq, render_waypoints, zoom, markersize)
            axs.set_title('{:s}{:s}'.format(prepend_title, axs.get_title()))

    def _render_obstacle_map(self, ax):
        self.obstacle_map.render(ax)

    def _render_trajectory(self, ax, freq=4, render_waypoints = False, zoom=0, markersize=10):
        p = self.params

        self._render_obstacle_map(ax)

        if render_waypoints and 'waypoint_config' in self.vehicle_data.keys():
            # Dont want ax in a list 
            self.vehicle_trajectory.render(ax, freq=freq, plot_quiver=False)
            self._render_waypoints(ax, markersize)
        else:
            self.vehicle_trajectory.render(ax, freq=freq, plot_quiver=False)

        boundary_params = {'norm': p.goal_dist_norm, 'cutoff':
                           p.goal_cutoff_dist, 'color': 'g'}
        self.start_config.render(ax, batch_idx=0, marker='o', markersize=markersize, color='blue')
        self.goal_config.render_with_boundary(ax, batch_idx=0, marker='*', markersize=markersize, color='black',
                                              boundary_params=boundary_params)

        goal = self.goal_config.position_nk2()[0, 0]
        start = self.start_config.position_nk2()[0, 0]
        text_color = p.episode_termination_colors[self.episode_type]
        ax.set_title('Start: [{:.2f}, {:.2f}] '.format(*start) +
                     '\n Goal: [{:.2f}, {:.2f}]'.format(*goal), color=text_color)

        final_pos = self.vehicle_trajectory.position_nk2()[0, -1]
        num_waypts = 0
        if(self.vehicle_data['waypoint_config'] is not None):
            #Nonetype wont have attribute 'n', safeguard
            num_waypts = self.vehicle_data['waypoint_config'].n 

        ax.set_xlabel('Cost: {cost:.3f} '.format(cost=self.obj_val) +
                      '\n End: [{:.2f}, {:.2f}]'.format(*final_pos) + 
                      '\n Num Waypts: [{:.2f}]'.format(num_waypts), color=text_color)
        final_x = final_pos.numpy()[0]
        final_y = final_pos.numpy()[1]
        ax.plot(final_x, final_y, text_color+'.')
        
    def _render_waypoints(self, ax, plot_quiver=False, plot_text=True,text_offset=(0,0), markersize=10):
        # Plot the system configuration and corresponding
        # waypoint produced in the same color
        if(self.vehicle_data['waypoint_config'] is not None):
            system_configs = self.vehicle_data['system_config']
            waypt_configs = self.vehicle_data['waypoint_config']
            cmap = matplotlib.cm.get_cmap(self.params.waypt_cmap)
            for i, (system_config, waypt_config) in enumerate(zip(system_configs, waypt_configs)):
                color = cmap(i / system_configs.n)
                system_config.render(ax, batch_idx=0, plot_quiver=plot_quiver,
                                    marker='o', markersize=markersize, color=color)

                # Render the waypoint's number at each
                # waypoint's location
                pos_2 = system_config.position_nk2()[0, 0].numpy()
                if(plot_text):
                    ax.text(pos_2[0]+text_offset[0], pos_2[1]+text_offset[1], str(i), color=color)

    def _render_velocities(self, ax0, ax1):
        speed_k = self.vehicle_trajectory.speed_nk1()[0, :, 0].numpy()
        angular_speed_k = self.vehicle_trajectory.angular_speed_nk1()[0, :, 0].numpy()

        time = np.r_[:self.vehicle_trajectory.k]*self.vehicle_trajectory.dt

        if self.system_dynamics.simulation_params.simulation_mode == 'realistic':
            ax0.plot(time, speed_k, 'r--', label='Applied')
            ax0.plot(time, self.commanded_actions_1kf[0, :,  0], 'b--', label='Commanded')
            ax0.set_title('Linear Velocity')
            ax0.legend()

            ax1.plot(time, angular_speed_k, 'r--', label='Applied')
            ax1.plot(time, self.commanded_actions_1kf[0, :,  1], 'b--', label='Commanded')
            ax1.set_title('Angular Velocity')
            ax1.legend()
        else:

            ax0.plot(time, speed_k, 'r--')
            ax0.set_title('Linear Velocity')

            ax1.plot(time, angular_speed_k, 'r--')
            ax1.set_title('Angular Velocity')
