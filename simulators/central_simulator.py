import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg') # for rendering without a display
import matplotlib.pyplot as plt
from utils.utils import touch, print_colors
import numpy as np
import copy, os, glob, imageio
from trajectory.trajectory import SystemConfig, Trajectory
from simulators.simulator_helper import SimulatorHelper
from simulators.agent import Agent
from utils.fmm_map import FmmMap
from utils.utils import print_colors, natural_sort
from params.renderer_params import get_path_to_humanav
import matplotlib

class CentralSimulator(SimulatorHelper):

    def __init__(self, params, environment, renderer=None):
        self.params = params.simulator.parse_params(params)
        self.obstacle_map = self._init_obstacle_map(renderer)
        self.environment = environment
        self.humanav_dir = get_path_to_humanav()
        # theoretially all the agents can have their own system dynamics as well
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
        agent.obj_fn = agent._init_obj_fn(self.params, self.obstacle_map)
        agent.planner = agent._init_planner(self.params)
        agent.vehicle_data = agent.planner.empty_data_dict()
        agent.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)
        agent.system_dynamics = agent._init_system_dynamics(self.params)
        agent._update_fmm_map(self.params, self.obstacle_map)
        self.agents.append(agent)

    def exists_running_agent(self):
        for x in self.agents:
            # if there is even just a single agent acting 
            if (x.end_acting == False):
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
                a.update(self.params, self.obstacle_map)
            # Takes screenshot of the simulation state as long as the update is still going
            self.take_snapshot(np.array([9., 22., -np.pi/4]), 
                                "simulate_obs" + str(i) + ".png")
            # print("Progress: %d\r" %i, end="")
            i = i + 1
        print(" Took", i, "iterations")
        self.save_to_gif()
        # Can also save to mp4 using imageio-ffmpeg or this bash script:
        # ffmpeg -r 10 -i simulate_obs%01d.png -vcodec mpeg4 -y movie.mp4

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

    def _render_obstacle_map(self, ax, zoom=0):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(
            ax,
            margin0=p.avoid_obstacle_objective.obstacle_margin0,
            margin1=p.avoid_obstacle_objective.obstacle_margin1,
            zoom=zoom)

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
        img_nmkd = self.get_observation(
            pos_n3=data_dict['vehicle_state_nk3'][:, 0],
            **kwargs)
        return img_nmkd

    def save_to_gif(self):
        """Takes the image directory and naturally sorts the images into a singular movie.gif"""
        images = []
        IMAGES_DIR = os.path.join(self.humanav_dir, "tests/socnav/images")
        if(not os.path.exists(IMAGES_DIR)):
            print('\033[31m', "ERROR: Failed to image directory at", IMAGES_DIR, '\033[0m')
            os._exit(1) # Failure condition
        files = natural_sort(glob.glob(os.path.join(IMAGES_DIR, '*.png')))
        for filename in files:
            if(self.params.verbose_printing):
                print("appending", filename)
            images.append(imageio.imread(filename))
        output_location = os.path.join(IMAGES_DIR, 'movie.gif')
        imageio.mimsave(output_location, images)
        print('\033[32m', "SUCCESS: rendered gif at", output_location, '\033[0m')
        # Clearing remaining files to not affect next render
        for f in files:
            os.remove(f)

    def plot_topview(self, ax, extent, traversible, human_traversible, camera_pos_13, 
                    humans, plot_quiver=False):
        ax.imshow(traversible, extent=extent, cmap='gray',
                vmin=-.5, vmax=1.5, origin='lower')

        if human_traversible is not None:
            # NOTE: the human radius is only available given the openGL human modeling
            # and rendering, thus p.render_with_display must be True
            # Plot the 5x5 meter human radius grid atop the environment traversible
            alphas = np.empty(np.shape(human_traversible))
            for y in range(human_traversible.shape[1]):
                for x in range(human_traversible.shape[0]):
                    alphas[x][y] = not(human_traversible[x][y])
            ax.imshow(human_traversible, extent=extent, cmap='autumn_r',
                    vmin=-.5, vmax=1.5, origin='lower', alpha=alphas)
            alphas = np.all(np.invert(human_traversible))

        # Plot the camera
        ax.plot(camera_pos_13[0], camera_pos_13[1],
                'bo', markersize=10, label='Camera')
        ax.quiver(camera_pos_13[0], camera_pos_13[1], np.cos(
            camera_pos_13[2]), np.sin(camera_pos_13[2]))

        for i, human in enumerate(humans):
            # human_pos_2 = human.get_start_config().position_nk2().numpy()[0][0]
            # human_heading = (human.get_start_config().heading_nk1().numpy())[0][0]
            human_goal_2 = human.get_goal_config().position_nk2().numpy()[0][0]
            goal_heading = (human.get_goal_config().heading_nk1().numpy())[0][0]

            color = human.get_termination()
            human.get_trajectory().render(ax, freq=1, color=color, plot_quiver=False)
            if(i == 0):
                # Only add label on the first humans
                # ax.plot(human_pos_2[0], human_pos_2[1],
                #         'ro', markersize=10, label='Human')
                ax.plot(human_goal_2[0], human_goal_2[1], markerfacecolor="#FF7C00",
                        marker='o', markersize=10, label='Goal')
            else:
                # ax.plot(human_pos_2[0], human_pos_2[1], 'ro', markersize=10)
                ax.plot(human_goal_2[0], human_goal_2[1],
                        markerfacecolor="#FF7C00", marker='o', markersize=10)
            if(plot_quiver):
                # human start quiver
                # ax.quiver(human_pos_2[0], human_pos_2[1], np.cos(human_heading), np.sin(
                #     human_heading), scale=2, scale_units='inches')
                # goal quiver
                ax.quiver(human_goal_2[0], human_goal_2[1], np.cos(goal_heading), np.sin(
                    goal_heading), scale=2, scale_units='inches')


    def plot_images(self, p, rgb_image_1mk3, depth_image_1mk1, environment, room_center,
                    camera_pos_13, humans, filename):

        map_scale = environment["map_scale"]
        # Obstacles/building traversible
        traversible = environment["traversibles"][0]
        human_traversible = None

        if len(environment["traversibles"]) > 1:
            human_traversible = environment["traversibles"][1]
        # Compute the real_world extent (in meters) of the traversible
        extent = [0., traversible.shape[1], 0., traversible.shape[0]]
        extent = np.array(extent) * map_scale

        num_frames = 2
        if rgb_image_1mk3 is not None:
            num_frames = num_frames + 1
        if depth_image_1mk1 is not None:
            num_frames = num_frames + 1
        
        img_size = 10
        fig = plt.figure(figsize=(num_frames * img_size, img_size))

        # Plot the 5x5 meter occupancy grid centered around the camera
        zoom = 5.5  # zoom in by a constant amount
        ax = fig.add_subplot(1, num_frames, 1)
        ax.set_xlim([room_center[0] - zoom, room_center[0] + zoom])
        ax.set_ylim([room_center[1] - zoom, room_center[1] + zoom])
        self.plot_topview(ax, extent, traversible, human_traversible,
                    camera_pos_13, humans, plot_quiver=True)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Topview (zoomed)')

        # Render entire map-view from the top
        # to keep square plot
        outer_zoom = min(traversible.shape[0], traversible.shape[1]) * map_scale
        ax = fig.add_subplot(1, num_frames, 2)
        ax.set_xlim(0., outer_zoom)
        ax.set_ylim(0., outer_zoom)
        self.plot_topview(ax, extent, traversible,
                    human_traversible, camera_pos_13, humans)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Topview')

        if rgb_image_1mk3 is not None:
            # Plot the RGB Image
            ax = fig.add_subplot(1, num_frames, 3)
            ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('RGB')

        if depth_image_1mk1 is not None:
            # Plot the Depth Image
            ax = fig.add_subplot(1, num_frames, 4)
            ax.imshow(depth_image_1mk1[0, :, :, 0].astype(np.uint8), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Depth')

        full_file_name = os.path.join(self.humanav_dir, 'tests/socnav/images', filename)
        if(not os.path.exists(full_file_name)):
            if(self.params.verbose_printing):
                print('\033[31m', "Failed to find:", full_file_name,
                    '\033[33m', "and therefore it will be created", '\033[0m')
            touch(full_file_name)  # Just as the bash command
        fig.savefig(full_file_name, bbox_inches='tight', pad_inches=0)
        fig.clear()
        plt.close(fig)
        del fig
        plt.clf()
        if(self.params.verbose_printing):
            print('\033[32m', "Successfully rendered:", full_file_name, '\033[0m')

    def take_snapshot(self, camera_pos_13, filename):
        """
        takes screenshot
        """
        humans = []
        for a in self.agents:
            humans.append(Agent.agent_to_human(Agent, a))

        room_center = np.array([12., 17., 0.])

        self.plot_images(self.params, None, None, self.environment, room_center,
                    camera_pos_13, humans, filename)


