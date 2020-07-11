import tensorflow as tf
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
import matplotlib as mpl
mpl.use('Agg') # for rendering without a display
import matplotlib.pyplot as plt
from utils.utils import touch, print_colors
import numpy as np
import copy, os, glob, imageio
import time
from trajectory.trajectory import SystemConfig, Trajectory
from simulators.simulator_helper import SimulatorHelper
from simulators.agent import Agent
from simulators.sim_state import SimState, HumanState
from utils.fmm_map import FmmMap
from utils.utils import print_colors, natural_sort
from params.renderer_params import get_path_to_humanav

class CentralSimulator(SimulatorHelper):

    def __init__(self, params, environment, renderer=None):
        self.params = CentralSimulator.parse_params(params)
        self.r = renderer
        self.obstacle_map = self._init_obstacle_map(renderer)
        self.environment = environment
        self.humanav_dir = get_path_to_humanav()
        # theoretially all the agents can have their own system dynamics as well
        self.agents = {}
        self.states = {}
        self.wall_clock_time = 0

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
        # Much more optimized to only render topview, but can also render Humans
        p.only_render_topview = True
        return p

    def add_agent(self, a):
        name = a.get_name()
        a.simulation_init(self.params, self.obstacle_map)
        self.agents[name] = a
        # update all agents' knowledge of other agents
        Agent.all_agents = self.agents

    def exists_running_agent(self):
        for a in self.agents.values():
            # if there is even just a single agent acting 
            if (not a.end_acting):
                return True
        return False

    def exists_planning_agent(self):
        for a in self.agents.values():
            # if there is even just a single agent acting 
            if (not a.end_episode):
                return True
        return False

    def simulate(self):
        """ A function that simulates an entire episode. The agent starts
        at self.start_config, repeatedly calling _iterate to generate 
        subtrajectories. Generates a vehicle_trajectory for the episode, 
        calculates its objective value, and sets the episode_type 
        (timeout, collision, success) """
        print(print_colors()["blue"], 
            "Running simulation on", len(self.agents), "agents", 
            print_colors()["reset"])
        time_step = 0.05 # seconds for each agent to "act"
        total_time = 0 # keep track of overall time in the simulator
        for a in self.agents.values():
            # All agents share the same starting time
            a.init_time(0)
        iteration = 0
        start_time = time.clock()
        while self.exists_running_agent():
            init_time = time.clock()

            for a in self.agents.values():
                a.update(self.params, self.obstacle_map, time_step=time_step)
            
            # Takes screenshot of the simulation state as long as the update is still going
            fin_time = time.clock() - init_time
            total_time = total_time + fin_time
            
            self.print_sim_progress(iteration)
            
            self.save_state(total_time)
            iteration = iteration + 1
            
        self.wall_clock_time = time.clock() - start_time
        print("\nSimulation completed in", self.wall_clock_time, total_time, "seconds")
        self.generate_frames()
        self.save_to_gif()
        # Can also save to mp4 using imageio-ffmpeg or this bash script:
        # ffmpeg -r 10 -i simulate_obs%01d.png -vcodec mpeg4 -y movie.mp4

    def num_conditions_in_agents(self, condition):
        num = 0
        for a in self.agents.values():
            if(a.termination_cause is condition):
                num = num + 1
        return num

    def print_sim_progress(self, rendered_frames):
        print("A:", len(self.agents), 
            print_colors()["green"],
            "Success:", 
            self.num_conditions_in_agents("green"), 
            print_colors()["red"],
            "Collide:", 
            self.num_conditions_in_agents("red"), 
            print_colors()["blue"],
            "Time:", 
            self.num_conditions_in_agents("blue"), 
            print_colors()["reset"],
            "Frames:", rendered_frames,
            "\r", end="")
    
    def save_state(self, current_time):
        saved_env = copy.deepcopy(self.environment)
        # deepcopy all agents individually using a HumanState copy
        saved_agents = {}
        for a in self.agents.values():
            saved_agents[a.get_name()] = HumanState(a, deepcpy=True)
        current_state = SimState(saved_env, saved_agents, current_time)
        # Save current state to a local dictionary
        self.states[current_time] = current_state

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

    def generate_frames(self):
        num_frames = len(self.states)
        np.set_printoptions(precision=3)
        if(self.params.only_render_topview):
            # optimized to use multiple processesw
            import multiprocessing
            gif_processes = []
            for frame, s in enumerate(self.states.values()):
                gif_processes.append(
                    multiprocessing.Process(
                                    target=self.take_snapshot, 
                                    args=(s, np.array([9., 22., -np.pi/4]),"simulate_obs" + str(frame) + ".png"))
                                    )
                gif_processes[frame].start()
            for frame, p in enumerate(gif_processes):
                p.join()
                print("Generated Frames:", frame, "out of", num_frames, frame/num_frames, "\r", end="")
        else:
            for frame, s in enumerate(self.states.values()):
                self.take_snapshot(s, np.array([9., 22., -np.pi/4]),"simulate_obs" + str(frame) + ".png")
                print("Generated Frames:", frame, "out of", num_frames, frame/num_frames, "\r", end="")
        # newline to not interfere with previous prints
        print("\n")
        

    def save_to_gif(self, clear_old_files = True):
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
        print('\033[32m', "Rendered gif at", output_location, '\033[0m')
        # Clearing remaining files to not affect next render
        if clear_old_files:
            for f in files:
                os.remove(f)

    def plot_topview(self, ax, extent, traversible, human_traversible, camera_pos_13, 
                    agents, plot_quiver=False):
        ax.imshow(traversible, extent=extent, cmap='gray',
                vmin=-.5, vmax=1.5, origin='lower')
        # Plot human traversible
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
                'bo', markersize=10, label='Robot')
        ax.quiver(camera_pos_13[0], camera_pos_13[1], np.cos(
            camera_pos_13[2]), np.sin(camera_pos_13[2]))

        for i, a in enumerate(agents.values()):
            pos_2 = a.get_current_config().position_nk2().numpy()[0][0]
            heading= (a.get_current_config().heading_nk1().numpy())[0][0]
            # TODO: make colours of trajectories random rather than hardcoded
            a.get_trajectory().render(ax, freq=1, color=None, plot_quiver=False)
            color = 'go' # agents are green and solid unless collided
            if(a.get_collided()):
                color='ro' # collided agents are drawn red
            if(i == 0):
                # Only add label on the first humans
                ax.plot(pos_2[0], pos_2[1],
                        color, markersize=10, label='Agent')
            else:
                ax.plot(pos_2[0], pos_2[1], color, markersize=10)
            # TODO: use agent radius instead of hardcode
            ax.plot(pos_2[0], pos_2[1], color, alpha=0.2, markersize=25)
            if(plot_quiver):
                # Agent heading
                ax.quiver(pos_2[0], pos_2[1], np.cos(heading), np.sin(heading), 
                          scale=2, scale_units='inches')
                          
    def plot_images(self, p, rgb_image_1mk3, depth_image_1mk1, environment, room_center,
                    camera_pos_13, agents, current_time, filename):

        map_scale = environment["map_scale"]
        # Obstacles/building traversible
        traversible = environment["traversibles"][0]
        human_traversible = None

        if len(environment["traversibles"]) > 1 and not p.only_render_topview:
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
                    camera_pos_13, agents, plot_quiver=True)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        time_string = "T="+str(current_time)
        ax.set_title('Topview (zoomed) '+time_string, fontsize=20)

        # Render entire map-view from the top
        # to keep square plot
        outer_zoom = min(traversible.shape[0], traversible.shape[1]) * map_scale
        ax = fig.add_subplot(1, num_frames, 2)
        ax.set_xlim(0., outer_zoom)
        ax.set_ylim(0., outer_zoom)
        self.plot_topview(ax, extent, traversible, human_traversible,
                        camera_pos_13, agents)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Topview '+time_string, fontsize=20)

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

    def render_rgb_and_depth(self, r, camera_pos_13, dx_m, human_visible=True):
        # Convert from real world units to grid world units
        camera_grid_world_pos_12 = camera_pos_13[:, :2]/dx_m

        # Render RGB and Depth Images. The shape of the resulting
        # image is (1 (batch), m (width), k (height), c (number channels))
        rgb_image_1mk3 = r._get_rgb_image(
            camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=True)

        depth_image_1mk1, _, _ = r._get_depth_image(
            camera_grid_world_pos_12, camera_pos_13[:, 2:3], xy_resolution=.05, 
            map_size=1500, pos_3=camera_pos_13[0, :3], human_visible=True)

        return rgb_image_1mk3, depth_image_1mk1

    def take_snapshot(self, state, camera_pos_13, filename):
        """
        takes screenshot of a specific state of the world
        """
        room_center = np.array([12., 17., 0.])
        rgb_image_1mk3 = None
        depth_image_1mk1 = None
        if self.params.humanav_params.render_with_display and not self.params.only_render_topview:
            # environment should hold building and human traversibles
            assert(len(state.get_environment()["traversibles"]) == 2)
            # only when rendering with opengl
            for a in state.get_agents().values():
                self.r.update_human(a) #Agent.agent_to_human(a, human_exists=True))
            # Update human traversible
            state.get_environment()["traversibles"][1] = self.r.get_human_traversible()
            # compute the rgb and depth images
            rgb_image_1mk3, depth_image_1mk1 = \
                self.render_rgb_and_depth(self.r, np.array([camera_pos_13]), 
                                          state.get_environment()["map_scale"], human_visible=True)
        # plot the rbg, depth, and topview images if applicable
        self.plot_images(self.params, rgb_image_1mk3, depth_image_1mk1, 
                        state.get_environment(), room_center, camera_pos_13, 
                        state.get_agents(), state.get_time(), filename)


