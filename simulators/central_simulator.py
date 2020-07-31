import tensorflow as tf
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
import matplotlib as mpl
mpl.use('Agg') # for rendering without a display
import matplotlib.pyplot as plt
import numpy as np
import copy, os, glob, imageio
import time, threading, multiprocessing
from humans.human import Human
from humans.recorded_human import PrerecordedHuman
from humanav.humanav_renderer_multi import HumANavRendererMulti
from simulators.robot_agent import RoboAgent
from trajectory.trajectory import SystemConfig, Trajectory
from simulators.simulator_helper import SimulatorHelper
from simulators.agent import Agent
from simulators.sim_state import SimState, HumanState, AgentState
from utils.fmm_map import FmmMap
from utils.utils import touch, print_colors, natural_sort
from params.renderer_params import get_path_to_humanav

class CentralSimulator(SimulatorHelper):

    def __init__(self, params, environment, renderer=None):
        self.params = CentralSimulator.parse_params(params)
        self.r = renderer
        self.obstacle_map = self._init_obstacle_map(renderer)
        self.environment = environment
        # keep track of all agents in dictionary with names as the key
        self.agents = {}
        # keep track of all robots in dictionary with names as the key
        self.robots = {}
        # keep track of all prerecorded humans in a dictionary like the otherwise
        self.prerecs = {}
        # keep a single (important) robot as a value
        self.robot = None
        self.states = {}
        self.wall_clock_time = 0
        self.t = 0

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        # Parse the dependencies
        p.humanav_dir = get_path_to_humanav()
        p.planner_params.planner.parse_params(p.planner_params)
        p.obstacle_map_params.obstacle_map.parse_params(p.obstacle_map_params)
        # Time discretization step
        dt = p.planner_params.control_pipeline_params.system_dynamics_params.dt
        # Updating horizons
        p.episode_horizon = max(1, int(np.ceil(p.episode_horizon_s / dt)))
        p.control_horizon = max(1, int(np.ceil(p.control_horizon_s / dt)))
        p.dt = dt
        # Much more optimized to only render topview, but can also render Humans
        if(not p.render_3D):
            print("Printing Topview movie with multithreading")
        else:
            print("Printing 3D movie sequentially")
        # verbose printing
        p.verbose_printing = False
        # Save memory by updating renderer rather than deepcopying it, but very slow & sequential
        # p.use_one_renderer = True
        return p

    def add_agent(self, a):
        name = a.get_name()
        if(isinstance(a, RoboAgent)):
            # Same simulation init for agents *however* the robot wont include a planner
            a.simulation_init(self.params, self.obstacle_map, with_planner=False)
            self.robots[name] = a
            self.robot = a
        elif (isinstance(a, PrerecordedHuman)):
            a.simulation_init(self.params, self.obstacle_map, with_planner=False)
            self.prerecs[name] = a
        else:
            # assert(isinstance(a, Human))) # TODO: could be a prerecordedAgent
            a.simulation_init(self.params, self.obstacle_map, with_planner=True)
            self.agents[name] = a

    def exists_running_agent(self):
        for a in self.agents.values():
            # if there is even just a single agent acting 
            if (not a.end_acting):
                return True
        return False

    def exists_running_prerec(self):
        for a in self.prerecs.values():
            # if there is even just a single prerec acting 
            if (not a.end_acting):
                return True
        return False

    """BEGIN thread utils"""

    def init_robot_thread(self):
        # wait for joystick connection to be established
        if(self.robot is not None):
            self.robot.establish_joystick_receiver_connection()
            time.sleep(0.01)
            self.robot.establish_joystick_sender_connection()
            self.robot.update_time(0)
            robot_thread = threading.Thread(target=self.robot.update)
            robot_thread.start()
            return robot_thread
        print(print_colors()["red"],"No robot in simulator",print_colors()['reset'])
        return None

    def decommission_robot(self, thread):
        if(thread is not None):
            assert(self.robot is not None)
            # turn off the robot
            self.robot.power_off()
            # close robot agent threads
            if(thread.is_alive()):
                thread.join()
            del(thread)
        return        

    def init_agent_threads(self, time, t_step, current_state):
        agent_threads = []
        for a in self.agents.values():
            agent_threads.append(threading.Thread(target=a.update, args=(time, t_step, current_state,)))
        return agent_threads

    def init_prerec_threads(self, time):
        prerec_threads = []
        for a in self.prerecs.values():
            if(not a.end_acting):
                prerec_threads.append(threading.Thread(target=a.update, args=(time,)))
            else:
                self.prerecs.pop(a.get_name())
                del(a)
        return prerec_threads

    def start_threads(self, thread_group):
        for t in thread_group: 
            t.start()
    
    def join_threads(self, thread_group):
        for t in thread_group:
            t.join()
            del(t)

    """END thread utils"""

    def simulate(self):
        """ A function that simulates an entire episode. The agent starts
        at self.start_config, repeatedly calling _iterate to generate 
        subtrajectories. Generates a vehicle_trajectory for the episode, 
        calculates its objective value, and sets the episode_type 
        (timeout, collision, success) """
        num_agents = len(self.agents) + len(self.prerecs)
        print("Running simulation on", num_agents, "agents")
        
        r_t = self.init_robot_thread()
        # continue to spawn the simulation with an established (independent) connection

        # keep track of wall-time in the simulator
        start_time = time.clock()
        # save initial state before the simulator is spawned
        self.t = 0
        delta_t = 3*self.params.dt
        # delta_t = XYZ # NOTE: can tune this number to be whatever one wants
        # TODO: make all agents, robots, and prerecs be internal threads in THIS update 
        while self.exists_running_agent() or self.exists_running_prerec():
            # update "wall clock" time
            wall_clock = time.clock() - start_time
            # Takes screenshot of the simulation state as long as the update is still going
            current_state = self.save_state(self.t, wall_clock) # saves to self.states and returns most recent
            # Complete thread operations
            agent_threads = self.init_agent_threads(self.t, delta_t, current_state)
            prerec_threads = self.init_prerec_threads(self.t)
            # start all thread groups
            self.start_threads(agent_threads)
            self.start_threads(prerec_threads)
            # join all thread groups
            self.join_threads(agent_threads)
            self.join_threads(prerec_threads)
            # capture time after all the agents have updated
            self.t += delta_t # update "simulaiton time"
            # print simulation progress
            iteration = int(self.t * (1./delta_t))
            self.print_sim_progress(iteration)
            # if (iteration > 40 * num_agents):
            #     # hard limit of 40 frames per agent
            #     break
        # free all the agents
        for a in self.agents.values():
            del(a)

        # free all the prerecs
        for p in self.prerecs.values():
            del(p)

        self.decommission_robot(r_t)
        
        # capture wall clock time
        wall_clock = time.clock() - start_time

        print("\nSimulation completed in", wall_clock, "seconds")

        # TODO: make SURE to clean the simulation of all "leaks" since these are 
        # MULTIPLIED for multiple processes

        # convert the saved states to rendered png's to be rendered into a movie
        self.generate_frames()

        # convert all the generated frames into a gif file
        self.save_to_gif(clear_old_files = False)
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
            "Frames:", 
            rendered_frames+1,
            "T = %.3f" % (self.t),
            "\r", end="")
    
    def save_state(self, simulator_time, wall_clock_time):
        #TODO: when using a modular environment, make saved_env a deepcopy
        saved_env = self.environment
        # deepcopy all agents individually using a HumanState copy
        saved_agents = {}
        for a in self.agents.values():
            saved_agents[a.get_name()] = HumanState(a, deepcpy=True)
        # deepcopy all prerecorded agents
        saved_prerecs = {}
        for a in self.prerecs.values():
            saved_prerecs[a.get_name()] = HumanState(a, deepcpy=True)
        # Save all the robots
        saved_robots = {}
        for r in self.robots.values():
            saved_robots[r.get_name()] = AgentState(r, deepcpy=True)
        current_state = SimState(saved_env, 
                                saved_agents, saved_prerecs, saved_robots, 
                                simulator_time, wall_clock_time
                                )
        # Save current state to a class dictionary indexed by simulator time
        self.states[simulator_time] = current_state
        return current_state

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

    def generate_frames(self, filename="obs"):
        num_frames = len(self.states)
        np.set_printoptions(precision=3)
        if(not self.params.render_3D):
            # optimized to use multiple processes
            # TODO: put a limit on the maximum number of processes that can be run at once
            gif_processes = []
            for p, s in enumerate(self.states.values()):
                # pool.apply_async(self.take_snapshot, args=(s, filename + str(p) + ".png"))
                gif_processes.append(multiprocessing.Process(
                                        target=self.take_snapshot, 
                                        args=(s, filename + str(p) + ".png"))
                                        )
                gif_processes[-1].start()
                p += 1
                print("Started processes:", p, "out of", num_frames, "%.3f" % (p/num_frames), "\r", end="")
            print("\n")
            for frame, p in enumerate(gif_processes):
                p.join()
                frame += 1
                print("Generated Frames:", frame, "out of", num_frames, "%.3f" % (frame/num_frames), "\r", end="")
        else:
            # generate frames sequentially (non multiproceses)
            for frame, s in enumerate(self.states.values()):
                self.take_snapshot(s, filename + str(frame) + ".png")
                frame += 1
                print("Generated Frames:", frame, "out of", num_frames, "%.3f" % (frame/num_frames), "\r", end="")
                del(s) # free the state from memory
        
        # newline to not interfere with previous prints
        print("\n")
        
    def save_to_gif(self, clear_old_files = True, with_multiprocessing=True):
        num_robots = len(self.robots)
        rendering_processes = []
        for i in range(num_robots):
            dirname = "tests/socnav/sim_movie" + str(i)
            IMAGES_DIR = os.path.join(self.params.humanav_dir, dirname)
            if(with_multiprocessing):
                # little use to use pools here, since this is for multiple robot agents in a scene
                # and the assumption here is that is a small number
                rendering_processes.append(multiprocessing.Process(
                                        target=self._save_to_gif, 
                                        args=(IMAGES_DIR, clear_old_files))
                                        )
                rendering_processes[i].start()
            else:
                self._save_to_gif(IMAGES_DIR, clear_old_files=clear_old_files) # sequentially
        
        for p in rendering_processes:
            p.join()

    def _save_to_gif(self, IMAGES_DIR, clear_old_files = True):
        """Takes the image directory and naturally sorts the images into a singular movie.gif"""
        images = []
        if(not os.path.exists(IMAGES_DIR)):
            print('\033[31m', "ERROR: Failed to image directory at", IMAGES_DIR, '\033[0m')
            os._exit(1) # Failure condition
        files = natural_sort(glob.glob(os.path.join(IMAGES_DIR, '*.png')))
        num_images = len(files)
        for i, filename in enumerate(files):
            if(self.params.verbose_printing):
                print("appending", filename)
            try:
                images.append(imageio.imread(filename))
            except:
                print(print_colors()["red"], 
                "Unable to read file:", filename, "Try clearing the directory of old files and rerunning", 
                print_colors()["reset"])
                exit(1)
            print("Movie progress:", i, "out of", num_images, "%.3f" % (i/num_images), "\r", end="")
        output_location = os.path.join(IMAGES_DIR, 'movie.gif')
        imageio.mimsave(output_location, images)
        print('\033[32m', "Rendered gif at", output_location, '\033[0m')
        # Clearing remaining files to not affect next render
        if clear_old_files:
            for f in files:
                os.remove(f)

    def plot_topview(self, ax, extent, traversible, human_traversible, camera_pos_13, 
                    agents, prerecs, robots, room_center, plot_quiver=False):
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

        # Plot the camera (robots)
        for i, r in enumerate(robots.values()):
            r_pos_3 = r.get_current_config().to_3D_numpy()
            r.get_trajectory().render(ax, freq=1, color=None, plot_quiver=False)
            color = 'bo' # robots are blue and solid unless collided
            if(r.get_collided()):
                color='ro' # collided robots are drawn red
            if i == 0:
                # only add label on first robot
                ax.plot(r_pos_3[0], r_pos_3[1], color, markersize=10, label='Robot')
            else:
                ax.plot(r_pos_3[0], r_pos_3[1], color, markersize=10)
            if np.array_equal(camera_pos_13, r_pos_3):
                # this is the "camera" robot (add quiver) 
                ax.quiver(camera_pos_13[0], camera_pos_13[1], np.cos(
                    camera_pos_13[2]), np.sin(camera_pos_13[2]))

        # plot all the simulated prerecorded agents
        for i, a in enumerate(prerecs.values()):
            pos_3 = a.get_current_config().to_3D_numpy()
            # pos_2 = a.get_current_config().position_nk2().numpy()[0][0]
            # heading= (a.get_current_config().heading_nk1().numpy())[0][0]
            # TODO: make colours of trajectories random rather than hardcoded
            a.get_trajectory().render(ax, freq=1, color=None, plot_quiver=False)
            color = 'yo' # agents are green and solid unless collided
            if(a.get_collided()):
                color='ro' # collided agents are drawn red
            if(i == 0):
                # Only add label on the first humans
                ax.plot(pos_3[0], pos_3[1],
                        color, markersize=10, label='Prerec')
            else:
                ax.plot(pos_3[0], pos_3[1], color, markersize=10)
            # TODO: use agent radius instead of hardcode
            ax.plot(pos_3[0], pos_3[1], color, alpha=0.2, markersize=25)
            if(plot_quiver):
                # Agent heading
                ax.quiver(pos_3[0], pos_3[1], np.cos(pos_3[2]), np.sin(pos_3[2]), 
                          scale=2, scale_units='inches')

        # plot all the randomly generated simulated agents
        for i, a in enumerate(agents.values()):
            pos_3 = a.get_current_config().to_3D_numpy()
            # pos_2 = a.get_current_config().position_nk2().numpy()[0][0]
            # heading= (a.get_current_config().heading_nk1().numpy())[0][0]
            # TODO: make colours of trajectories random rather than hardcoded
            a.get_trajectory().render(ax, freq=1, color=None, plot_quiver=False)
            color = 'go' # agents are green and solid unless collided
            if(a.get_collided()):
                color='ro' # collided agents are drawn red
            if(i == 0):
                # Only add label on the first humans
                ax.plot(pos_3[0], pos_3[1],
                        color, markersize=10, label='Agent')
            else:
                ax.plot(pos_3[0], pos_3[1], color, markersize=10)
            # TODO: use agent radius instead of hardcode
            ax.plot(pos_3[0], pos_3[1], color, alpha=0.2, markersize=25)
            if(plot_quiver):
                # Agent heading
                ax.quiver(pos_3[0], pos_3[1], np.cos(pos_3[2]), np.sin(pos_3[2]), 
                          scale=2, scale_units='inches')

        # plot other useful informational visuals in the topview
        # such as the key to the length of a "meter" unit
        plot_line_loc = room_center[:2] * 0.7
        start = [0, 0] + plot_line_loc
        end = [1, 0] + plot_line_loc
        gather_xs = [start[0], end[0]]
        gather_ys = [start[1], end[1]]
        col = 'k-'
        h = 0.1 # height of the "ticks" of the key
        ax.plot(gather_xs, gather_ys, col) # main line
        ax.plot([start[0], start[0]], [start[1] + h, start[1] - h], col) # tick left
        ax.plot([end[0], end[0]], [end[1] + h, end[1] - h], col) # tick right
        if(plot_quiver):
            ax.text(0.5*(start[0] + end[0]) - 0.2, start[1] + 0.5, "1m", fontsize=14,verticalalignment='top')

    def plot_images(self, p, rgb_image_1mk3, depth_image_1mk1, environment, room_center,
                    camera_pos_13, agents, prerecs, robots, sim_time, wall_time, filename, img_dir):

        map_scale = environment["map_scale"]
        # Obstacles/building traversible
        traversible = environment["traversibles"][0]
        human_traversible = None

        if len(environment["traversibles"]) > 1:
            assert(p.render_3D)
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
                    camera_pos_13, agents, prerecs, robots, room_center, plot_quiver=True)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        time_string = "sim_t=%.3f" % sim_time + " wall_t=%.3f" % wall_time
        ax.set_title(time_string, fontsize=20)

        # Render entire map-view from the top
        # to keep square plot
        outer_zoom = min(traversible.shape[0], traversible.shape[1]) * map_scale
        ax = fig.add_subplot(1, num_frames, 2)
        ax.set_xlim(0., outer_zoom)
        ax.set_ylim(0., outer_zoom)
        self.plot_topview(ax, extent, traversible, human_traversible,
                        camera_pos_13, agents, prerecs, robots, room_center)
        ax.legend()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(time_string, fontsize=20)

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

        dirname = 'tests/socnav/sim_movie' + str(img_dir)
        full_file_name = os.path.join(self.params.humanav_dir, dirname, filename)
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

    def take_snapshot(self, state, filename):
        """
        takes screenshot of a specific state of the world
        """
        # TODO: find a way to plot something in teh openGL 3D human view representing the robots
        # as right now they are invisible in all but the topview
        room_center = np.array([12., 17., 0.]) # to focus on in the zoomed image
        for i, r in enumerate(state.get_robots().values()):
            camera_pos_13 = r.get_current_config().to_3D_numpy()
            rgb_image_1mk3 = None
            depth_image_1mk1 = None
            # TODO: make the prerecs also generate a random human identity (probably human child)
            if self.params.render_3D:
                # only when rendering with opengl
                assert(len(state.get_environment()["traversibles"]) == 2) # environment holds building and human traversibles
                for a in state.get_agents().values():
                    self.r.update_human(a) 
                # update prerecorded humans
                for r_a in state.get_prerecs().values():
                    self.r.update_human(r_a) 
                # Update human traversible
                state.get_environment()["traversibles"][1] = self.r.get_human_traversible()
                # compute the rgb and depth images
                rgb_image_1mk3, depth_image_1mk1 = \
                    self.render_rgb_and_depth(self.r, np.array([camera_pos_13]), 
                                            state.get_environment()["map_scale"], human_visible=True)
            
                # TODO: Fix multiprocessing for properly deepcopied renderers 
            
            # plot the rbg, depth, and topview images if applicable
            self.plot_images(self.params, rgb_image_1mk3, depth_image_1mk1, 
                            state.get_environment(), room_center, camera_pos_13, 
                            state.get_agents(), state.get_prerecs(), state.get_robots(), 
                            state.get_sim_t(), state.get_wall_t(), "rob" + str(i) + filename, i)
        # Delete state to save memory after frames are generated
        del(state)
            


