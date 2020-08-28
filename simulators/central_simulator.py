import matplotlib as mpl
mpl.use('Agg')  # for rendering without a display
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import threading
import multiprocessing
from simulators.simulator_helper import SimulatorHelper
from simulators.sim_state import SimState, HumanState, AgentState
from params.central_params import get_path_to_socnav, create_sbpd_simulator_params
from utils.utils import *
from utils.image_utils import *


class CentralSimulator(SimulatorHelper):
    """The centralized simulator of TBD_SocNavBench """

    obstacle_map = None

    def __init__(self, environment: dict, renderer=None,
                 render_3D: bool = None, episode_name: str = "sim"):
        """ Initializer for the central simulator

        Args:
            params (Map): parameter configuration file from test_socnav.py
            environment (dict): dictionary housing the obj map (bitmap) and more
            renderer (optional): OpenGL renderer for 3D models. Defaults to None
            episode (str, optional): Name of the episode test that the simulator runs
        """
        self.r = renderer
        self.environment = environment
        self.params = create_sbpd_simulator_params(render_3D=render_3D)
        self.episode_name = episode_name
        CentralSimulator.obstacle_map = self._init_obstacle_map(renderer)
        # keep track of all agents in dictionary with names as the key
        self.agents = {}
        # keep track of all robots in dictionary with names as the key
        self.robots = {}
        # keep track of all prerecorded humans in a dictionary like the otherwise
        self.prerecs = {}
        # keep a single (important) robot as a value
        self.robot = None
        self.states = {}
        self.wall_clock_time: float = 0
        self.t: float = 0
        self.delta_t: float = 0  # will be updated in simulator based off dt

    def add_agent(self, a):
        """Adds an agent member to the central simulator's pool of gen_agents
        NOTE: this function works for robots (RoboAgent), prerecorded gen_agents (PrerecordedHuman),
              and general gen_agents (Agent)

        Args:
            a (Agent/PrerecordedAgent/RoboAgent): The agent to be added to the simulator
        """
        assert(CentralSimulator.obstacle_map is not None)
        name = a.get_name()
        from simulators.robot_agent import RoboAgent
        from humans.recorded_human import PrerecordedHuman
        if(isinstance(a, RoboAgent)):
            # initialize the robot and add to simulator's known "robot" field
            a.simulation_init(sim_map=CentralSimulator.obstacle_map,
                              with_planner=False)
            self.robots[name] = a
            self.robot = a
        elif isinstance(a, PrerecordedHuman):
            # generic agent initializer but without a planner (already have trajectories)
            a.simulation_init(sim_map=CentralSimulator.obstacle_map,
                              with_planner=False)
            self.prerecs[name] = a
        else:
            # initialize agent and add to simulator
            a.simulation_init(sim_map=CentralSimulator.obstacle_map,
                              with_planner=True)
            self.agents[name] = a

    def exists_running_agent(self):
        """Checks whether or not a generated agent is still running (acting)

        Returns:
            bool: True if there is at least one running agent, False otherwise
        """
        for a in self.agents.values():
            if (not a.end_acting):  # if there is even just a single agent acting
                return True
        return False

    def exists_running_prerec(self):
        """Checks whether or not a prerecorded agent is still running (acting)

        Returns:
            bool: True if there is at least one running prerec, False otherwise
        """
        for a in self.prerecs.values():
            if (not a.end_acting):  # if there is even just a single prerec acting
                return True
        return False

    def simulate(self):
        """ A function that simulates an entire episode. The gen_agents are updated with simultaneous
        threads running their update() functions and updating the robot with commands from the
        external joystick process.
        """
        num_agents: int = len(self.agents) + len(self.prerecs)
        print("Running simulation on", num_agents, "gen_agents")

        # scale the simulator time
        self.delta_t = self.params.delta_t_scale * self.params.dt

        # get initial state
        current_state = self.save_state(0, self.delta_t, 0)
        if(self.robot is None):
            print("%sNo robot in simulator%s" % (color_red, color_reset))
            return

        # give the robot knowledge of the initial world
        self.robot.repeat_joystick = not self.params.block_joystick
        self.robot.update_world(current_state)

        # initialize the robot to establish joystick connection
        r_t = self.init_robot_listener_thread()

        # keep track of wall-time in the simulator
        start_time = time.clock()

        # save initial state before the simulator is spawned
        self.t = 0.0
        if self.delta_t < self.params.dt:
            print("%sSimulation dt is too small; either lower the gen_agents' dt's" % (color_red),
                  self.params.dt, "or increase simulation delta_t%s" % (color_reset))
            exit(1)

        iteration = 1  # loop iteration
        self.print_sim_progress(iteration)

        while self.exists_running_agent() or self.exists_running_prerec():
            # update "wall clock" time
            wall_clock = time.clock() - start_time

            # Complete thread operations
            agent_threads = \
                self.init_agent_threads(self.t, self.delta_t, current_state)
            prerec_threads = self.init_prerec_threads(self.t, current_state)

            # start all thread groups
            self.start_threads(agent_threads)
            self.start_threads(prerec_threads)

            # calls a single iteration of the robot update
            self.robot.update(iteration)

            # join all thread groups
            self.join_threads(agent_threads)
            self.join_threads(prerec_threads)

            # capture time after all the gen_agents have updated
            # Takes screenshot of the new simulation state
            current_state = self.save_state(self.t, self.delta_t, wall_clock)
            self.robot.update_world(current_state)

            # update simulator time
            self.t += self.delta_t

            # update iteration count
            iteration += 1

            # print simulation progress
            self.print_sim_progress(iteration)

            if(iteration > self.params.max_frames):
                # hard limit of simulation
                break

        # free all the gen_agents
        for a in self.agents.values():
            del a

        # free all the prerecs
        for p in self.prerecs.values():
            del p

        self.decommission_robot(r_t)

        # capture final wall clock (completion) time
        wall_clock = time.clock() - start_time
        print("\nSimulation completed in", wall_clock, "seconds")

        # convert the saved states to rendered png's to be rendered into a movie
        self.generate_frames(filename=self.episode_name + "_obs")

        # convert all the generated frames into a gif file
        self.save_frames_to_gif(clear_old_files=True,
                                dir_title=self.episode_name)

    def _init_obstacle_map(self, renderer=None, ):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p, renderer,
                              res=self.environment["map_scale"] * 100,
                              trav=self.environment["traversibles"][0])

    def num_conditions_in_agents(self, condition: str):
        """Counts the number of termination causes matching condition in all the generated gen_agents

        Args:
            condition (str): either "green" (success), "red" (collision), or "blue" (timeout)

        Returns:
            num (int): number of termination causes matching condition
        """
        num: int = 0
        for a in self.agents.values():
            if(a.termination_cause is condition):
                num = num + 1
        return num

    def print_sim_progress(self, rendered_frames: int):
        """prints an inline simulation progress message based off agent planning termination
            TODO: account for agent<->agent collisions
        Args:
            rendered_frames (int): how many frames have been generated so far
        """
        print("A:", len(self.agents),
              "%sSuccess:" % (color_green),
              self.num_conditions_in_agents("green"),
              "%sCollide:" % (color_red),
              self.num_conditions_in_agents("red"),
              "%sTime:" % (color_blue),
              self.num_conditions_in_agents("blue"),
              "%sFrames:" % (color_reset),
              rendered_frames + 1,
              "T = %.3f" % (self.t),
              "\r", end="")

    def save_state(self, sim_t: float, delta_t: float, wall_t: float):
        """Captures the current state of the world to be saved to self.states

        Args:
            sim_t (float): the current time in the simulator in seconds
            wall_t (float): the current wall clock time

        Returns:
            current_state (SimState): the most recent state of the world
        """
        # NOTE: when using a modular environment, make saved_env a deepcopy
        saved_env = self.environment
        saved_agents = {}
        for a in self.agents.values():
            saved_agents[a.get_name()] = HumanState(a, deepcpy=True)
        # deepcopy all prerecorded gen_agents
        saved_prerecs = {}
        for a in self.prerecs.values():
            saved_prerecs[a.get_name()] = HumanState(a, deepcpy=True)
        # Save all the robots
        saved_robots = {}
        for r in self.robots.values():
            saved_robots[r.get_name()] = AgentState(r, deepcpy=True)
        current_state = SimState(saved_env,
                                 saved_agents, saved_prerecs, saved_robots,
                                 sim_t, wall_t, delta_t, self.episode_name
                                 )
        # Save current state to a class dictionary indexed by simulator time
        self.states[sim_t] = current_state
        return current_state

    def generate_frames(self, filename: str = "obs"):
        """Generates a png frame for each world state saved in self.states. Note, based off the
        render_3D options, the function will generate the frames in multiple separate processes to
        optimize performance on multicore machines, else it can also be done sequentially.
        NOTE: the 3D renderer can currently only be run sequentially

        Args:
            filename (str, optional): name of each png frame (unindexed). Defaults to "obs".
        """
        fps = (1.0 / self.delta_t) * self.params.fps_scale_down
        print("%sRendering movie with fps=%d%s" %
              (color_orange, fps, color_reset))
        num_frames = int(len(self.states) * self.params.fps_scale_down)
        np.set_printoptions(precision=3)
        if(not self.params.render_3D):
            # optimized to use multiple processes
            # TODO: put a limit on the maximum number of processes that can be run at once
            gif_processes = []
            skip = 0
            frame = 0
            for p, s in enumerate(self.states.values()):
                if(skip == 0):
                    # pool.apply_async(self.convert_state_to_frame, args=(s, filename + str(p) + ".png"))
                    gif_processes.append(multiprocessing.Process(
                        target=self.convert_state_to_frame,
                        args=(s, filename + str(p) + ".png"))
                    )
                    gif_processes[-1].start()
                    print("Started processes:", frame + 1,
                          "out of", num_frames, "\r", end="")
                    # reset skip counter for frames
                    skip = int(1.0 / self.params.fps_scale_down) - 1
                    frame += 1
                else:
                    # skip certain other frames as directed by the fps_scale_down
                    skip -= 1
            print("\n")
            for frame, p in enumerate(gif_processes):
                p.join()
                frame += 1
                print("Generated Frames:", frame, "out of", num_frames,
                      "%.3f" % (frame / num_frames), "\r", end="")
        else:
            # generate frames sequentially (non multiproceses)
            skip = 0
            for frame, s in enumerate(self.states.values()):
                if(skip == 0):
                    self.convert_state_to_frame(
                        s, filename + str(frame) + ".png")
                    frame += 1
                    print("Generated Frames:", frame, "out of", num_frames,
                          "%.3f" % (frame / num_frames), "\r", end="")
                    del(s)  # free the state from memory
                    skip = int(1.0 / self.params.fps_scale_down) - 1
                else:
                    skip -= 1.0

        # newline to not interfere with previous prints
        print("\n")

    def save_frames_to_gif(self, clear_old_files=True, with_multiprocessing=True, dir_title="sim"):
        """Convert a directory full of png's to a gif movie
        NOTE: One can also save to mp4 using imageio-ffmpeg or this bash script:
              "ffmpeg -r 10 -i simulate_obs%01d.png -vcodec mpeg4 -y movie.mp4"
        Args:
            clear_old_files (bool, optional): Whether or not to clear old image files. Defaults to True.
            with_multiprocessing (bool, optional): for multiple directories of images, run with multiprocessing. Defaults to True.
        """
        num_robots = len(self.robots)
        rendering_processes = []
        # fps = 1 / duration # where the duration is the simulation capture rate
        duration = self.delta_t * (1.0 / self.params.fps_scale_down)
        for i in range(num_robots):
            dirname = "tests/socnav/" + dir_title + "_movie"
            IMAGES_DIR = os.path.join(
                self.params.socnav_params.socnav_dir, dirname)
            if(with_multiprocessing):
                # little use to use pools here, since this is for multiple robot gen_agents in a scene
                # and the assumption here is that is a small number
                rendering_processes.append(multiprocessing.Process(
                    target=save_to_gif,
                    args=(IMAGES_DIR, duration, dir_title + "movie", clear_old_files))
                )
                rendering_processes[i].start()
            else:
                # sequentially
                save_to_gif(IMAGES_DIR, duration,
                            clear_old_files=clear_old_files)

        for p in rendering_processes:
            p.join()

    def plot_topview(self, ax, extent, traversible, human_traversible, camera_pos_13,
                     agents, prerecs, robots, room_center, plot_quiver=False):
        """Uses matplotlib to plot a birds-eye-view image of the world by plotting the environment
        and the gen_agents on every frame. The frame also includes the simulator time and wall clock time

        Args:
            ax (matplotlib.axes): the axes to plot on
            extent (np.array): the real_world extent (in meters) of the traversible
            traversible (np.array): the environment traversible (map bitmap)
            human_traversible (np.array/None): the human traversibles (or None for non-3D-plots)
            camera_pos_13 (np.array): the position of the camera (and robot)
            agents (AgentState dict): the agent states
            prerecs (HumanState dict): the prerecorded agent states
            robots (AgentState dict): the robots states
            room_center (np.array): the center of the "room" to focus the image plot off of
            plot_quiver (bool, optional): whether or not to plot the quiver (arrow). Defaults to False.
        """
        # get number of pixels per meter based off the ax plot space
        img_scale = ax.transData.transform(
            (0, 1)) - ax.transData.transform((0, 0))
        # number of pixels per "meter" unit in the plot
        ppm = int(img_scale[1] * (1.0 / self.params.img_scale))
        ax.imshow(traversible, extent=extent, cmap='gray',
                  vmin=-0.5, vmax=1.5, origin='lower')
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
            alphas = np.all(np.logical_not(human_traversible))

        # Plot the camera (robots)
        plot_agent_dict(ax, ppm, robots, label="Robot", normal_color="bo",
                        collided_color="ko", plot_quiver=plot_quiver, plot_start_goal=True)

        # plot all the simulated prerecorded gen_agents
        plot_agent_dict(ax, ppm, prerecs, label="Prerec", normal_color="yo",
                        collided_color="ro", plot_quiver=plot_quiver, plot_start_goal=False)

        # plot all the randomly generated simulated gen_agents
        plot_agent_dict(ax, ppm, agents, label="Agent", normal_color="go",
                        collided_color="ro", plot_quiver=plot_quiver, plot_start_goal=False)

        # plot other useful informational visuals in the topview
        # such as the key to the length of a "meter" unit
        plot_line_loc = room_center[:2] * 0.65
        start = [0, 0] + plot_line_loc
        end = [1, 0] + plot_line_loc
        gather_xs = [start[0], end[0]]
        gather_ys = [start[1], end[1]]
        col = 'k-'
        h = 0.1  # height of the "ticks" of the key
        ax.plot(gather_xs, gather_ys, col)  # main line
        ax.plot([start[0], start[0]], [start[1] +
                                       h, start[1] - h], col)  # tick left
        ax.plot([end[0], end[0]], [end[1] + h, end[1] - h], col)  # tick right
        if(plot_quiver):
            ax.text(0.5 * (start[0] + end[0]) - 0.2, start[1] +
                    0.5, "1m", fontsize=14, verticalalignment='top')

    def plot_images(self, p, rgb_image_1mk3, depth_image_1mk1, environment,
                    camera_pos_13, agents, prerecs, robots,
                    sim_t: float, wall_t: float, filename: str, img_dir: str, with_zoom=False):
        """Plots a single frame from information provided about the world state

        Args:
            p (Map): CentralSimulator params
            rgb_image_1mk3: the RGB image of the world_state to plot
            depth_image_1mk1: the depth-map image of the world_state to plot
            environment (dict): dictionary housing the obj map (bitmap) and more
            camera_pos_13 (np.array): the position of the camera (and robot)
            agents (AgentState dict): the agent states
            prerecs (HumanState dict): the prerecorded agent states
            robots (AgentState dict): the robots states
            sim_t (float): the simulator time in seconds
            wall_t (float): the wall clock time in seconds
            filename (str): the name of the file to save
            img_dir (str): the name of the directory to save the file
        """
        map_scale = environment["map_scale"]
        room_center = environment["room_center"]
        # Obstacles/building traversible
        traversible = environment["traversibles"][0]
        human_traversible = None

        if len(environment["traversibles"]) > 1:
            assert(p.render_3D)
            human_traversible = environment["traversibles"][1]
        # Compute the real_world extent (in meters) of the traversible
        extent = [0., traversible.shape[1], 0., traversible.shape[0]]
        extent = np.array(extent) * map_scale

        # count used to signify the number of images that will be generated in a single frame
        # default 1, for normal view (can add a second for zoomed view)
        plot_count = 1
        if(with_zoom):
            plot_count += 1
            zoomed_img_plt_indx = plot_count  # 2
        if rgb_image_1mk3 is not None:
            plot_count = plot_count + 1
            rgb_img_plt_indx = plot_count  # 3
        if depth_image_1mk1 is not None:
            plot_count = plot_count + 1
            depth_img_plt_indx = plot_count  # 4

        img_size = 10 * p.img_scale
        fig = plt.figure(figsize=(plot_count * img_size, img_size))
        ax = fig.add_subplot(1, plot_count, 1)
        ax.set_xlim(0., traversible.shape[1] * map_scale)
        ax.set_ylim(0., traversible.shape[0] * map_scale)
        self.plot_topview(ax, extent, traversible, human_traversible,
                          camera_pos_13, agents, prerecs, robots, room_center, plot_quiver=True)
        ax.legend()
        time_string = "sim_t=%.3f" % sim_t + " wall_t=%.3f" % wall_t
        ax.set_title(time_string, fontsize=20)

        if(with_zoom):
            # Plot the 5x5 meter occupancy grid centered around the camera
            zoom = 8.5  # zoom out in by a constant amount
            ax = fig.add_subplot(1, plot_count, zoomed_img_plt_indx)
            ax.set_xlim([room_center[0] - zoom, room_center[0] + zoom])
            ax.set_ylim([room_center[1] - zoom, room_center[1] + zoom])
            self.plot_topview(ax, extent, traversible, human_traversible,
                              camera_pos_13, agents, prerecs, robots, room_center, plot_quiver=True)
            ax.legend()
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(time_string, fontsize=20)

        if rgb_image_1mk3 is not None:
            # Plot the RGB Image
            ax = fig.add_subplot(1, plot_count, rgb_img_plt_indx)
            ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('RGB')

        if depth_image_1mk1 is not None:
            # Plot the Depth Image
            ax = fig.add_subplot(1, plot_count, depth_img_plt_indx)
            ax.imshow(depth_image_1mk1[0, :, :, 0].astype(
                np.uint8), cmap='gray')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title('Depth')

        full_file_name = \
            os.path.join(p.socnav_params.socnav_dir, img_dir, filename)
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
            print('\033[32m', "Successfully rendered:",
                  full_file_name, '\033[0m')

    def convert_state_to_frame(self, state: SimState, filename: str):
        """Converts a state into an image to be later converted to a gif movie

        Args:
            state (SimState): the state of the world to convert to an image
            filename (str): the name of the resulting image (unindexed)
        """

        camera_pos_13 = self.robot.get_current_config().to_3D_numpy()
        rgb_image_1mk3 = None
        depth_image_1mk1 = None
        # NOTE: 3d renderer can only be used with sequential plotting, much slower
        if self.params.render_3D:
            # TODO: Fix multiprocessing for properly deepcopied renderers
            # only when rendering with opengl
            assert(len(state.get_environment()["traversibles"]) == 2)
            for a in state.get_gen_agents().values():
                self.r.update_human(a)
            # update prerecorded humans
            for r_a in state.get_prerecs().values():
                self.r.update_human(r_a)
            # Update human traversible
            state.get_environment()["traversibles"][1] = \
                self.r.get_human_traversible()
            # compute the rgb and depth images
            rgb_image_1mk3, depth_image_1mk1 = \
                render_rgb_and_depth(self.r, np.array([camera_pos_13]),
                                     state.get_environment()["map_scale"],
                                     human_visible=True)
        # plot the rbg, depth, and topview images if applicable
        self.plot_images(self.params, rgb_image_1mk3, depth_image_1mk1,
                         state.get_environment(), camera_pos_13,
                         state.get_gen_agents(), state.get_prerecs(), state.get_robots(),
                         state.get_sim_t(), state.get_wall_t(), self.episode_name + filename,
                         'tests/socnav/' + self.episode_name + '_movie')
        # Delete state to save memory after frames are generated
        del(state)

    """BEGIN thread utils"""

    def init_robot_listener_thread(self, power_on=True):
        """Initializes the robot listener by establishing socket connections to 
        the joystick, transmitting the (constant) obstacle map (environment), 
        and starting the robot thread.

        Args:
            power_on (bool, optional): Whether or not the robot should start on. Defaults to True.

        Returns:
            Thread: The robot's update thread if it exists in the simulator, else None
        """
        # wait for joystick connection to be established
        r = self.robot
        if(r is not None):
            r.establish_joystick_receiver_connection()
            time.sleep(0.01)
            r.establish_joystick_sender_connection()
            assert(r.world_state is not None)
            # send first transaction to the joystick
            print("sending map to joystick")
            r.send_to_joystick(r.world_state.to_json(include_map=True))
            r_listener_thread = threading.Thread(target=r.listen_to_joystick)
            if(power_on):
                r_listener_thread.start()
            # wait until joystick is ready
            while(not r.joystick_ready):
                # wait until joystick recieves the environment (once)
                time.sleep(0.01)
            print("Robot powering on")
            return r_listener_thread
        print("%sNo robot in simulator%s" % (color_red, color_reset))
        return None

    def decommission_robot(self, r_listener_thread):
        """Turns off the robot and joins the robot's update thread

        Args:
            r_listener_thread (Thread): the robot update thread to join
        """
        if(r_listener_thread is not None):
            assert(self.robot is not None)
            # turn off the robot
            self.robot.power_off()
            # close robot listener threads
            if(r_listener_thread.is_alive()):
                r_listener_thread.join()
            del(r_listener_thread)
        return

    def init_agent_threads(self, sim_t: float, t_step: float, current_state: SimState):
        """Spawns a new agent thread for each agent (running or finished)

        Args:
            sim_t (float): the simulator time in seconds
            t_step (float): the time step (update frequency) of the simulator
            current_state (SimState): the most recent state of the world

        Returns:
            agent_threads (list): list of all spawned (not started) agent threads
        """
        agent_threads = []
        for a in self.agents.values():
            agent_threads.append(threading.Thread(
                target=a.update, args=(sim_t, t_step, current_state,)))
        return agent_threads

    def init_prerec_threads(self, sim_t: float, current_state: SimState):
        """Spawns a new prerec thread for each running prerecorded agent

        Args:
            sim_t (float): the simulator time in seconds
            current_state (SimState): the current state of the world

        Returns:
            prerec_threads (list): list of all spawned (not started) prerecorded agent threads
        """
        prerec_threads = []
        prerec_agents = list(self.prerecs.values())
        for a in prerec_agents:
            if(not a.end_acting):
                prerec_threads.append(threading.Thread(
                    target=a.update, args=(sim_t, current_state,)))
            else:
                # remove the agent once it's completed its trajectory
                self.prerecs.pop(a.get_name())
                del(a)
        return prerec_threads

    def start_threads(self, thread_group):
        """Starts a group of threads at once

        Args:
            thread_group (list): a group of threads to be started
        """
        for t in thread_group:
            t.start()

    def join_threads(self, thread_group):
        """Joins a group of threads at once

        Args:
            thread_group (list): a group of threads to be joined
        """
        for t in thread_group:
            t.join()
            del(t)

    """END thread utils"""
