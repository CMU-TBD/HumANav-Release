import numpy as np
import os
import time
import threading
import multiprocessing
from simulators.simulator_helper import SimulatorHelper
from simulators.sim_state import SimState, HumanState, AgentState
from params.central_params import create_simulator_params, get_seed
from utils.utils import *
from utils.image_utils import *


class CentralSimulator(SimulatorHelper):
    """The centralized simulator of TBD_SocNavBench """

    obstacle_map = None

    def __init__(self, environment: dict, renderer=None,
                 render_3D: bool = None, episode_params=None):
        """ Initializer for the central simulator

        Args:
            params (Map): parameter configuration file from test_socnav.py
            environment (dict): dictionary housing the obj map (bitmap) and more
            renderer (optional): OpenGL renderer for 3D models. Defaults to None
            episode_params (str, optional): Name of the episode test that the simulator runs
        """
        self.r = renderer
        self.environment = environment
        self.params = create_simulator_params(render_3D=render_3D)
        self.episode_params = episode_params
        # name of the directory to output everything
        self.params.output_directory = \
            os.path.join(self.params.socnav_params.socnav_dir,
                         "tests/socnav/" + episode_params.name + "_output")
        CentralSimulator.obstacle_map = self._init_obstacle_map(renderer)
        # keep track of all agents in dictionary with names as the key
        self.agents = {}
        # keep track of all robots in dictionary with names as the key
        self.robots = {}
        # keep track of all prerecorded humans in a dictionary like the otherwise
        self.backstage_prerecs = {}
        self.prerecs = {}
        # keep a single (important) robot as a value
        self.robot = None
        self.states = {}
        self.wall_clock_time: float = 0
        self.t: float = 0
        self.delta_t: float = 0  # will be updated in simulator based off dt

    def add_agent(self, a):
        """Adds an agent member to the central simulator's pool of gen_agents
        NOTE: this function works for robots (RobotAgent), prerecorded gen_agents (PrerecordedHuman),
              and general gen_agents (Agent)

        Args:
            a (Agent/PrerecordedAgent/RobotAgent): The agent to be added to the simulator
        """
        assert(CentralSimulator.obstacle_map is not None)
        name = a.get_name()
        from simulators.robot_agent import RobotAgent
        from humans.recorded_human import PrerecordedHuman
        if isinstance(a, RobotAgent):
            # initialize the robot and add to simulator's known "robot" field
            a.simulation_init(sim_map=CentralSimulator.obstacle_map,
                              with_planner=False)
            self.robots[name] = a
            self.robot = a
        elif isinstance(a, PrerecordedHuman):
            # generic agent initializer but without a planner (already have trajectories)
            a.simulation_init(sim_map=CentralSimulator.obstacle_map,
                              with_planner=False)
            # added to backstage prerecs which will add to self.prerecs when the time is right
            self.backstage_prerecs[name] = a
        else:
            # initialize agent and add to simulator
            a.simulation_init(sim_map=CentralSimulator.obstacle_map,
                              with_planner=True)
            self.agents[name] = a

    def init_sim_data(self):
        # Create pre-simulation metadata
        self.total_agents: int = len(self.agents) + len(self.backstage_prerecs)
        print("Running simulation on", self.total_agents, "agents")
        self.num_collided_agents = 0
        self.num_collided_prerecs = 0
        self.num_completed_agents = 0
        self.num_completed_prerecs = 0
        # scale the simulator time
        self.delta_t = self.params.delta_t_scale * self.params.dt
        if(self.robot):
            self.robot.set_sim_delta_t(self.delta_t)

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

    def loop_condition(self):
        if(self.robot):
            # run for the full time if the robot exists
            return self.t <= self.episode_params.max_time
        # else just run until there are no more agents
        return self.exists_running_agent() or self.exists_running_prerec()

    def simulate(self):
        """ A function that simulates an entire episode. The gen_agents are updated with simultaneous
        threads running their update() functions and updating the robot with commands from the
        external joystick process.
        """
        # initialize pre-simulation metadata
        self.init_sim_data()

        # add the first (when t=0) agents to the self.prerecs dict
        self.init_prerec_threads(sim_t=0, current_state=None)

        # get initial state
        current_state = self.save_state(0, self.delta_t, 0)
        if self.robot is None:
            print("%sNo robot in simulator%s" % (color_red, color_reset))
        else:
            # give the robot knowledge of the initial world
            self.robot.repeat_joystick = not self.params.block_joystick
            self.robot.update_world(current_state)

            # initialize the robot to establish joystick connection
            r_t = self.init_robot_listener_thread()

        # keep track of wall-time in the simulator
        start_time = time.time()

        # save initial state before the simulator is spawned
        self.t = 0.0
        if self.delta_t < self.params.dt:
            print("%sSimulation dt is too small; either lower the gen_agents' dt's" % (color_red),
                  self.params.dt, "or increase simulation delta_t%s" % (color_reset))
            exit(1)

        iteration = 0  # loop iteration
        self.print_sim_progress(iteration)

        while self.loop_condition():
            wall_t = time.time() - start_time

            # Complete thread operations
            agent_threads = \
                self.init_agent_threads(self.t, self.delta_t, current_state)
            prerec_threads = self.init_prerec_threads(self.t, current_state)

            # start all thread groups
            self.start_threads(agent_threads)
            self.start_threads(prerec_threads)

            if(self.robot):
                # calls a single iteration of the robot update
                self.robot.update(iteration)

            # join all thread groups
            self.join_threads(agent_threads)
            self.join_threads(prerec_threads)

            # update simulator time
            self.t += self.delta_t

            # capture time after all the gen_agents have updated
            # Takes screenshot of the new simulation state
            current_state = self.save_state(self.t, self.delta_t, wall_t)

            if(self.robot):
                self.robot.update_world(current_state)

            # update iteration count
            iteration += 1

            # print simulation progress
            self.print_sim_progress(iteration)

        if(self.robot):
            self.robot.power_off()

        # free all the gen_agents
        for a in self.agents.values():
            a = None
            del(a)

        # free all the prerecs
        for p in self.prerecs.values():
            p = None
            del(p)

        # capture final wall clock (completion) time
        self.sim_wall_clock = time.time() - start_time
        print("\nSimulation completed in",
              self.sim_wall_clock, "real world seconds")

        if(self.episode_params.write_episode_log):
            self.generate_sim_log()

        # convert the saved states to rendered png's to be rendered into a movie
        self.generate_frames(filename=self.episode_params.name + "_obs")

        # convert all the generated frames into a gif file
        self.save_frames_to_gif(clear_old_files=True)

        if(self.robot):
            # finally close the robot listener thread
            self.decommission_robot(r_t)

    def _init_obstacle_map(self, renderer=None, ):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p, renderer,
                              res=self.environment["map_scale"] * 100,
                              map_trav=self.environment["map_traversible"])

    def print_sim_progress(self, rendered_frames: int):
        """prints an inline simulation progress message based off agent planning termination
            TODO: account for agent<->agent collisions
        Args:
            rendered_frames (int): how many frames have been generated so far
        """
        num_completed_agents = self.num_completed_agents + self.num_completed_prerecs
        num_collided_agents = self.num_collided_agents + self.num_collided_prerecs
        num_timeout = self.total_agents - num_completed_agents - num_collided_agents
        print("A:", self.total_agents,
              "%sSuccess:" % (color_green), num_completed_agents,
              "%sCollide:" % (color_red), num_collided_agents,
              "%sTime:" % (color_blue), num_timeout,
              "%sFrames:" % (color_reset), rendered_frames,
              "T = %.3f" % (self.t),
              "\r", end="")

    def save_state(self, sim_t: float, delta_t: float, wall_t: float):
        """Captures the current state of the world to be saved to self.states

        Args:
            sim_t (float): the current time in the simulator in seconds
            delta_t (float): the timestep size in the simulator in seconds
            wall_t (float): the current wall clock time

        Returns:
            current_state (SimState): the most recent state of the world
        """
        # NOTE: when using a modular environment, make saved_env a deepcopy
        saved_env = self.environment
        pedestrians = {}
        for a in self.agents.values():
            pedestrians[a.get_name()] = HumanState(a, deepcpy=True)
        # deepcopy all prerecorded gen_agents
        for a in self.prerecs.values():
            pedestrians[a.get_name()] = HumanState(a, deepcpy=True)
        # Save all the robots
        saved_robots = {}
        for r in self.robots.values():
            saved_robots[r.get_name()] = AgentState(r, deepcpy=True)
        current_state = SimState(saved_env, pedestrians, saved_robots,
                                 sim_t, wall_t, delta_t, self.episode_params.name,
                                 self.episode_params.max_time)

        # Save current state to a class dictionary indexed by simulator time
        self.states[sim_t] = current_state
        return current_state

    """ BEGIN IMAGE UTILS """

    def generate_frames(self, filename: str = "obs"):
        """Generates a png frame for each world state saved in self.states. Note, based off the
        render_3D options, the function will generate the frames in multiple separate processes to
        optimize performance on multicore machines, else it can also be done sequentially.
        NOTE: the 3D renderer can currently only be run sequentially

        Args:
            filename (str, optional): name of each png frame (unindexed). Defaults to "obs".
        """
        if(self.params.fps_scale_down == 0):
            print("%sNot rendering movie%s" %
                  (color_orange, color_reset))
            return
        fps = (1.0 / self.delta_t) * self.params.fps_scale_down
        print("%sRendering movie with fps=%d%s" %
              (color_orange, fps, color_reset))
        num_frames = \
            int(np.ceil(len(self.states) * self.params.fps_scale_down))
        np.set_printoptions(precision=3)
        if(not self.params.render_3D):
            # optimized to use multiple processes
            # TODO: put a limit on the maximum number of processes that can be run at once
            gif_processes = []
            skip = 0
            frame = 0
            for p, s in enumerate(self.states.values()):
                if(skip == 0):
                    # pool.apply_async(self.render_sim_state, args=(s, filename + str(p) + ".png"))
                    frame += 1
                    gif_processes.append(multiprocessing.Process(
                        target=self.render_sim_state,
                        args=(s, filename + str(p) + ".png"))
                    )
                    gif_processes[-1].start()
                    print("Started processes: %d out of %d, %.3f%% \r" %
                          (frame, num_frames, 100.0 * (frame / num_frames)), end="")
                    # reset skip counter for frames
                    skip = int(1.0 / self.params.fps_scale_down) - 1
                else:
                    # skip certain other frames as directed by the fps_scale_down
                    skip -= 1
            print()  # not overwrite next line
            for frame, p in enumerate(gif_processes):
                p.join()
                print("Finished processes: %d out of %d, %.3f%% \r" %
                      (frame + 1, num_frames, 100.0 * ((frame + 1) / num_frames)), end="")
            print()  # not overwrite next line
        else:
            # generate frames sequentially (non multiproceses)
            skip = 0
            frame = 0
            for s in self.states.values():
                if(skip == 0):
                    self.render_sim_state(s, filename + str(frame) + ".png")
                    frame += 1
                    print("Generated Frames:", frame, "out of", num_frames,
                          "%.3f" % (frame / num_frames), "\r", end="")
                    del(s)  # free the state from memory
                    skip = int(1.0 / self.params.fps_scale_down) - 1
                else:
                    skip -= 1.0

    def save_frames_to_gif(self, clear_old_files=True):
        """Convert a directory full of png's to a gif movie
        NOTE: One can also save to mp4 using imageio-ffmpeg or this bash script:
              "ffmpeg -r 10 -i simulate_obs%01d.png -vcodec mpeg4 -y movie.mp4"
        Args:
            clear_old_files (bool, optional): Whether or not to clear old image files. Defaults to True.
        """
        if self.params.fps_scale_down == 0:
            return
        # fps = 1 / duration # where the duration is the simulation capture rate
        duration = self.delta_t * (1.0 / self.params.fps_scale_down)
        # sequentially
        save_to_gif(self.params.output_directory, duration, gif_filename="movie_%d" % (get_seed()),
                    clear_old_files=clear_old_files)

    def render_sim_state(self, state: SimState, filename: str):
        """Converts a state into an image to be later converted to a gif movie

        Args:
            state (SimState): the state of the world to convert to an image
            filename (str): the name of the resulting image (unindexed)
        """
        if(self.robot):
            robot = list(state.get_robots().values())[0]
            camera_pos_13 = robot.get_current_config().to_3D_numpy()
        else:
            robot = None
            camera_pos_13 = state.get_environment()["room_center"]
        rgb_image_1mk3 = None
        depth_image_1mk1 = None
        # NOTE: 3d renderer can only be used with sequential plotting, much slower
        if self.params.render_3D:
            # TODO: Fix multiprocessing for properly deepcopied renderers
            # only when rendering with opengl
            assert("human_traversible" in state.get_environment().keys())
            # update pedestrians humans
            for a in state.get_pedestrians().values():
                self.r.update_human(a)
            # Update human traversible
            # NOTE: this is technically not R-O since it modifies the human trav
            # TODO: use a separate variable to keep SimStates as R-O
            state.get_environment()["human_traversible"] = \
                self.r.get_human_traversible()
            # compute the rgb and depth images
            rgb_image_1mk3, depth_image_1mk1 = \
                render_rgb_and_depth(self.r, np.array([camera_pos_13]),
                                     state.get_environment()["map_scale"],
                                     human_visible=True)
        # plot the rbg, depth, and topview images if applicable
        render_scene(self.params, rgb_image_1mk3, depth_image_1mk1,
                     state.get_environment(), camera_pos_13,
                     state.get_pedestrians(), state.get_robots(),
                     state.get_sim_t(), state.get_wall_t(), filename)
        # Delete state to save memory after frames are generated
        del(state)

    """ END IMAGE UTILS """

    def generate_sim_log(self, filename='episode_log.txt'):
        import io
        abs_filename = os.path.join(self.params.output_directory, filename)
        touch(abs_filename)  # create if dosent already exist
        ep_params = self.episode_params
        data = ""
        data += "****************EPISODE INFO****************\n"
        data += "Episode name: %s\n" % ep_params.name
        data += "Building name: %s\n" % ep_params.map_name
        data += "Robot start: %s\n" % str(ep_params.robot_start_goal[0])
        data += "Robot goal: %s\n" % str(ep_params.robot_start_goal[1])
        data += "Time budget: %.3f\n" % ep_params.max_time
        # data += "Prerec start indx: %d\n" % ep_params.prerec_start_indx
        data += "Total agents in scene: %d\n" % self.total_agents
        data += "****************SIMULATOR INFO****************\n"
        data += "Simulator refresh rate (s): %0.3f\n" % self.delta_t
        data += "Total duration of simulation (s): %0.3f\n" % self.sim_wall_clock
        num_successful = self.num_completed_agents + self.num_completed_prerecs
        data += "Num Successful agents: %d\n" % num_successful
        num_collision = self.num_collided_agents + self.num_collided_prerecs
        data += "Num Collided agents: %d\n" % num_collision
        num_timeout = self.total_agents - (num_successful + num_collision)
        data += "Num Timeout agents: %d\n" % num_timeout
        if(self.robot):
            data += "****************ROBOT INFO****************\n"
            data += "Robot termination cause: %s\n" % self.robot.termination_cause
            if(self.robot.termination_cause == "Collision"):
                data += "Robot collided with agent: %s\n" % self.robot.collider
            data += "Num commands received from joystick: %d\n" % \
                len(self.robot.joystick_inputs)
            data += "Total time blocking for joystick input (s): %0.3f\n" % \
                self.robot.get_block_t_total()
            data += "Num commands executed by robot: %d\n" % self.robot.num_executed
            rob_displacement = euclidean_dist2(ep_params.robot_start_goal[0],
                                               self.robot.get_current_config().to_3D_numpy())
            data += "Robot displacement (m): %0.3f\n" % rob_displacement
            data += "Max robot velocity (m/s): %0.3f\n" % \
                absmax(self.robot.vehicle_trajectory.speed_nk1())
            data += "Max robot acceleration: %0.3f\n" % \
                absmax(self.robot.vehicle_trajectory.acceleration_nk1())
            data += "Max robot angular velocity: %0.3f\n" % \
                absmax(self.robot.vehicle_trajectory.angular_speed_nk1())
            data += "Max robot angular acceleration: %0.3f\n" % \
                absmax(self.robot.vehicle_trajectory.angular_acceleration_nk1())
        try:
            with open(abs_filename, 'w') as f:
                f.write(data)
                f.close()
            print("%sSuccessfully wrote episode log to %s%s" %
                  (color_green, filename, color_reset))
        except:
            print("%sWriting episode log failed%s" % (color_red, color_reset))

    """ BEGIN THREAD UTILS """

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
            assert(r.world_state is not None)
            # send first transaction to the joystick
            print("sending episode data to joystick")
            r.simulator_running = True
            r.send_to_joystick(r.world_state.to_json(send_metadata=True))
            r_listener_thread = threading.Thread(target=r.listen_to_joystick)
            if(power_on):
                r_listener_thread.start()
            # wait until joystick is ready
            while(not r.joystick_ready):
                # wait until joystick receives the environment (once)
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
        if(self.robot):
            if(r_listener_thread):
                # turn off the robot
                self.robot.power_off()
                self.robot.simulator_running = False
                # close robot listener threads
                if(r_listener_thread.is_alive() and self.params.join_threads):
                    # TODO: connect to the socket and close it
                    import socket
                    from simulators.robot_agent import RobotAgent
                    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
                        (RobotAgent.host, RobotAgent.port_recv))
                    r_listener_thread.join()
                del(r_listener_thread)

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
        all_agents = list(self.agents.values())
        for a in all_agents:
            if(not a.end_acting):
                agent_threads.append(threading.Thread(
                    target=a.update, args=(sim_t, t_step, current_state,)))
            else:
                if(a.termination_cause == "Success"):
                    self.num_completed_agents += 1
                else:
                    self.num_collided_agents += 1
                self.agents.pop(a.get_name())
                del(a)
        return agent_threads

    def init_prerec_threads(self, sim_t: float, current_state: SimState = None):
        """Spawns a new prerec thread for each running prerecorded agent

        Args:
            sim_t (float): the simulator time in seconds
            current_state (SimState): the current state of the world

        Returns:
            prerec_threads (list): list of all spawned (not started) prerecorded agent threads
        """
        prerec_threads = []
        all_prerec_agents = list(self.backstage_prerecs.values())
        for a in all_prerec_agents:
            if(not a.end_acting and a.get_start_time() <= sim_t < a.get_end_time()):
                # only add (or keep) agents in the time frame
                self.prerecs[a.get_name()] = a
                if(current_state):  # not None
                    prerec_threads.append(threading.Thread(
                        target=a.update, args=(sim_t, current_state,)))
            else:
                # remove agent since its not within the time frame or finished
                if(a.get_name() in self.prerecs.keys()):
                    if(a.get_collided()):
                        self.num_collided_prerecs += 1
                    else:
                        self.num_completed_prerecs += 1
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

    """ END THREAD UTILS """
