import socket
import threading
import multiprocessing
import threading
import time
import sys
import json
import os
from copy import deepcopy
from random import randint
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # for rendering without a display
import matplotlib.pyplot as plt
from utils.utils import *
from params.robot_params import create_params
from params.renderer_params import get_path_to_humanav, get_seed
from params.simulator.sbpd_simulator_params import create_params as create_agent_params
from simulators.agent import Agent
from trajectory.trajectory import Trajectory


# seed the random number generator
random.seed(get_seed())


class Joystick():
    def __init__(self):
        self.t = 0
        self.latest_state = None
        self.sim_states = []
        self.velocities = {}     # testing simstate utils
        self.accelerations = {}  # testing simstate utils
        self.environment = None
        self.joystick_params = create_params()
        # sockets for communication
        self.robot_sender_socket = None
        self.robot_running = False
        self.host = socket.gethostname()
        # port for sending commands to the robot
        self.port_send = self.joystick_params.port
        self.port_recv = self.port_send + 1  # port for recieving commands from the robot
        self.frame_num = 0
        self.request_world = False  # True whenever the joystick wants data about the world
        print("Initiated joystick at", self.host, self.port_send)
        self.start_config = None
        self.goal_config = None
        self.current_config = None
        # planned controls
        self.commanded_actions = []
        self.num_sent = 0
        self.lin_vels = []
        self.ang_vels = []
        self.delta_t = None

    def set_host(self, h):
        self.host = h

    def _init_obstacle_map(self, renderer=0):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p, renderer,
                              res=float(self.environment["map_scale"]) * 100.,
                              trav=np.array(
                                  self.environment["traversibles"][0])
                              )

    def init_control_pipeline(self):
        assert(self.sim_states is not None)
        assert(self.environment is not None)
        self.params = create_agent_params()
        self.params.dt = 0.05
        self.params.control_horizon = 200  # based off central_simulator's parse params
        # self.environment["traversibles"][0]
        self.obstacle_map = self._init_obstacle_map()
        self.obj_fn = Agent._init_obj_fn(self)
        # Initialize Fast-Marching-Method map for agent's pathfinding
        self.fmm_map = Agent._init_fmm_map(self)
        Agent._update_fmm_map(self)
        # Initialize system dynamics and planner fields
        self.planner = Agent._init_planner(self)
        self.vehicle_data = self.planner.empty_data_dict()
        self.system_dynamics = Agent._init_system_dynamics(self)
        self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)

    def create_message(self, joystick_power: bool, lin_vels: list, ang_vels: list,
                       j_time: float = 0.0, req_world: bool = False):
        json_dict = {}
        json_dict["joystick_on"] = joystick_power
        if(joystick_power):
            json_dict["j_time"] = j_time
            json_dict["lin_vels"] = lin_vels
            json_dict["ang_vels"] = ang_vels
            json_dict["req_world"] = req_world
        return json.dumps(json_dict, indent=1)

    def robot_input(self, lin_commands: list, ang_commands: list,
                    request_world: bool, override_power_off: bool = False):
        # singular input
        message = self.create_message(self.robot_running or override_power_off,
                                      lin_commands, ang_commands, time.clock(), request_world)
        if(request_world):
            # only a single message being sent
            self.request_world = False
        self.send_to_robot(message)

    def random_robot_joystick(self, action_dt: int):
        self.robot_running = True
        assert(self.environment is not None)
        while(self.robot_running is True):
            try:
                lin_vels = []
                ang_vels = []
                lin = randint(10, 100) / 100.
                ang = randint(-100, 100) / 100.
                for i in range(action_dt):
                    lin_vels.append(lin)
                    ang_vels.append(ang)
                self.robot_input(lin_vels, ang_vels, self.request_world)
                time.sleep(0.5)  # NOTE: Tune this to whatever you'd like
                # now update the robot with the "ready" ping
            except KeyboardInterrupt:
                print("%sJoystick disconnected by user%s" %
                      (color_yellow, color_reset))
                self.power_off()
                break

    def send_robot_group(self, freq):
        while(self.robot_running):
            if(self.num_sent < len(self.commanded_actions)):
                command = self.commanded_actions[self.num_sent]
                lin = command[0]
                ang = command[1]
                if(lin != 0 and ang != 0):
                    self.lin_vels.append(float(lin))
                    self.ang_vels.append(float(ang))
                    if(len(self.lin_vels) >= freq):
                        self.robot_input(deepcopy(self.lin_vels),
                                         deepcopy(self.ang_vels), self.request_world)
                        # reset the containers
                        self.lin_vels = []
                        self.ang_vels = []
                        # NOTE: this robot sender delay is tunable to ones liking
                        time.sleep(0.1)  # planner delay
                    if(self.num_sent % 20 == 0):
                        self.request_world = True
                    self.num_sent += 1
            else:
                # wait until a new command is added
                time.sleep(0.001)

    def planned_robot_joystick(self):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data"""
        while(self.current_config is None):
            # wait until robot's current position is known
            time.sleep(0.01)
        self.planned_next_config = copy.deepcopy(self.current_config)
        while(self.robot_running):
            self.planner_data = self.planner.optimize(
                self.planned_next_config, self.goal_config)
            # LQR feedback control loop
            t_seg = Trajectory.new_traj_clip_along_time_axis(self.planner_data['trajectory'],
                                                             self.params.control_horizon,
                                                             repeat_second_to_last_speed=True)
            _, commanded_actions_nkf = self.system_dynamics.parse_trajectory(
                t_seg)
            # NOTE: the format for the velocity commands to the open loop for the robot is:
            # np.array([[[L, A]]], dtype=np.float32) where L is linear, A is angular
            self.planned_next_config = \
                SystemConfig.init_config_from_trajectory_time_index(
                    t_seg,
                    t=-1
                )
            self.vehicle_trajectory.append_along_time_axis(t_seg)
            self.commanded_actions.extend(commanded_actions_nkf[0])
            # print(self.planner_data['optimal_control_nk2'])
            # TODO: match the action_dt with the number of signals sent to the robot at once
            self.current_config = \
                SystemConfig.init_config_from_trajectory_time_index(
                    self.vehicle_trajectory, t=-1)

    def force_close_socket(self):
        # connect to the socket, closing it, and continuing the thread to completion
        try:
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(
                (self.host, self.port_recv))
        except:
            print("%sForce closing socket%s" %
                  (color_red, color_reset))
        self.robot_receiver_socket.close()

    def update(self, random_commands: bool = False):
        """ Independent process for a user (at a designated host:port) to receive
        information from the simulation while also sending commands to the robot """
        while(self.delta_t is None):
            time.sleep(0.01)
        action_dt = int(np.floor(self.delta_t / self.params.dt))
        print("simulator's refresh rate =", self.delta_t)
        print("joystick's refresh rate  =", self.params.dt)
        sender_thread = threading.Thread(
            target=self.send_robot_group,
            args=(action_dt,)
        )
        sender_thread.start()
        if(random_commands):
            self.random_robot_joystick(action_dt)
        else:
            self.planned_robot_joystick()
        # this point is reached once the planner/randomizer are finished
        self.force_close_socket()
        if(self.listen_thread.is_alive()):
            self.listen_thread.join()
        # begin gif (movie) generation
        try:
            save_to_gif(os.path.join(get_path_to_humanav(), self.dirname))
        except:
            print("unable to render gif")

    def power_off(self):
        if(self.robot_running):
            print("%sConnection closed by robot%s" % (color_red, color_reset))
            self.robot_running = False
            self.force_close_socket()
            try:
                # send one last command to the robot with indication that self.robot_running=False
                self.robot_input([], [], False, override_power_off=True)
            except:
                pass

    """BEGIN socket utils"""

    def send_to_robot(self, json_message: str):
        # Create a TCP/IP socket
        self.robot_sender_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        server_address = ((self.host, self.port_send))
        try:
            self.robot_sender_socket.connect(server_address)
        except ConnectionRefusedError:  # used to turn off the joystick
            self.power_off()
            return
        # Send data
        self.robot_sender_socket.sendall(bytes(json_message, "utf-8"))
        self.robot_sender_socket.close()
        print("sent", json_message)

    def listen_to_robot(self):
        self.robot_receiver_socket.listen(10)
        self.robot_running = True

        while self.robot_running:
            connection, client = self.robot_receiver_socket.accept()
            data_b, response_len = conn_recv(connection)
            # quickly close connection to open up for the next input
            connection.close()
            print("%sreceived" % color_blue, response_len,
                  "bytes from server%s" % color_reset)

            if data_b is not None and response_len > 0:
                self.request_world = False
                data_str = data_b.decode("utf-8")  # bytes to str
                current_world = json.loads(data_str)
                if not current_world['robot_on']:
                    return
                # append new world to storage of all past worlds
                self.sim_states.append(current_world)
                # self.velocities[current_world['sim_t']] = compute_all_velocities(self.sim_states)
                # self.accelerations[current_world['sim_t']] = compute_all_accelerations(self.sim_states)

                if current_world['robot_on'] is True:
                    if current_world['environment']:  # not empty
                        # notify the robot that the joystick received the environment
                        joystick_ready = self.create_message(
                            True, [], [], -1, False)
                        self.send_to_robot(joystick_ready)

                        # only update the environment if it is non-empty
                        self.environment = current_world['environment']
                        robots = list(current_world["robots"].values())
                        assert(len(robots) == 1)  # there should only be one
                        robot = robots[0]
                        self.current_config = generate_config_from_pos_3(
                            robot["current_config"])
                        print("Updated environment from robot")

                        # update the start and goal configs
                        self.start_config = generate_config_from_pos_3(
                            robot["start_config"])
                        self.goal_config = generate_config_from_pos_3(
                            robot["goal_config"])
                        self.delta_t = current_world["delta_t"]
                    else:
                        # render when not receiving a new environment
                        self.generate_frame(current_world, self.frame_num)
                else:
                    print("powering off joystick")
                    self.power_off()
                    break
            else:
                break

    def establish_robot_sender_connection(self):
        """This is akin to a client connection (joystick is client)"""
        self.robot_sender_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        robot_address = ((self.host, self.port_send))
        try:
            self.robot_sender_socket.connect(robot_address)
        except:
            print("%sUnable to connect to robot%s" % (color_red, color_reset))
            print("Make sure you have a simulation instance running")
            exit(1)
        print("%sJoystick->Robot connection established%s" %
              (color_green, color_reset))
        assert(self.robot_sender_socket is not None)

    def establish_robot_receiver_connection(self):
        """This is akin to a server connection (robot is server)"""
        self.robot_receiver_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.robot_receiver_socket.bind((self.host, self.port_recv))
        # wait for a connection
        self.robot_receiver_socket.listen(1)
        connection, client = self.robot_receiver_socket.accept()
        print("%sRobot---->Joystick connection established%s" %
              (color_green, color_reset))
        # start the listening thread for recieving world states from robot
        self.listen_thread = threading.Thread(target=self.listen_to_robot)
        self.listen_thread.start()
        while(self.environment is None):
            # wait until environment is fully sent
            time.sleep(0.01)
        return connection, client

    """ END socket utils """

    def generate_frame(self, world_state, frame_count, plot_quiver=False):
        # extract the information from the world state
        environment = self.environment
        agents = world_state['gen_agents']
        prerecs = world_state['prerecs']
        robots = world_state['robots']
        sim_time = world_state['sim_t']
        # process the information
        map_scale = eval(environment["map_scale"])  # float
        room_center = np.array(environment["room_center"])
        traversible = np.array(environment['traversibles'][0])
        # Compute the real_world extent (in meters) of the traversible
        extent = [0., traversible.shape[1], 0., traversible.shape[0]]
        extent = np.array(extent) * map_scale
        # plot the matplot imgs
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        # plot the img on the axis
        ax.imshow(traversible, extent=extent, cmap='gray',
                  vmin=-.5, vmax=1.5, origin='lower')

        # get number of pixels per meter based off the ax plot space
        img_scale = ax.transData.transform(
            (0, 1)) - ax.transData.transform((0, 0))
        # print(img_scale)
        ppm = img_scale[1]  # number of pixels per "meter" unit in the plot

        # Plot the camera (robots)
        plot_agents(ax, ppm, robots, json_key="current_config", label="Robot",
                    normal_color="bo", collided_color="ko", plot_trajectory=False, plot_quiver=True,
                    plot_start_goal=True, start_3=self.start_config.to_3D_numpy(),
                    goal_3=self.goal_config.to_3D_numpy())

        # plot all the simulated prerecorded gen_agents
        plot_agents(ax, ppm, prerecs, json_key="current_config", label="Prerec",
                    normal_color="yo", collided_color="ro", plot_trajectory=False, plot_quiver=plot_quiver)

        # plot all the randomly generated simulated gen_agents
        plot_agents(ax, ppm, agents, json_key="current_config", label="Agent",
                    normal_color="go", collided_color="ro", plot_trajectory=False, plot_quiver=plot_quiver)

        # save the axis to a file
        filename = "jview" + str(frame_count) + ".png"
        self.dirname = 'tests/socnav/joystick_movie'
        full_file_name = os.path.join(
            get_path_to_humanav(), self.dirname, filename)
        if(not os.path.exists(full_file_name)):
            touch(full_file_name)  # Just as the bash command
        fig.savefig(full_file_name, bbox_inches='tight', pad_inches=0)
        # clear matplot from memory
        fig.clear()
        plt.close(fig)
        del fig
        plt.clf()
        # update frame count
        self.frame_num += 1
