import tensorflow as tf
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
        self.world_state = []
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
        self.ready_to_send = False
        self.requests_world = False  # True whenever the joystick wants data about the world
        print("Initiated joystick at", self.host, self.port_send)
        self.start_config = None
        self.goal_config = None
        self.current_config = None

    def set_host(self, h):
        self.host = h

    def _init_obstacle_map(self, renderer=0):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p, renderer, res=float(self.environment["map_scale"]) * 100., trav=np.array(self.environment["traversibles"][0]))

    def init_start_goal(self):
        self.start_config = generate_random_config(self.environment)
        self.goal_config = generate_random_config(self.environment)

    def init_control_pipeline(self):
        assert(self.world_state is not None)
        assert(self.environment is not None)
        self.init_start_goal()
        self.params = create_agent_params()
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
        self.commanded_actions = []

    def create_message(self, joystick_power: bool, j_time: float = 0.0,
                       lin_vel: float = 0.0, ang_vel: float = 0.0, req_world: bool = False):
        json_dict = {}
        json_dict["joystick_on"] = joystick_power
        if(joystick_power):
            json_dict["j_time"] = j_time
            json_dict["lin_vel"] = float(lin_vel)
            json_dict["ang_vel"] = float(ang_vel)
            json_dict["req_world"] = req_world
        return json.dumps(json_dict, indent=1)

    def robot_input(self, lin_command: float, ang_command: float, request_world: bool):
        message = self.create_message(self.robot_running, time.clock(),
                                      float(lin_command), float(ang_command), request_world)
        self.send_to_robot(message)
        print("sent", message)

    def random_robot_joystick(self):
        sent_commands = 0
        self.robot_running = True
        while(self.robot_running is True):
            try:
                if(self.ready_to_send and self.environment is not None):
                    # robot can only more forwards
                    lin_command = (randint(10, 100) / 100.)
                    ang_command = (randint(-100, 100) / 100.)
                    self.robot_input(lin_command, ang_command,
                                     self.requests_world)
                    sent_commands += 1
                    # now update the robot with the "ready" ping
                    time.sleep(2)  # NOTE: this is tunable to ones liking
                    self.ready_to_send = True
                # TODO: create a backlog of commands that were not sent bc the robot wasn't ready
            except KeyboardInterrupt:
                print("%sJoystick disconnected by user%s" %
                      (color_yellow, color_reset))
                self.power_off()
                break

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
            self.commanded_actions.append(commanded_actions_nkf)
            # print(self.planner_data['optimal_control_nk2'])
            for c in commanded_actions_nkf.numpy()[0]:
                lin = c[0]
                ang = c[1]
                self.robot_input(lin, ang, False)
                # TODO: get rid of time, make the joystick send all at once instead of one at a time
                time.sleep(0.01)
            self.current_config = \
                SystemConfig.init_config_from_trajectory_time_index(
                    self.vehicle_trajectory, t=-1)

    def update(self, random_commands: bool = True):
        """ Independent process for a user (at a designated host:port) to recieve
        information from the simulation while also sending commands to the robot """
        if(random_commands):
            self.random_robot_joystick()
        else:
            self.planned_robot_joystick()
        self.listen_thread.join()
        # Close communication channel
        self.robot_sender_socket.close()
        # begin gif (movie) generation
        try:
            save_to_gif(os.path.join(get_path_to_humanav(), self.dirname))
        except:
            print("unable to render gif")

    def power_off(self):
        if(self.robot_running):
            print("%sConnection closed by robot%s" % (color_red, color_reset))
            self.robot_running = False
            try:
                quit_message = self.create_message(False)
                self.send_to_robot(quit_message)  # stop
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

    def listen_to_robot(self):
        self.robot_receiver_socket.listen(10)
        self.robot_running = True
        while(self.robot_running):
            connection, client = self.robot_receiver_socket.accept()
            data_b, response_len = conn_recv(connection)
            # quickly close connection to open up for the next input
            connection.close()
            print("%sreceived" % (color_blue), response_len,
                  "bytes from server%s" % (color_reset))
            if(data_b is not None):
                self.ready_to_send = True  # has received a world state from the robot
                self.requests_world = False
                data_str = data_b.decode("utf-8")  # bytes to str
                self.world_state.append(json.loads(data_str))
                current_world = self.world_state[-1]
                if(current_world['robot_on'] is True):
                    if(current_world['environment']):  # not empty
                        # notify the robot that the joystick received the environment
                        joystick_ready = self.create_message(
                            True, -1, 0, 0, False)
                        self.send_to_robot(joystick_ready)
                        # only update the environment if it is non-empty
                        self.environment = current_world['environment']
                        robot = list(current_world["robots"].values())[0]
                        self.current_config = generate_config_from_pos_3(
                            robot["current_config"])
                        print("Updated environment from robot")
                    self.generate_frame(self.frame_num)
                else:
                    print("powering off joystick")
                    self.power_off()
                    break
            else:
                break
            # this should be a separate thread
            self.requests_world = True

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

    def generate_frame(self, frame_count, plot_quiver=False):
        # extract the information from the world state
        environment = self.environment
        agents = self.world_state[-1]['agents']
        prerecs = self.world_state[-1]['prerecs']
        robots = self.world_state[-1]['robots']
        sim_time = self.world_state[-1]['sim_t']
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
                    plot_start_goal=True, new_start=self.start_config, new_goal=self.goal_config)

        # plot all the simulated prerecorded agents
        plot_agents(ax, ppm, prerecs, json_key="current_config", label="Prerec",
                    normal_color="yo", collided_color="ro", plot_trajectory=False, plot_quiver=plot_quiver)

        # plot all the randomly generated simulated agents
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
