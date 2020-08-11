from utils.utils import *
from simulators.agent import Agent
from humans.human_configs import HumanConfigs
from trajectory.trajectory import SystemConfig
from params.robot_params import create_params
import numpy as np
import socket
import ast
import time
import threading
import sys


class RoboAgent(Agent):
    def __init__(self, name, start_configs, trajectory=None):
        self.name = name
        self.commands = []
        self.running = False
        self.freq = 100.  # update frequency
        self.params = create_params()
        # sockets for communication
        self.joystick_receiver_socket = None
        self.joystick_sender_socket = None
        self.host = socket.gethostname()
        self.port_recv = self.params.port  # port for recieving commands from the joystick
        self.port_send = self.port_recv + 1  # port for sending commands to the joystick
        # robot's knowledge of the current state of the world
        self.world_state = None
        super().__init__(start_configs.get_start_config(),
                         start_configs.get_goal_config(), name)
        self.radius = self.params.radius
        self.joystick_ready = False  # josystick is ready once it has been sent an environment
        self.joystick_requests_world = False  # to send the world state
        self.joystick_requests_heard = 0

    # Getters for the robot class
    def get_name(self):
        return self.name

    # Setters for the robot class
    def update_world(self, state):
        self.world_state = state

    @staticmethod
    def generate_robot(configs, name=None, verbose=False):
        """
        Sample a new random robot agent from all required features
        """
        robot_name = None
        if(name is None):
            robot_name = generate_name(20)
        else:
            robot_name = name
        # In order to print more readable arrays
        np.set_printoptions(precision=2)
        pos_2 = (configs.get_start_config().position_nk2().numpy())[0][0]
        goal_2 = (configs.get_goal_config().position_nk2().numpy())[0][0]
        if(verbose):
            print("Robot", robot_name, "at", pos_2, "with goal", goal_2)
        return RoboAgent(robot_name, configs)

    @staticmethod
    def generate_random_robot_from_environment(environment,
                                               radius=5.):
        """
        Sample a new robot without knowing any configs or appearance fields
        NOTE: needs environment to produce valid configs
        """
        configs = HumanConfigs.generate_random_human_config(environment,
                                                            radius=radius)
        return RoboAgent.generate_robot(configs)

    def sense(self):
        """use this to take in a world state and compute obstacles (agents/walls) to affect the robot"""
        # TODO: make sure these termination conditions ignore any 'success' or 'timeout' states
        if(not self.end_episode):
            if(self.world_state is not None):
                # check for collisions with other agents
                own_pos = self.get_current_config().position_nk2().numpy()
                for a in self.world_state.get_agents().values():
                    othr_pos = a.get_current_config().position_nk2().numpy()
                    if(euclidean_dist(own_pos[0][0], othr_pos[0][0]) < self.get_radius() + a.get_radius()):
                        # instantly collide and stop updating
                        self.has_collided = True
                        self.end_acting = True
            self._enforce_episode_termination_conditions()
            # NOTE: enforce_episode_terminator updates the self.end_episode variable
            if(self.end_episode or self.has_collided):
                self.has_collided = True
                self.power_off()

    def execute(self, command_indx):
        current_config = self.get_current_config()
        # the command is indexed by command_indx and is safe due to the size constraints in the update()
        command = np.array([[self.commands[command_indx]]], dtype=np.float32)
        # NOTE: the format for the acceleration commands to the open loop for the robot is:
        # np.array([[[L, A]]], dtype=np.float32) where L is linear, A is angular
        t_seg, actions_nk2 = self.apply_control_open_loop(current_config,
                                                          command, 1, sim_mode='ideal'
                                                          )
        self.vehicle_trajectory.append_along_time_axis(t_seg)
        # act trajectory segment
        self.current_config = \
            SystemConfig.init_config_from_trajectory_time_index(
                t_seg,
                t=-1
            )
        if (self.params.verbose):
            print(self.get_current_config().to_3D_numpy())

    def update(self):
        print("Robot powering on")
        listen_thread = threading.Thread(target=self.listen_to_joystick)
        listen_thread.start()
        self.running = True
        self.last_command = None
        num_executed = 0  # keeps track of the latest command that is to be executed
        while(self.running):
            # only execute the most recent commands
            if(num_executed >= len(self.commands)):
                time.sleep(1. / self.freq)
                # NOTE: send a command to the joystick letting it know to send another command
            else:
                # using a loop to carry through the backlock of commands over time
                while(num_executed < len(self.commands) and self.running):
                    self.sense()
                    self.execute(num_executed)
                    num_executed += 1
                    if(self.get_trajectory().k != self.get_trajectory().position_nk2().shape[1]):
                        # TODO: fix this uncommonly-occuring nonfatal bug
                        print("ERROR: robot_trajectory dimens mismatch")
                    time.sleep(1. / self.freq)
            # notify the joystick that the robot can take another input
            if(self.joystick_requests_world):  # only send when joystick requests
                self.send_to_joystick(
                    self.world_state.to_json(robot_on=True, include_map=False))
                # immediately note that the world has been sent:
                self.joystick_requests_world = False
        # notify the joystick to stop sending commands to the robot
        self.send_to_joystick(self.world_state.to_json(robot_on=False))
        print("\nRobot powering off, recieved", len(self.commands), "commands")
        self.power_off()
        listen_thread.join()
        sys.exit(0)

    def power_off(self):
        if(self.running):
            # if the robot is already "off" do nothing
            self.running = False
            self.joystick_receiver_socket.close()

    """BEGIN socket utils"""

    def send_to_joystick(self, message):
        # Create a TCP/IP socket
        self.joystick_sender_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        server_address = ((self.host, self.port_send))
        # print(self.host, self.port)
        try:
            self.joystick_sender_socket.connect(server_address)
        except ConnectionRefusedError:  # used to turn off the joystick
            self.joystick_running = False
            print("%sConnection closed by joystick%s" %
                  (color_red, color_reset))
            exit(1)
        # Send data
        if(not isinstance(message, str)):
            message = str(message)
        self.joystick_sender_socket.sendall(bytes(message, "utf-8"))
        self.joystick_sender_socket.close()

    def listen_to_joystick(self):
        self.joystick_receiver_socket.listen(10)
        self.running = True  # initialize listener
        while(self.running):
            connection, client = self.joystick_receiver_socket.accept()
            self.joystick_requests_heard += 1  # update number of heard joystick requests
            while(True):
                data_b, response_len = conn_recv(connection, buffr_amnt=128)

                if(data_b is not None):
                    # TODO: commands can also be a dictionary indexed by time
                    # NOTE: data is in the form:
                    # (joystick running, joystick time, lin acc, ang comm, requests_world)
                    data_str = data_b.decode("utf-8")  # bytes to str
                    data = ast.literal_eval(data_str)
                    if(data[-1] is False and data[1] != -1):
                        np_data = np.array(
                            [data[2], data[3]], dtype=np.float32)
                        self.commands.append(np_data)
                        if(data[0] is False):
                            self.running = False
                    elif data[1] == -1:  # only sent by joystick when "ready" and needs the map
                        self.joystick_ready = True
                    else:
                        assert(data[-1] is True)
                        self.joystick_requests_world = True
                    break
                else:
                    break
            # close connection to be reaccepted when the joystick sends data
            connection.close()

    def establish_joystick_receiver_connection(self):
        """This is akin to a server connection (robot is server)"""
        self.joystick_receiver_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.joystick_receiver_socket.bind((self.host, self.port_recv))
        # wait for a connection
        self.joystick_receiver_socket.listen(1)
        print("Waiting for Joystick connection")
        connection, client = self.joystick_receiver_socket.accept()
        print("%sRobot---->Joystick connection established%s" %
              (color_green, color_reset))
        return connection, client

    def establish_joystick_sender_connection(self):
        """This is akin to a client connection (joystick is client)"""
        # TODO: prior to simply connecting, wait for the joystick to send a "ready to connect" message
        self.joystick_sender_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        address = ((self.host, self.port_send))
        try:
            self.joystick_sender_socket.connect(address)
        except:
            print("%sUnable to connect to joystick%s" %
                  (color_red, color_reset))
            print("Make sure you have a joystick instance running")
            exit(1)
        assert(self.joystick_sender_socket is not None)
        print("%sJoystick->Robot connection established%s" %
              (color_green, color_reset))

    """ END socket utils """
