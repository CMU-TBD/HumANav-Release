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
                         start_configs.get_goal_config(),
                         name)
        self.radius = self.params.radius
        self.repeat_freq = self.params.repeat_freq
        self.joystick_ready = False  # josystick is ready once it has been sent an environment
        self.joystick_requests_world = False  # to send the world state
        # whether or not to repeat the last joystick input
        self.repeat_joystick = False
        # simulation update init
        self.running = True
        self.last_command = None
        self.num_executed = 0  # keeps track of the latest command that is to be executed
        self.amnt_per_joystick = 1

    # Getters for the robot class
    def get_name(self):
        return self.name

    # Setters for the robot class
    def update_world(self, state):
        self.world_state = state

    def get_num_executed(self):
        return int(np.floor(len(self.commands) / self.amnt_per_joystick))

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
        if(not self.end_episode):
            # check for collisions with other agents
            self.check_collisions(self.world_state)
            # enforce planning termination upon condition
            self._enforce_episode_termination_conditions()
            # NOTE: enforce_episode_terminator updates the self.end_episode variable
            if(self.end_episode or self.has_collided):
                self.has_collided = True
                self.power_off()

    def execute(self):
        for _ in range(self.amnt_per_joystick):
            if(not self.running):
                break
            self.sense()
            current_config = self.get_current_config()
            cmd_grp = self.commands[self.num_executed]
            num_cmds_in_grp = len(cmd_grp)

            # the command is indexed by self.num_executed and is safe due to the size constraints in the update()
            command = np.array([[cmd_grp]], dtype=np.float32)
            # NOTE: the format for the acceleration commands to the open loop for the robot is:
            # np.array([[[L, A]]], dtype=np.float32) where L is linear, A is angular
            t_seg, actions_nk2 = Agent.apply_control_open_loop(self, current_config,
                                                               command, num_cmds_in_grp, sim_mode='ideal'
                                                               )
            self.num_executed += 1
            self.vehicle_trajectory.append_along_time_axis(t_seg)
            # act trajectory segment
            self.current_config = \
                SystemConfig.init_config_from_trajectory_time_index(
                    t_seg,
                    t=-1
                )
            if (self.params.verbose):
                print(self.get_current_config().to_3D_numpy())

    def update(self, iteration):
        if(self.running):
            # only execute the most recent commands
            self.sense()
            if(iteration < self.get_num_executed()):
                print(self.num_executed, len(self.commands))
                self.execute()
            # block joystick until recieves next command
            while(iteration >= self.get_num_executed()):
                time.sleep(0.001)
            #     if(self.repeat_joystick and len(self.commands) > 0):
            #         last_command = self.commands[-1]
            #         self.commands.append(last_command)
            #         self.execute()
            # send the (JSON serialized) world state per joystick's request
            self.ping_joystick()
            # quit the robot if it died
            if(not self.running):
                # notify the joystick to stop sending commands to the robot
                self.send_to_joystick(self.world_state.to_json(robot_on=False))
                self.power_off()

    def power_off(self):
        if(self.running):
            print("\nRobot powering off, recieved",
                  len(self.commands), "commands")
            # if the robot is already "off" do nothing
            self.running = False
            self.joystick_receiver_socket.close()

    """BEGIN socket utils"""

    def ping_joystick(self):
        if(self.joystick_requests_world):  # only send when joystick requests
            self.send_to_joystick(
                self.world_state.to_json(robot_on=True, include_map=False))
            # immediately note that the world has been sent:
            self.joystick_requests_world = False

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
            self.power_off()
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
            while(True):
                data_b, response_len = conn_recv(connection, buffr_amnt=128)
                if(data_b is not b''):
                    data_str = data_b.decode("utf-8")  # bytes to str
                    data = json.loads(data_str)
                    if(data["joystick_on"]):
                        if(data["j_time"] >= 0):  # normal command input
                            lin_vels: list = data["lin_vels"]
                            ang_vels: list = data["ang_vels"]
                            assert(len(lin_vels) == len(ang_vels))
                            self.amnt_per_joystick = len(lin_vels)
                            for i in range(self.amnt_per_joystick):
                                np_data = np.array(
                                    [lin_vels[i], ang_vels[i]], dtype=np.float32)
                                # add at least one command
                                self.commands.append(np_data)
                                if(self.repeat_joystick):  # if need be, repeat n-1 times
                                    for i in range(int(np.floor((self.repeat_freq / self.amnt_per_joystick) - 1))):
                                        # adds command to local list of individual commands
                                        self.commands.append(np_data)
                        # only sent by joystick when "ready" and needs the map
                        elif data["j_time"] == -1:
                            self.joystick_ready = True
                        # whether or not the world state is requested
                        if(data["req_world"] is True):
                            # to send the world in the next update
                            self.joystick_requests_world = True
                    else:
                        self.power_off()
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
