import socket
import json
import os
import numpy as np
import random
from utils.utils import *
from params.central_params import create_robot_params, get_path_to_socnav, get_seed
from simulators.sim_state import SimState
from simulators.episode import Episode


# seed the random number generator
random.seed(get_seed())


class JoystickBase():
    def __init__(self):
        self.t = 0
        self.joystick_params = create_robot_params()
        # episode fields
        self.episode_names = []
        self.current_ep = None
        # main fields
        self.joystick_on = True  # status of the joystick
        # socket fields
        self.robot_sender_socket = None    # the socket for sending commands to the robot
        self.robot_receiver_socket = None  # world info receiver socket
        self.host = socket.gethostname()   # using localhost for now
        self.port_send = self.joystick_params.port  # sender port
        self.port_recv = self.port_send + 1         # receiver port
        print("Initiated joystick at", self.host, self.port_send)
        # our 'positions' are modeled as (x, y, theta)
        self.robot_current = None    # current position of the robot
        self.sim_delta_t = None      # the delta_t (tickrate) of the simulator
        # potentially add more fields based off the params
        self.param_based_init()

    def param_based_init(self):
        # data tracking with pandas
        if self.joystick_params.write_pandas_log:
            global pd
            import pandas as pd
            self.pd_df = None    # pandas dataframe for writing to a csv
            self.agent_log = {}  # log of all the agents as updated by sensing

        # keeping an explicit log of every sim_state indexed by time
        if(self.joystick_params.track_sim_states):
            self.sim_states = {}  # log of simulator states indexed by time

        # tracking velocity and acceleration of the agents from the sim states
        if(self.joystick_params.track_vel_accel):
            global velocities
            from simulators.sim_state import compute_all_velocities as velocities
            global accelerations
            from simulators.sim_state import compute_all_accelerations as accelerations
            self.velocities = {}     # velocities of all agents as sensed by the joystick
            self.accelerations = {}  # accelerations of all agents as sensed by the joystick

        # plotting the sim states received from the robot
        if (self.joystick_params.generate_movie):
            import matplotlib as mpl
            mpl.use('Agg')  # for rendering without a display
            import matplotlib.pyplot as plt
            from utils.image_utils import save_to_gif

    def get_episodes(self):
        return self.episode_names

    def _init_obstacle_map(self):
        raise NotImplementedError

    def init_control_pipeline(self):
        raise NotImplementedError

    def create_message(self, joystick_status: bool, v_cmds: list, w_cmds: list,
                       j_time: float = 0.0, req_world: bool = False):
        json_dict = {}
        json_dict["joystick_on"] = joystick_status
        if(joystick_status):
            json_dict['j_time'] = j_time
            json_dict["v_cmds"] = v_cmds
            json_dict["w_cmds"] = w_cmds
            json_dict["sense"] = req_world
        return json.dumps(json_dict, indent=1)

    def robot_input(self, v_cmds: list, w_cmds: list,
                    sense: bool = True,
                    override_power_off: bool = False):
        r_status = self.joystick_on and not override_power_off  # robot on or off
        message = self.create_message(r_status, v_cmds, w_cmds, self.t, sense)
        self.send_to_robot(message)

    def update_loop(self):
        raise NotImplementedError

    def finish_episode(self):
        # finished this episode
        print("%sFinished episode:" % color_green,
              self.current_ep.get_name(), "%s" % color_reset)
        # If current episode is the last one, the joystick is done
        if self.current_ep.get_name() == self.episode_names[-1]:
            self.close_recv_socket()
            print("Finished all episodes")
        else:
            self.current_ep = None

    def power_off(self):
        if self.joystick_on:
            print("%sConnection closed by robot%s" % (color_red, color_reset))
            self.joystick_on = False
            try:
                # if the robot socket is still alive, notify it that the joystick has stopped
                self.robot_input([], [], False, override_power_off=True)
            except:
                pass

    """ BEGIN LISTEN UTILS """

    def get_all_episode_names(self):
        # sets the data to look for all the episode names
        return self.listen_once(0)

    def get_episode_metadata(self):
        # sets data_type to look for the episode metadata
        return self.listen_once(1)

    def listen_once(self, data_type: int = 2):
        """Runs a single instance of listening to the receiver socket
        to obtain information about the world and episode metadata

        Args:
            data_type (int, optional): 0 if obtaining all episode names,
                                       1 if obtaining specific episode metadata,
                                       2 if obtaining simulator info from a sim_state.
                                       Defaults to 2.

        Returns:
            [bool]: True if the listening was successful, False otherwise
        """
        connection, _ = self.robot_receiver_socket.accept()
        data_b, response_len = conn_recv(connection)
        # quickly close connection to open up for the next input
        connection.close()
        if self.joystick_params.verbose:
            print("%sreceived" % color_blue, response_len,
                  "bytes from robot%s" % color_reset)
        if response_len > 0:
            data_str = data_b.decode("utf-8")  # bytes to str
            data_json = json.loads(data_str)
            if (data_type == 0):
                return self.manage_episodes_name_data(data_json)
            elif (data_type == 1):
                return self.manage_episode_data(data_json)
            else:
                return self.manage_sim_state_data(data_json)
        else:
            self.joystick_on = False
            return False
        return True

    def manage_episodes_name_data(self, episode_names_json: dict):
        # case where there is no simulator yet, just episodes
        assert('episodes' in episode_names_json.keys())
        self.episode_names = episode_names_json['episodes']
        print("Received episodes:", self.episode_names)
        assert(len(self.episode_names) > 0)
        return True  # valid parsing of the data

    def manage_episode_data(self, initial_sim_state_json: dict):
        current_world = SimState.from_json(initial_sim_state_json)
        # not empty dictionary
        assert(not (not current_world.get_environment()))
        # ping the robot that the joystick received the environment
        # this is signified by the unique fact that j_time = -1
        joystick_ready = \
            self.create_message(True, [], [], -1, False)
        self.update_knowledge_from_episode(current_world, init_ep=True)
        self.send_to_robot(joystick_ready)
        return True

    def manage_sim_state_data(self, sim_state_json: dict):
        # case where the robot sends a power-off signal
        if not sim_state_json['robot_on']:
            # TODO: fix cause for 'success' state
            print("\npowering off joystick, robot episode terminated with:",
                  sim_state_json['termination_cause'])
            self.power_off()
            return False  # robot is off, do not continue
        else:
            # TODO: make this not a SimState (since that requires importing sim_state.py)
            # and instead just keep as a dictionary for now.
            current_world = SimState.from_json(sim_state_json)
            # only update the SimStates for non-environment configs
            self.update_knowledge_from_episode(current_world)

            # update the history of past sim_states if requested
            if self.joystick_params.track_sim_states:
                self.sim_states[current_world.get_sim_t()] = \
                    current_world

            print("%sUpdated state of the world for time = %.3f out of %.3f\r" %
                  (color_blue, current_world.get_sim_t(),
                   self.current_ep.get_time_budget()),
                  "%s" % color_reset, end="")

            # self.track_vel_accel(current_world)  # TODO: remove

            if self.joystick_params.write_pandas_log:
                # used for file IO such as pandas logging
                self.dirname = 'tests/socnav/' + self.current_ep.get_name() + \
                    '_movie/joystick_data'
                # Write the Agent's trajectory data into a pandas file
                self.update_logs(current_world)
                self.write_pandas()
        return True

    def update_knowledge_from_episode(self, current_world, init_ep: bool = False):
        name = current_world.get_episode_name()
        env = current_world.get_environment()
        # get all the agents in the scene except for the robot
        agents = current_world.get_all_agents(include_robot=False)
        max_t = current_world.get_episode_max_time()
        # gather robot information
        robots = list(current_world.get_robots().values())
        # only one robot is supported
        assert(len(robots) == 1)
        robot = robots[0]
        # update robot's current position
        self.robot_current = robot.get_current_config().to_3D_numpy()
        # episode data
        if(init_ep):
            # only update start/goal when creating an episode
            r_start = robot.get_start_config().to_3D_numpy()
            r_goal = robot.get_goal_config().to_3D_numpy()
            # creates a new instance of the episode for further use
            self.current_ep = \
                Episode(name, env, agents, max_t, r_start, r_goal)
            print("%sRunning test for %s%s" %
                  (color_yellow, self.current_ep.get_name(), color_reset))
            assert(self.current_ep.get_name() in self.episode_names)
        else:
            # option to update the env and agents in the existing (running) episode
            self.current_ep.update(env, agents)
        # update the delta_t of the simulator, which we dont assume is consistent
        self.sim_delta_t = current_world.get_delta_t()

    """ END LISTEN UTILS """

    def track_vel_accel(self, current_world):
        assert(self.joystick_params.track_vel_accel)
        # TODO: ensure the sim_states are sorted by their time, so that traversing
        # the hash map maintains relative order of when they entered
        sim_state_list = list(self.sim_states.values())
        self.velocities[current_world.get_sim_t()] = velocities(sim_state_list)
        self.accelerations[current_world.get_sim_t()] = \
            accelerations(sim_state_list)

    """ BEGIN PANDAS UTILS """

    def update_logs(self, world_state: SimState):
        self.update_log_of_type('robots', world_state)
        self.update_log_of_type('gen_agents', world_state)
        self.update_log_of_type('prerecs', world_state)

    def update_log_of_type(self, agent_type: str, world_state: SimState):
        from simulators.sim_state import get_agents_from_type
        agents_of_type = get_agents_from_type(world_state, agent_type)
        for a in agents_of_type.keys():
            if a not in self.agent_log.keys():
                # initialize dict for a specific agent if dosent already exist
                self.agent_log[a] = {}
            self.agent_log[a][world_state.get_sim_t()] = \
                agents_of_type[a].get_current_config().to_3D_numpy()

    def write_pandas(self):
        assert(self.joystick_params.write_pandas_log)
        pd_df = pd.DataFrame(self.agent_log)
        abs_path = \
            os.path.join(get_path_to_socnav(), self.dirname, 'agent_data.csv')
        if not os.path.exists(abs_path):
            touch(abs_path)  # Just as the bash command
        pd_df.to_csv(abs_path)
        if self.joystick_params.verbose:
            print("%sUpdated pandas dataframe%s" %
                  (color_green, color_reset))

    """ END PANDAS UTILS """

    """ BEGIN socket utils"""

    def close_recv_socket(self):
        if self.joystick_on:
            # connect to the socket, closing it, and continuing the thread to completion
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((self.host, self.port_recv))
            except:
                print("%sClosing listener socket%s" % (color_red, color_reset))
            self.robot_receiver_socket.close()

    def send_to_robot(self, json_message: str):
        # Create a TCP/IP socket
        self.robot_sender_socket = \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        robot_addr = (self.host, self.port_send)
        try:
            self.robot_sender_socket.connect(robot_addr)
            self.robot_sender_socket.sendall(bytes(json_message, "utf-8"))
            self.robot_sender_socket.close()
        except:  # used to turn off the joystick
            self.power_off()
            return
        # Send data
        if self.joystick_params.print_data:
            print("sent", json_message)

    def establish_sender_connection(self):
        """Creates the initial handshake between the joystick and the robot to
        have a communication channel with the external robot process """
        self.robot_sender_socket = \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.robot_sender_socket.connect((self.host, self.port_send))
        except:
            print("%sUnable to connect to robot%s" % (color_red, color_reset))
            print("Make sure you have a simulation instance running")
            exit(1)
        print("%sJoystick->Robot connection established%s" %
              (color_green, color_reset))
        assert(self.robot_sender_socket is not None)
        # set socket timeout
        self.robot_sender_socket.settimeout(5)

    def establish_receiver_connection(self):
        """Creates the initial handshake between the joystick and the meta test
        controller that sends information about the episodes as well as the 
        RobotAgent that sends it's SimStates serialized through json as a 'sense'"""
        self.robot_receiver_socket = \
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.robot_receiver_socket.bind((self.host, self.port_recv))
        # wait for a connection
        self.robot_receiver_socket.listen(1)
        connection, client = self.robot_receiver_socket.accept()
        print("%sRobot---->Joystick connection established%s" %
              (color_green, color_reset))
        return connection, client

    """ END socket utils """
