import socket
import sys
import json
import os
from time import sleep
from copy import deepcopy
from random import randint
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # for rendering without a display
import matplotlib.pyplot as plt
from utils.utils import *
from utils.image_utils import *
from params.central_params import create_robot_params, get_path_to_socnav, get_seed, create_agent_params
from simulators.sim_state import SimState


# seed the random number generator
random.seed(get_seed())


class Episode():
    def __init__(self, name: str, environment: dict, agents: dict,
                 t_budget: float, r_start: list, r_goal: list):
        self.name = name
        self.environment = environment
        self.agents = agents
        self.time_budget = t_budget
        self.robot_start = r_start    # starting position of the robot
        self.robot_goal = r_goal      # goal position of the robot

    def get_name(self):
        return self.name

    def get_environment(self):
        return self.environment

    def get_agents(self):
        return self.agents

    def get_time_budget(self):
        return self.time_budget

    def update(self, env, agents):
        if(not (not env)):  # only upate env if non-empty
            self.environment = env
        # update dict of agents
        self.agents = agents

    def get_robot_start(self):
        return self.robot_start

    def get_robot_goal(self):
        return self.robot_goal


class Joystick():
    def __init__(self):
        self.t = 0
        # TODO: create explicit joystick params
        self.joystick_params = create_robot_params()
        # episode fields
        self.episode_names = []
        self.current_ep = None
        # main fields
        self.robot_sender_socket = None    # the socket for sending commands to the robot
        self.robot_receiver_socket = None  # world info receiver socket
        self.robot_running = False         # status of the robot in the simulator
        self.host = socket.gethostname()   # using localhost for now
        self.port_send = self.joystick_params.port  # sender port
        self.port_recv = self.port_send + 1         # receiver port
        print("Initiated joystick at", self.host, self.port_send)
        # our 'positions' are modeled as (x, y, theta)
        self.robot_current = None  # current position of the robot
        # planner variables
        self.commanded_actions = []  # the list of commands sent to the robot to execute
        self.sim_delta_t = None      # the delta_t (tickrate) of the simulator
        # data tracking with pandas
        if self.joystick_params.write_pandas_log:
            global pd
            import pandas as pd
            self.pd_df = None    # pandas dataframe for writing to a csv
            self.agent_log = {}  # log of all the agents as updated by sensing
        if(self.joystick_params.track_sim_states):
            self.sim_states = {}  # log of simulator states indexed by time
        if(self.joystick_params.track_vel_accel):
            self.velocities = {}     # velocities of all agents as sensed by the joystick
            self.accelerations = {}  # accelerations of all agents as sensed by the joystick

    def get_episodes(self):
        return self.episode_names

    def _init_obstacle_map(self, renderer=0):
        """ Initializes the sbpd map."""
        p = self.agent_params.obstacle_map_params
        env = self.current_ep.get_environment()
        return p.obstacle_map(p, renderer,
                              res=float(env["map_scale"]) * 100.,
                              trav=np.array(env["traversibles"][0])
                              )

    def init_control_pipeline(self):
        self.agent_params = create_agent_params(with_obstacle_map=True)
        self.obstacle_map = self._init_obstacle_map()
        # TODO: establish explicit limits of freedom for users to use this code
        # self.obj_fn = Agent._init_obj_fn(self, params=self.agent_params)
        # self.obj_fn.add_objective(Agent._init_psc_objective(params=self.agent_params))

        # Initialize Fast-Marching-Method map for agent's pathfinding
        # self.fmm_map = Agent._init_fmm_map(self, params=self.agent_params)
        # Agent._update_fmm_map(self)

        # Initialize system dynamics and planner fields
        # self.planner = Agent._init_planner(self, params=self.agent_params)
        # self.vehicle_data = self.planner.empty_data_dict()
        # self.system_dynamics = Agent._init_system_dynamics(self, params=self.agent_params)
        # self.vehicle_trajectory = Trajectory(dt=self.agent_params.dt, n=1, k=0)

    def create_message(self, joystick_status: bool, v_cmds: list, w_cmds: list,
                       j_time: float = 0.0, req_world: bool = False):
        json_dict = {}
        json_dict["joystick_on"] = joystick_status
        # TODO: rewrute lin_vels and ang_vels to be v_cmds and a_cmds
        if(joystick_status):
            json_dict['j_time'] = j_time
            json_dict["lin_vels"] = v_cmds
            json_dict["ang_vels"] = w_cmds
            json_dict["req_world"] = req_world
        return json.dumps(json_dict, indent=1)

    def robot_input(self, v_cmds: list, w_cmds: list,
                    sense: bool = True,
                    override_power_off: bool = False):
        r_status = self.robot_running and not override_power_off  # robot on or off
        message = self.create_message(r_status, v_cmds, w_cmds, self.t, sense)
        self.send_to_robot(message)

    def random_inputs(self, amnt: int, pr: int = 100):
        # TODO: get these from params
        v_bounds = [0, 1.2]
        w_bounds = [-1.2, 1.2]
        lin_vels = []
        ang_vels = []
        num_cmds_sent = 0  # track how many commands were sent to the robot
        for _ in range(amnt):
            lin_vels.append(randint(v_bounds[0] * pr, v_bounds[1] * pr) / pr)
            ang_vels.append(randint(w_bounds[0] * pr, w_bounds[1] * pr) / pr)
            num_cmds_sent += 1
        self.robot_input(lin_vels, ang_vels, sense=True)

    def update(self):
        assert(self.sim_delta_t)  # obtained from the second J.listen_once()
        print("simulator's refresh rate = %.4f" % self.sim_delta_t)
        print("joystick's refresh rate  = %.4f" % self.agent_params.dt)
        self.robot_receiver_socket.listen(1)  # init listener thread
        self.robot_running = True
        while(self.robot_running):
            # send a command to the robot
            num_actions_per_dt = \
                int(np.floor(self.sim_delta_t / self.agent_params.dt))
            self.random_inputs(num_actions_per_dt)
            # listen to the robot's reply
            if(not self.listen_once()):
                # occurs if the robot is unavailable or it finished
                self.power_off()
                break
        # finished this episode
        print("%sFinished episode:" % color_green,
              self.current_ep.get_name(), "%s" % color_reset)
        # listening to robot
        if self.current_ep.get_name() == self.episode_names[-1]:
            self.close_recv_socket()
            print("Finished all episodes")
        else:
            self.current_ep = None

    def power_off(self):
        if self.robot_running:
            print("%sConnection closed by robot%s" % (color_red, color_reset))
            self.robot_running = False
            try:
                # if the robot socket is still alive, notify it that the joystick has stopped
                self.robot_input([], [], False, override_power_off=True)
            except:
                pass

    """BEGIN socket utils"""

    def close_recv_socket(self):
        if self.robot_running:
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
            self.robot_running = False
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

    def track_vel_accel(self, current_world):
        assert(self.joystick_params.track_vel_accel)
        from simulators.sim_state import compute_all_velocities, compute_all_accelerations
        self.velocities[current_world.get_sim_t()] = \
            compute_all_velocities(list(self.sim_states.values()))
        self.accelerations[current_world.get_sim_t()] = \
            compute_all_accelerations(list(self.sim_states.values()))

    """ BEGIN PD UTILS """

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

    """ END PD UTILS """

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
