from utils.utils import *
from simulators.agent import Agent
from humans.human_configs import HumanConfigs
from trajectory.trajectory import SystemConfig
from params.central_params import create_robot_params
import numpy as np
import socket
import ast
import time
import threading
import sys


lock = threading.Lock()  # for asynchronous data sending


class RobotAgent(Agent):
    joystick_receiver_socket = None
    joystick_sender_socket = None
    host = '127.0.0.1'
    port_send = None  # to be added later from params
    port_recv = None  # to be added later from params

    def __init__(self, name, start_configs, trajectory=None):
        self.name = name
        super().__init__(start_configs.get_start_config(),
                         start_configs.get_goal_config(),
                         name=name, with_init=False)
        self.joystick_inputs = []
        # robot's knowledge of the current state of the world
        self.world_state = None
        # josystick is ready once it has been sent an environment
        self.joystick_ready = False
        # To send the world state on the next joystick ping
        self.joystick_requests_world = -1
        # whether or not to repeat the last joystick input
        self.repeat_joystick = False
        # told the joystick that the robot is powered off
        self.notified_joystick = False
        # used to keep the listener thread alive even if the robot isnt
        self.simulator_running = False
        # amount of time the robot is blocking on the joystick
        self.block_time_total = 0

    def simulation_init(self, sim_map, with_planner=False):
        super().simulation_init(sim_map, with_planner=with_planner)
        self.params.robot_params = create_robot_params()
        # velocity bounds when teleporting to positions (if not using sys dynamics)
        self.v_bounds = self.params.system_dynamics_params.v_bounds
        self.w_bounds = self.params.system_dynamics_params.w_bounds
        self.repeat_freq = self.params.repeat_freq
        # simulation update init
        self.running = True
        self.num_executed = 0  # keeps track of the latest command that is to be executed
        self.num_cmds_per_batch = 1
        # default simulator delta_t, to be updated via set_sim_delta_t() later
        self.sim_delta_t = 0.05

    # Getters for the robot class

    def get_name(self):
        return self.name

    def get_radius(self):
        return self.params.robot_params.physical_params.radius

    # Setters for the robot class
    def update_world(self, state):
        self.world_state = state

    def get_num_executed(self):
        return int(np.floor(len(self.joystick_inputs) / self.num_cmds_per_batch))

    def set_sim_delta_t(self, sim_delta_t):
        self.sim_delta_t = sim_delta_t

    def get_block_t_total(self):
        return self.block_time_total

    @staticmethod
    def generate_robot(configs, name=None, verbose=False):
        """
        Sample a new random robot agent from all required features
        """
        robot_name = "robot_agent"  # constant name for the robot since there will only ever be one
        # In order to print more readable arrays
        np.set_printoptions(precision=2)
        pos_2 = configs.get_start_config().to_3D_numpy()
        goal_2 = configs.get_goal_config().to_3D_numpy()

        if verbose:
            print("Robot", robot_name, "at", pos_2, "with goal", goal_2)
        return RobotAgent(robot_name, configs)

    @staticmethod
    def random_from_environment(environment):
        """
        Sample a new robot without knowing any configs or appearance fields
        NOTE: needs environment to produce valid configs
        """
        configs = HumanConfigs.generate_random_human_config(environment)
        return RobotAgent.generate_robot(configs)

    def check_termination_conditions(self):
        """use this to take in a world state and compute obstacles (gen_agents/walls) to affect the robot"""
        # check for collisions with other gen_agents
        self.check_collisions(self.world_state)

        # enforce planning termination upon condition
        self._enforce_episode_termination_conditions()

        if self.vehicle_trajectory.k >= self.collision_point_k:
            self.end_acting = True

        if self.get_collided():
            # either Pedestrian Collision or Obstacle Collision
            assert("Collision" in self.termination_cause)
            self.power_off()

        if self.get_completed():
            assert(self.termination_cause == "Success")
            self.power_off()

    def _clip_vel(self, vel, bounds):
        vel = round(float(vel), 3)
        assert(bounds[0] < bounds[1])
        if(bounds[0] <= vel <= bounds[1]):
            return vel
        clipped = min(max(bounds[0], vel), bounds[1])
        print("%svelocity %s out of bounds, clipped to %s%s" %
              (color_red, vel, clipped, color_reset))
        return clipped

    def _clip_posn(self, old_pos3, new_pos3, epsilon: float = 0.01):
        # margin of error for the velocity bounds
        assert(self.sim_delta_t > 0)
        dist_to_new = euclidean_dist2(old_pos3, new_pos3)
        if(abs(dist_to_new / self.sim_delta_t) <= self.v_bounds[1] + epsilon):
            return new_pos3
        # calculate theta of vector
        valid_theta = \
            np.arctan2(new_pos3[1] - old_pos3[1], new_pos3[0] - old_pos3[0])
        # create new position scaled off the invalid one
        max_vel = self.sim_delta_t * self.v_bounds[1]
        valid_x = max_vel * np.cos(valid_theta) + old_pos3[0]
        valid_y = max_vel * np.sin(valid_theta) + old_pos3[1]
        reachable_pos3 = [valid_x, valid_y, valid_theta]
        print("%sposition [%s] is unreachable with v bounds, clipped to [%s]%s" %
              (color_red, iter_print(new_pos3), iter_print(reachable_pos3), color_reset))
        return reachable_pos3

    def execute_velocity_cmds(self):
        for _ in range(self.num_cmds_per_batch):
            if(not self.running):
                break
            self.check_termination_conditions()
            current_config = self.get_current_config()
            # the command is indexed by self.num_executed and is safe due to the size constraints in the update()
            vel_cmd = self.joystick_inputs[self.num_executed]
            assert(len(vel_cmd) == 2)  # always a 2 tuple of v and w
            v = self._clip_vel(vel_cmd[0], self.v_bounds)
            w = self._clip_vel(vel_cmd[1], self.w_bounds)
            # NOTE: the format for the acceleration commands to the open loop for the robot is:
            # np.array([[[L, A]]], dtype=np.float32) where L is linear, A is angular
            command = np.array([[[v, w]]], dtype=np.float32)
            t_seg, _ = Agent.apply_control_open_loop(self, current_config,
                                                     command, 1,
                                                     sim_mode='ideal'
                                                     )
            self.num_executed += 1
            self.vehicle_trajectory.append_along_time_axis(
                t_seg, track_trajectory_acceleration=True)
            # act trajectory segment
            self.current_config = \
                SystemConfig.init_config_from_trajectory_time_index(
                    t_seg,
                    t=-1
                )
            if (self.params.verbose):
                print(self.get_current_config().to_3D_numpy())

    def execute_position_cmds(self):
        for _ in range(self.num_cmds_per_batch):
            if(not self.running):
                break
            self.check_termination_conditions()
            joystick_input = self.joystick_inputs[self.num_executed]
            assert(len(joystick_input) == 4)  # has x,y,theta,velocity
            new_pos3 = joystick_input[:3]
            new_v = joystick_input[3]
            old_pos3 = self.current_config.to_3D_numpy()
            # ensure the new position is reachable within velocity bounds
            new_pos3 = self._clip_posn(old_pos3, new_pos3)
            # move to the new position and update trajectory
            new_config = generate_config_from_pos_3(new_pos3, v=new_v)
            self.set_current_config(new_config)
            self.vehicle_trajectory.append_along_time_axis(
                new_config, track_trajectory_acceleration=True)
            self.num_executed += 1
            if (self.params.verbose):
                print(self.get_current_config().to_3D_numpy())

    def execute(self):
        if(self.params.robot_params.use_system_dynamics):
            self.execute_velocity_cmds()
        else:
            self.execute_position_cmds()

    def update(self, iteration):
        if self.running:
            # send a sim_state if it was requested by the joystick
            if self.joystick_requests_world == 0:
                # has processed all prior commands
                self.send_sim_state()

            # only block on act()'s
            init_block_t = time.time()
            while self.running and self.num_executed >= len(self.joystick_inputs):
                if self.num_executed == len(self.joystick_inputs):
                    if self.joystick_requests_world == 0:
                        self.send_sim_state()
                time.sleep(0.001)
            self.block_time_total += time.time() - init_block_t

            # execute the next command in the queue
            if self.num_executed < len(self.joystick_inputs):
                # execute all the commands on the 'queue'
                self.execute()
                # decrement counter
                if(self.joystick_requests_world > 0):
                    self.joystick_requests_world -= 1
        else:
            self.power_off()

    def power_off(self):
        # if the robot is already "off" do nothing
        if self.running:
            print("\nRobot powering off, received",
                  len(self.joystick_inputs), "commands")
            self.running = False
            try:
                quit_message = self.world_state.to_json(
                    robot_on=False,
                    termination_cause=self.termination_cause
                )
                self.send_to_joystick(quit_message)
            except:
                return

    """BEGIN socket utils"""

    def send_sim_state(self):
        # send the (JSON serialized) world state per joystick's request
        if self.joystick_requests_world == 0:
            world_state = self.world_state.to_json(
                robot_on=self.running
            )
            self.send_to_joystick(world_state)
            # immediately note that the world has been sent:
            self.joystick_requests_world = -1

    def send_to_joystick(self, message: str):
        with lock:
            assert(isinstance(message, str))
            # Create a TCP/IP socket
            RobotAgent.joystick_sender_socket = \
                socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Connect the socket to the port where the server is listening
            server_address = ((RobotAgent.host, RobotAgent.port_send))
            try:
                RobotAgent.joystick_sender_socket.connect(server_address)
            except ConnectionRefusedError:  # used to turn off the joystick
                self.joystick_running = False
                # print("%sConnection closed by joystick%s" % (color_red, color_reset))
                return
            # Send data
            RobotAgent.joystick_sender_socket.sendall(bytes(message, "utf-8"))
            RobotAgent.joystick_sender_socket.close()

    def listen_to_joystick(self):
        """Constantly connects to the robot listener socket and receives information from the
        joystick about the input commands as well as the world requests
        """
        RobotAgent.joystick_receiver_socket.listen(1)
        while(self.simulator_running):
            connection, client = RobotAgent.joystick_receiver_socket.accept()
            data_b, response_len = conn_recv(connection, buffr_amnt=128)
            # close connection to be reaccepted when the joystick sends data
            connection.close()
            if(data_b is not b'' and response_len > 0):
                data_str = data_b.decode("utf-8")  # bytes to str
                if(not self.running):
                    self.joystick_requests_world = 0
                else:
                    self.manage_data(data_str)

    def is_keyword(self, data_str):
        # non json important keyword
        if(data_str == "sense"):
            self.joystick_requests_world = \
                len(self.joystick_inputs) - (self.num_executed)
            return True
        elif(data_str == "ready"):
            self.joystick_ready = True
            return True
        elif(data_str == "abandon"):
            self.power_off()
            return True
        return False

    def manage_data(self, data_str: str):
        if not self.is_keyword(data_str):
            data = json.loads(data_str)
            joystick_input: list = data["j_input"]
            self.num_cmds_per_batch = len(joystick_input)
            # add input commands to queue to keep track of
            for i in range(self.num_cmds_per_batch):
                np_data = np.array(joystick_input[i], dtype=np.float32)
                self.joystick_inputs.append(np_data)
                # duplicate commands if "repeating" instead of blocking
                if self.repeat_joystick:  # if need be, repeat n-1 times
                    repeat_amnt = int(np.floor(
                        (self.params.robot_params.physical_params.repeat_freq / self.num_cmds_per_batch) - 1))
                    for i in range(repeat_amnt):
                        # adds command to local list of individual commands
                        self.joystick_inputs.append(np_data)

    @ staticmethod
    def establish_joystick_receiver_connection():
        """This is akin to a server connection (robot is server)"""
        RobotAgent.joystick_receiver_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        RobotAgent.joystick_receiver_socket.bind(
            (RobotAgent.host, RobotAgent.port_recv))
        # wait for a connection
        RobotAgent.joystick_receiver_socket.listen(1)
        print("Waiting for Joystick connection...")
        connection, client = RobotAgent.joystick_receiver_socket.accept()
        print("%sRobot <-- Joystick (receiver) connection established%s" %
              (color_green, color_reset))
        return connection, client

    @ staticmethod
    def establish_joystick_sender_connection():
        """This is akin to a client connection (joystick is client)"""
        RobotAgent.joystick_sender_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        address = ((RobotAgent.host, RobotAgent.port_send))
        try:
            RobotAgent.joystick_sender_socket.connect(address)
        except:
            print("%sUnable to connect to joystick%s" %
                  (color_red, color_reset))
            print("Make sure you have a joystick instance running")
            exit(1)
        assert(RobotAgent.joystick_sender_socket is not None)
        print("%sRobot --> Joystick (sender) connection established%s" %
              (color_green, color_reset))

    @ staticmethod
    def close_robot_sockets():
        RobotAgent.joystick_sender_socket.close()
        RobotAgent.joystick_receiver_socket.close()

    @ staticmethod
    def establish_joystick_handshake(p):
        if(p.episode_params.without_robot):
            # lite-mode episode does not include a robot or joystick
            return
        import json
        # sockets for communication
        RobotAgent.host = '127.0.0.1'
        # port for recieving commands from the joystick
        RobotAgent.port_recv = p.robot_params.port
        # port for sending commands to the joystick (successor of port_recv)
        RobotAgent.port_send = RobotAgent.port_recv + 1
        RobotAgent.establish_joystick_receiver_connection()
        time.sleep(0.01)
        RobotAgent.establish_joystick_sender_connection()
        # send the preliminary episodes that the socnav is going to run
        json_dict = {}
        json_dict['episodes'] = list(p.episode_params.tests.keys())
        episodes = json.dumps(json_dict)
        # Create a TCP/IP socket
        send_episodes_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        server_address = ((RobotAgent.host, RobotAgent.port_send))
        send_episodes_socket.connect(server_address)
        send_episodes_socket.sendall(bytes(episodes, "utf-8"))
        send_episodes_socket.close()

    """ END socket utils """
