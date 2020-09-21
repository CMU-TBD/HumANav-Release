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
    host = None
    port_send = None
    port_recv = None

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
        self.joystick_requests_world = False
        # whether or not to repeat the last joystick input
        self.repeat_joystick = False
        # told the joystick that the robot is powered off
        self.notified_joystick = False
        # used to keep the listener thread alive even if the robot isnt
        self.simulator_running = False

    def simulation_init(self, sim_map, with_planner=False):
        super().simulation_init(sim_map, with_planner=with_planner)
        self.params.robot_params = create_robot_params()
        # velocity bounds when teleporting to positions (if not using sys dynamics)
        self.v_bounds = self.params.system_dynamics_params.v_bounds
        self.w_bounds = self.params.system_dynamics_params.w_bounds
        self.repeat_freq = self.params.repeat_freq
        # simulation update init
        self.running = True
        self.last_command = None
        self.num_executed = 0  # keeps track of the latest command that is to be executed
        self.amnd_per_batch = 1
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
        return int(np.floor(len(self.joystick_inputs) / self.amnd_per_batch))

    def set_sim_delta_t(self, sim_delta_t):
        self.sim_delta_t = sim_delta_t

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
        if(verbose):
            print("Robot", robot_name, "at", pos_2, "with goal", goal_2)
        return RobotAgent(robot_name, configs)

    @staticmethod
    def generate_random_robot_from_environment(environment):
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

        if(self.vehicle_trajectory.k >= self.collision_point_k):
            self.end_acting = True

        if(self.get_collided()):
            assert(self.termination_cause == 'Collision')
            self.power_off()

        if(self.get_completed()):
            assert(self.termination_cause == "Success")
            self.power_off()

    def _clip_vel(self, vel, bounds):
        vel = round(float(vel), 3) * 2
        assert(bounds[0] < bounds[1])
        if(bounds[0] <= vel <= bounds[1]):
            return vel
        clipped = min(max(bounds[0], vel), bounds[1])
        print("velocity {} out of bounds, clipped to {}".format(vel, clipped))
        return clipped

    def execute_velocity_cmds(self):
        for _ in range(self.amnd_per_batch):
            if(not self.running):
                break
            self.check_termination_conditions()
            current_config = self.get_current_config()
            vel_cmd = self.joystick_inputs[self.num_executed]
            assert(len(vel_cmd) == 2)  # always a 2 tuple of v and w
            # the command is indexed by self.num_executed and is safe due to the size constraints in the update()
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
        for _ in range(self.amnd_per_batch):
            if(not self.running):
                break
            self.check_termination_conditions()
            joystick_input = self.joystick_inputs[self.num_executed][0]
            assert(len(joystick_input) == 4)  # has x,y,theta,velocity
            new_pos3 = joystick_input[:3]
            new_v = joystick_input[3]
            old_pos3 = self.current_config.to_3D_numpy()
            # ensure the new position is reachable within velocity bounds
            dist_to_new = euclidean_dist2(old_pos3, new_pos3)
            assert(self.sim_delta_t > 0)
            if(abs(dist_to_new / self.sim_delta_t) > self.v_bounds[1]):
                # create new position scaled off the invalid one
                valid_theta = new_pos3[2]
                max_vel = self.sim_delta_t * self.v_bounds[1]
                valid_x = max_vel * np.cos(new_pos3[2]) + old_pos3[0]
                valid_y = max_vel * np.sin(new_pos3[2]) + old_pos3[1]
                new_pos3 = [valid_x, valid_y, valid_theta]
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
            # only execute the most recent commands
            self.check_termination_conditions()
            if self.num_executed < len(self.joystick_inputs):
                self.execute()
            # block joystick until recieves next command or finish sending world
            while (self.running and (self.joystick_requests_world or iteration >= self.get_num_executed())):
                time.sleep(0.001)
        else:
            self.power_off()

    def power_off(self):
        # if the robot is already "off" do nothing
        if(self.running):
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
        if self.joystick_requests_world:
            world_state = self.world_state.to_json(
                robot_on=self.running
            )
            self.send_to_joystick(world_state)
            # immediately note that the world has been sent:
            self.joystick_requests_world = False

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
                    # with the robot_on=False flag
                    self.send_sim_state()
                else:
                    self.manage_data(data_str)

    def is_keyword(self, data_str):
        # non json important keyword
        if(data_str == "sense"):
            self.joystick_requests_world = True
            self.send_sim_state()
            return True
        elif(data_str == "ready"):
            self.joystick_ready = True
            return True
        elif(data_str == "abandon"):
            self.power_off()
            return True
        return False

    def manage_data(self, data_str: str):
        if(not self.is_keyword(data_str)):
            data = json.loads(data_str)
            if(self.params.robot_params.use_system_dynamics):
                v_cmds: list = data["vel_cmds"]
                self.amnd_per_batch = len(v_cmds)
            else:
                posn_cmd: list = data["pos_cmds"]
                self.amnd_per_batch = len(posn_cmd)
            for i in range(self.amnd_per_batch):
                if(self.params.robot_params.use_system_dynamics):
                    np_data = np.array(v_cmds[i], dtype=np.float32)
                else:
                    np_data = np.array([posn_cmd[i]], dtype=np.float32)
                self.joystick_inputs.append(np_data)
                if(self.repeat_joystick):  # if need be, repeat n-1 times
                    repeat_amnt = int(np.floor(
                        (self.params.robot_params.physical_params.repeat_freq / self.amnd_per_batch) - 1))
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
        print("%sRobot---->Joystick connection established%s" %
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
        print("%sJoystick->Robot connection established%s" %
              (color_green, color_reset))

    @ staticmethod
    def close_robot_sockets():
        RobotAgent.joystick_sender_socket.close()
        RobotAgent.joystick_receiver_socket.close()

    """ END socket utils """
