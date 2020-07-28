from utils.utils import print_colors, generate_name
from simulators.agent import Agent
from humans.human_configs import HumanConfigs
from trajectory.trajectory import SystemConfig
from params.robot_params import create_params
import numpy as np
import socket, time, threading, sys

class RoboAgent(Agent):
    def __init__(self, name, start_configs, trajectory=None):
        self.name = name
        self.commands = []
        self.running = False
        self.freq = 100. # update frequency
        self.params = create_params()
        # sockets for communication
        self.joystick_reciever_socket = None
        self.host = socket.gethostname()
        self.port_recv = self.params.port # port for recieving commands from the joystick
        self.port_send = self.port_recv+1 # port for sending commands to the joystick
        # robot's knowledge of the current state of the world
        self.current_state = None
        super().__init__(start_configs.get_start_config(), start_configs.get_goal_config(), name)

    # Getters for the robot class
    def get_name(self):
        return self.name

    # Setters for the robot class
    def update_state(self, state):
        self.current_state = state

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
                                               center=np.array([0., 0., 0.]),
                                               radius=5.):
        """
        Sample a new robot without knowing any configs or appearance fields
        NOTE: needs environment to produce valid configs
        """
        configs = HumanConfigs.generate_random_human_config(environment,
                                                            center,
                                                            radius=radius)
        return RoboAgent.generate_robot(configs)

    def sense(self):
        """use this to take in a world state and compute obstacles (agents/walls) to affect the robot"""
        # TODO: make sure these termination conditions ignore any 'success' or 'timeout' states 
        self._enforce_episode_termination_conditions()
        if(self.end_episode):
            self.collided = True
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
        num_executed = 0 # keeps track of the latest command that is to be executed
        while(self.running):
            # only execute the most recent commands
            if(num_executed >= len(self.commands)):
                time.sleep(1./self.freq) 
                # NOTE: send a command to the joystick letting it know to send another command
                # self.ping_joystick(True)
            else:
                self.sense()
                # using a loop to carry through the backlock of commands over time
                while(num_executed < len(self.commands)):
                    self.execute(num_executed)
                    num_executed += 1
                    if(self.get_trajectory().k != self.get_trajectory().position_nk2().shape[1]):
                        # fix this nonfatal bug
                        print("ERROR: robot_trajectory dimens mismatch")
                        # print(self.get_trajectory().k, self.get_trajectory().position_nk2().shape[1])
                        # exit(0)
            # print(num_executed)
        # self.ping_joystick(False)
        print("\nRobot powering off, recieved", len(self.commands),"commands")
        self.power_off()
        listen_thread.join()
        sys.exit(0)
 
    def power_off(self):
        if(self.running):
            # if the robot is already "off" do nothing
            self.running = False
            self.joystick_reciever_socket.close()

    """BEGIN socket utils"""

    def ping_joystick(self, message):
        # Send data
        message = str(message)
        self.joystick_reciever_socket.sendall(bytes(message, "utf-8"))

    def listen_to_joystick(self):
        self.joystick_reciever_socket.listen(10)
        self.running = True # initialize listener
        while(self.running):
            connection, client = self.joystick_reciever_socket.accept()
            while(True):
                # TODO: allow for buffered data, thus no limit
                data = connection.recv(128)
                # quickly close connection to open up for the next input
                # connection.close()
                # NOTE: data is in the form (running, time, lin_command, ang_command)
                # TODO: use ast.literal_eval instead of eval to
                if(data):
                    # print(data)
                    data = eval(data)
                    np_data = np.array([data[2], data[3]], dtype=np.float32)
                    # NOTE: commands can also be a dictionary indexed by time
                    self.commands.append(np_data)
                    if(data[0] is False):
                        self.running = False
                        break
                else:
                    break
            # close connection to be reaccepted when the joystick sends data
            connection.close()

    def establish_joystick_connection(self):
        """This is akin to a server connection (robot is server)"""
        self.joystick_reciever_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.joystick_reciever_socket.bind((self.host, self.port_recv))
        # wait for a connection
        self.joystick_reciever_socket.listen(1)
        connection, client = self.joystick_reciever_socket.accept()
        # self.ping_joystick(True)
        return connection, client

    """ END socket utils """