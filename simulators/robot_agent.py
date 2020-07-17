from utils.utils import print_colors, generate_name
from simulators.agent import Agent
from humans.human_configs import HumanConfigs
import numpy as np
import socket, time

class RoboAgent(Agent):
    def __init__(self, name, start_configs, trajectory=None):
        self.name = name
        self.commanded_actions_nkf = []
        self.time_intervals = []
        super().__init__(start_configs.get_start_config(), start_configs.get_goal_config(), name)

    # Getters for the Human class
    # NOTE: most of the dynamics/configs implementation is in Agent.py
    def get_name(self):
        return self.name

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
            print(" robot", robot_name, "at", pos_2, "with goal", goal_2)
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

    def listen(self, host=None, port=None):
        """Loop through and update commanded actions as new data 
        comes from a listening socket"""
        while(len(self.time_intervals) < 10):  # TODO: make a SIM_DONE flag
            t, action = self._listen_for_commands(host, port)
            self.time_intervals.append(t)
            # TODO: shouldn't use commanded_actions_nkf, rather use a control scheme that
            # simply takes the control commands (without doing any fancy tf stuff) and runs them
            # through the open feedback loop in agents.py (generating control stuff and trajectory)
            self.commanded_actions_nkf.append(action)
            # self.apply_control_open_loop(self.get_current_config(),
            #                             self.commanded_actions_nkf,
            #                             T=self.params.control_horizon-1,
            #                             sim_mode=self.system_dynamics.simulation_params.simulation_mode)
            # TODO: make it so that the robot will update its current 
            # trajectory based off the commanded actions (ie. action)
            # possibly at a set interval (update freq), and figure out
            # how the transmitting of actions works exactly to test it

    def _listen_for_commands(self, host=None, port=None):
        # Create a TCP/IP socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Define host
        if(host is None):
            host = socket.gethostname()
        # define the communication port
        if (port is None):
            port = 5010
        # Bind the socket to the port
        sock.bind((host, port))
        # Listen for incoming connections
        sock.listen(10)
        # Wait for a connection
        connection, client = sock.accept()
        # Receive the data in small chunks (bytes)
        # NOTE: the #bytes is the MAX length of the string
        data = connection.recv(128)
        data = data.decode('utf-8')
        print(data)
        # Close the connection
        sock.close()
        # return time of retrieving data as well as the data itself
        return time.clock(), data
    
    @staticmethod
    def send_commands(commands, host = None, port = None):
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Define host
        if(host is None):
            host = socket.gethostname()
        # define the communication port
        if (port is None):
            port = 5010
        # Connect the socket to the port where the server is listening
        server_address = ((host, port))
        client_socket.connect(server_address)
        # Send data
        client_socket.sendall(bytes(str(commands), "utf-8"))
        # Close communication channel
        client_socket.close()