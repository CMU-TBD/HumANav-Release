from utils.utils import print_colors, generate_name
from simulators.agent import Agent
from humans.human_configs import HumanConfigs
from trajectory.trajectory import SystemConfig
import numpy as np
import socket, time, threading

class RoboAgent(Agent):
    def __init__(self, name, start_configs, trajectory=None):
        self.name = name
        self.commands = []
        self.running = False
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

    def old_listen(self, host=None, port=None):
        """Loop through and update commanded actions as new data 
        comes from a listening socket"""
        self.listening = True
        while(len(self.time_intervals) < 100):# self.listening):
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
            """
            tf_lin_vel = tf.constant([[[lin_vel]]], dtype=tf.float32)
            tf_ang_vel = tf.constant([[[ang_vel]]], dtype=tf.float32)
            message = tf.concat([tf_lin_vel, tf_ang_vel], 2)
            """

    def execute(self):
        if(len(self.commands) > 0):
            current_config = self.get_current_config()

            # print(np.ones((1, 1, 2), dtype=np.float32))

            t_seg, actions_nk2 = self.apply_control_open_loop(current_config,   
                                                            np.array([[self.commands[-1]]], dtype=np.float32), 
                                                            1,
                                                            sim_mode='ideal'
                                                            )
            # act trajectory segment
            self.current_config = \
                        SystemConfig.init_config_from_trajectory_time_index(
                        t_seg,
                        t=-1
                    )

    def update(self):
        listen_thread = threading.Thread(target=self.listen, args=(None,None))
        listen_thread.start()
        self.running = True
        while(self.running):
            # if(len(self.commands) > 0):
            #     print(len(self.commands), self.commands[-1])
            self.execute()
        listen_thread.join()
 
    def listen(self, host=None, port=None):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Define host
        if(host is None):
            host = socket.gethostname()
        # define the communication port
        if (port is None):
            port = 5010
        s.bind((host, port))
        s.listen(10)
        self.running = True # initialize listener
        while(self.running):
            connection, client = s.accept()
            while(True): # constantly taking in information until breaks
                # TODO: allow for buffered data, thus no limit
                data = connection.recv(128)
                # quickly close connection to open up for the next input
                connection.close()
                # NOTE: data is in the form (running, time, lin_command, ang_command)
                # TODO: use ast.literal_eval instead of eval to
                data = eval(data)
                np_data = np.array([data[2], data[3]], dtype=np.float32)
                # NOTE: commands can also be a dictionary indexed by time
                self.commands.append(np_data)
                if(data[0] is False):
                    self.running = False
                break
        s.close()

    def send_commands(self, commands, host=None, port=None):
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