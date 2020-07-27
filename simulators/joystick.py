import tensorflow as tf
import socket, threading, multiprocessing
import time, sys
from utils.utils import print_colors

class Joystick():
    def __init__(self, host=None, port=None):
        self.update_host_port(host, port) # defines class' host and port
        self.t = 0
        self.latest_state = None
        self.world_state = (False, 0, 0)
        # sockets for communication
        self.robot_socket = None
        self.robot_running = False
        print("Initiated joystick at", self.host, self.port)

    def set_host(self, h):
        self.host = h

    def set_port(self, p):
        self.port = p

    def random_robot_joystick(self):
        from random import randint
        self.world_state = (True, 0, 0)
        accel_scale = 100 # scale to multiply the raw acceleration values by 
        repeat = 1 # number of times to send the same command to the robot
        sent_commands = 0
        self.robot_running = True
        while(self.robot_running is True):
            lin_command = (randint(10, 100) / 100.) # robot can only more forwards
            ang_command = (randint(-100, 100) / 100.)
            # print(lin_command, ang_command)
            for _ in range(repeat):
                # TODO: remove robot_running stuff
                message = (self.robot_running, time.clock(), lin_command, ang_command)
                self.send(message)
                print("sent", message)
                sent_commands += 1
            # random delay for the Joystick to input commands
            try:
                time.sleep(0.1*randint(0,100)/100.)
            except KeyboardInterrupt:
                print(print_colors()["yellow"], "Joystick disconnected by user", print_colors()['reset'])
                self.send((False, time.clock(), 0, 0)) # stop signal
                sys.exit(0)

    def update(self):
        """ Independent process for a user (at a designated host:port) to recieve 
        information from the simulation while also sending commands to the robot """
        # listen_thread = threading.Thread(target=self.listen, args=(None,None))
        # listen_thread.start()
        self.random_robot_joystick()
        # send a message to the robot to stop execution    
        # halt_message = (False, time.clock(), 0, 0)
        # self.send(halt_message)
        # listen_thread.join()
        # Close communication channel
        self.robot_socket.close()

    """BEGIN socket utils"""

    def update_host_port(self, host, port):
        # Define host
        if(host is None):
            self.host = socket.gethostname()
        else:
            self.host = host
        # Define the communication port
        if (port is None):
            self.port = 6000
        else:
            self.port = port

    def send(self, commands, port=None, host=None):
        # Create a TCP/IP socket
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # update self's host and port
        self.update_host_port(host, port)
        # Connect the socket to the port where the server is listening
        server_address = ((self.host, self.port))
        # print(self.host, self.port)
        try:
            self.robot_socket.connect(server_address)
        except ConnectionRefusedError: # used to turn off the joystick
            self.robot_running = False
            print(print_colors()["red"], "Connection closed by robot", print_colors()['reset'])
            exit(1)
        # Send data
        message = str(commands)
        self.robot_socket.sendall(bytes(message, "utf-8"))
        self.robot_socket.close()

    def listen(self, host=None, port=None):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.update_host_port(host, port)
        s.bind((self.host, self.port))
        s.listen(10)
        self.world_state = (True, 0, 0) # initialize listener
        while(self.world_state[0] is True):
            connection, client = s.accept()
            while(True): # constantly taking in information until breaks
                # TODO: allow for buffered data, thus no limit
                data = connection.recv(128)
                # quickly close connection to open up for the next input
                connection.close()
                self.world_state = eval(data)
                break
        s.close()
    
    def establish_robot_connection(self):
        """This is akin to a client connection (robot is client)"""
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.update_host_port(None, 6000)
        robot_address = ((self.host, self.port))
        try:
            self.robot_socket.connect(robot_address)
        except:
            print(print_colors()["red"], "Unable to connect to robot", print_colors()['reset'])
            print("Make sure you have a simulation instance running")
            exit(1)
        print(print_colors()["green"], "Connection to robot established", print_colors()['reset'])
        assert(self.robot_socket is not None)
    """ END socket utils """