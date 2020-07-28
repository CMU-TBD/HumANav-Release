import tensorflow as tf
import socket, threading, multiprocessing
import time, sys
from utils.utils import print_colors
from params.robot_params import create_params

class Joystick():
    def __init__(self):
        self.t = 0
        self.latest_state = None
        self.world_state = (False, 0, 0)
        self.params = create_params()
        # sockets for communication
        self.robot_sender_socket = None
        self.robot_running = False
        self.host = socket.gethostname()
        self.port_send = self.params.port # port for sending commands to the robot
        self.port_recv = self.port_send+1 # port for recieving commands from the robot
        self.ready_to_send = False
        print("Initiated joystick at", self.host, self.port_send)

    def set_host(self, h):
        self.host = h

    def random_robot_joystick(self):
        from random import randint
        self.world_state = (True, 0, 0)
        accel_scale = 100   # scale to multiply the raw acceleration values by 
        repeat = 1          # number of times to send the same command to the robot
        sent_commands = 0
        self.robot_running = True
        while(self.robot_running is True):
            try:
                if(self.ready_to_send):
                    lin_command = (randint(10, 100) / 100.) # robot can only more forwards
                    ang_command = (randint(-100, 100) / 100.)
                    for _ in range(repeat):
                        # TODO: remove robot_running stuff
                        message = (self.robot_running, time.clock(), lin_command, ang_command)
                        self.send_to_robot(message)
                        print("sent", message)
                        sent_commands += 1
                    # now wait for robot to ping with "ready"
                    self.ready_to_send = False
                # TODO: create a backlog of commands that were not sent bc the robot wasn't ready
            except KeyboardInterrupt:
                print(print_colors()["yellow"], "Joystick disconnected by user", print_colors()['reset'])
                self.send_to_robot((False, time.clock(), 0, 0)) # stop signal
                sys.exit(0)

    def update(self):
        """ Independent process for a user (at a designated host:port) to recieve 
        information from the simulation while also sending commands to the robot """
        listen_thread = threading.Thread(target=self.listen_to_robot)
        listen_thread.start()
        self.random_robot_joystick()
        # send a message to the robot to stop execution    
        # halt_message = (False, time.clock(), 0, 0)
        # self.send(halt_message)
        listen_thread.join()
        # Close communication channel
        self.robot_sender_socket.close()

    def power_off(self):
        if(self.robot_running):
            print(print_colors()["red"], "Connection closed by robot", print_colors()['reset'])
            self.robot_running = False

    """BEGIN socket utils"""

    def send_to_robot(self, commands):
        # Create a TCP/IP socket
        self.robot_sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        server_address = ((self.host, self.port_send))
        # print(self.host, self.port)
        try:
            self.robot_sender_socket.connect(server_address)
        except ConnectionRefusedError: # used to turn off the joystick
            self.power_off()
            exit(1)
        # Send data
        message = str(commands)
        self.robot_sender_socket.sendall(bytes(message, "utf-8"))
        self.robot_sender_socket.close()

    def listen_to_robot(self):
        self.robot_receiver_socket.listen(10)
        self.robot_running = True
        while(self.robot_running):
            connection, client = self.robot_receiver_socket.accept()
            # TODO: allow for buffered data, thus no limit
            data = connection.recv(128)
            # quickly close connection to open up for the next input
            connection.close()
            # NOTE: data is either true or false
            # TODO: use ast.literal_eval instead of eval to
            if(data):
                data = eval(data)
                self.ready_to_send = data
                if(data is False):
                    self.power_off()
                    break
            else:
                break
    
    def establish_robot_sender_connection(self):
        """This is akin to a client connection (joystick is client)"""
        self.robot_sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        robot_address = ((self.host, self.port_send))
        try:
            self.robot_sender_socket.connect(robot_address)
        except:
            print(print_colors()["red"], "Unable to connect to robot", print_colors()['reset'])
            print("Make sure you have a simulation instance running")
            exit(1)
        print(print_colors()["green"], "Joystick->Robot connection established", print_colors()['reset'])
        assert(self.robot_sender_socket is not None)

    def establish_robot_receiver_connection(self):
        """This is akin to a server connection (robot is server)"""
        self.robot_receiver_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.robot_receiver_socket.bind((self.host, self.port_recv))
        # wait for a connection
        self.robot_receiver_socket.listen(1)
        connection, client = self.robot_receiver_socket.accept()
        print(print_colors()["green"],"Robot---->Joystick connection established", print_colors()['reset'])
        return connection, client
    
    """ END socket utils """