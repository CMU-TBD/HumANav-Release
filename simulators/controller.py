import tensorflow as tf
import socket, threading, multiprocessing
import time

class Controller():
    def __init__(self, robot = None, host=None, port=None):
        self.default_port = 6000 # can change this to be whatever you want
        self.update_host_port(host, port) # defines class' host and port
        self.robot = robot
        self.t = 0
        self.latest_state = None
        self.world_state = (False, 0, 0)
        print("Initiated controller at", self.host, self.port)

    def set_robot(self, r):
        self.robot = r

    def set_host(self, h):
        self.host = h

    def set_port(self, p):
        self.port = p

    def update_host_port(self, host, port):
        # Define host
        if(host is None):
            self.host = socket.gethostname()
        else:
            self.host = host
        # Define the communication port
        if (port is None):
            self.port = self.default_port
        else:
            self.port = port

    def serialize(self, data):
        """Serialize a data object into something that can be pickled."""
        # TODO: find a way to serialize tf objects (JSON?)
        return str(data)

    def unserialize(self, data):
        # TODO: use ast.literal_eval instead
        return eval(data)

    def send(self, simulation_info, host=None, port=None):
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # update self's host and port
        self.update_host_port(host, port)
        # Connect the socket to the port where the server is listening
        server_address = ((self.host, self.port))
        # print(self.host, self.port)
        client_socket.connect(server_address)
        # Send data
        client_socket.sendall(bytes(self.serialize(simulation_info), "utf-8"))
        # Close communication channel
        client_socket.close()
        # TODO: needs better synchronization

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
                self.world_state = self.unserialize(data)
                break
        s.close()

    def random_robot_controller(self):
        from random import randint
        self.world_state = (True, 0, 0)
        accel_scale = 100 # scale to multiply the raw acceleration values by 
        repeat = 1 # number of times to send the same command to the robot
        while(self.world_state[0] is True):
            # TODO: not really velocity, this is acceleration
            lin_command = accel_scale * 0.6 * (randint(-100, 100) / 100.)
            ang_command = accel_scale * 1.1 * (randint(-100, 100) / 100.)
            # print(lin_command, ang_command)
            # for _ in range(repeat):
            self.robot.send_commands((self.world_state[0], self.world_state[1], lin_command, ang_command))
            # random delay for the monkey to input commands
            time.sleep(0.5*randint(0,100)/100.)

    def update(self):
        """ Independent process for a user (at a designated host:port) to recieve 
        information from the simulation while also sending commands to the robot """
        listen_thread = threading.Thread(target=self.listen, args=(None,None))
        listen_thread.start()
        self.random_robot_controller()
        self.robot.send_commands((False, time.clock(), 0, 0))
        listen_thread.join()