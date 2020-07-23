import tensorflow as tf
import socket, threading, multiprocessing
import time

class Controller():
    def __init__(self, host=None, port=None):
        self.update_host_port(host, port) # defines class' host and port
        self.establish_robot_connection()
        self.t = 0
        self.latest_state = None
        self.world_state = (False, 0, 0)
        # sockets for communication
        self.robot_socket = None
        print("Initiated controller at", self.host, self.port)

    def set_host(self, h):
        self.host = h

    def set_port(self, p):
        self.port = p

    def random_robot_controller(self):
        from random import randint
        self.world_state = (True, 0, 0)
        accel_scale = 100 # scale to multiply the raw acceleration values by 
        repeat = 2 # number of times to send the same command to the robot
        sent_commands = 0
        while(self.world_state[0] is True):
            lin_command = (randint(10, 100) / 100.) # robot can only more forwards
            ang_command = (randint(-100, 100) / 100.)
            # print(lin_command, ang_command)
            for _ in range(repeat):
                if(sent_commands is 200):
                    self.world_state[0] = False
                message = (self.world_state[0], self.world_state[1], lin_command, ang_command)
                self.send(message)
                sent_commands += 1
            # random delay for the monkey to input commands
            time.sleep(0.1*randint(0,100)/100.)
        # Close communication channel
        self.robot_socket.close()

    def update(self):
        """ Independent process for a user (at a designated host:port) to recieve 
        information from the simulation while also sending commands to the robot """
        # listen_thread = threading.Thread(target=self.listen, args=(None,None))
        # listen_thread.start()
        self.random_robot_controller()
        # send a message to the robot to stop execution    
        halt_message = (False, time.clock(), 0, 0)
        self.send(halt_message)
        # listen_thread.join()

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

    def serialize(self, data):
        """Serialize a data object into something that can be pickled."""
        # TODO: find a way to serialize tf objects (JSON?)
        return str(data)

    def unserialize(self, data):
        # TODO: use ast.literal_eval instead
        return eval(data)

    def send(self, commands, host=None, port=None):
        # Send data
        message = self.serialize(commands)
        self.robot_socket.sendall(bytes(message, "utf-8"))

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
    
    def establish_robot_connection(self):
        """This is akin to a client connection (robot is client)"""
        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.update_host_port(None, 6000)
        robot_address = ((self.host, self.port))
        self.robot_socket.connect(robot_address)
        print("Connection to robot established")

    """ END socket utils """