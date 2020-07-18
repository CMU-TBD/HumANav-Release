import tensorflow as tf
import socket, threading, multiprocessing
import time

class Controller():
    def __init__(self, robot = None, host=None, port=None):
        if(host is None):
            self.host = socket.gethostname()
        if(port is None):
            self.port = 6000 
        self.robot = robot
        self.t = 0
        self.latest_state = None
        self.state = False
        print("Initiated controller at", self.host, self.port)

    def set_robot(self, r):
        self.robot = r

    def set_host(self, h):
        self.host = h

    def set_port(self, p):
        self.port = p
    
    def encode(self, data):
        assert(isinstance(data, tuple))
        return str(data)

    def decode(self, message):
        assert(isinstance(message, str))
        assert(message[0] is '(')
        results = ()
        value = ""
        for c in message[1:]: # skipping first char '('
            if(c is ',' or c is ')'):
                if value is "True" or value is "False":
                    value = bool(value)
                else:
                    value = float(value)
                results = (*results, value)
                value = ""
            else:
                # append character
                value += c
        return results
    

    def send(self, simulation_info):
        # Create a TCP/IP socket
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        print(self.host, self.port)
        server_address = ((self.host, self.port))
        client_socket.connect(server_address)
        # Send data
        client_socket.sendall(bytes(self.encode(simulation_info), "utf-8"))
        # Close communication channel
        client_socket.close()
        # TODO: needs better synchronization

    def listen(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind((self.host, self.port))
        s.listen(10)
        running = True
        while(running):
            connection, client = s.accept()
            while(True): # constantly taking in information until breaks
                data = connection.recv(128)
                data = self.decode(data.decode('utf-8'))
                print(data, data[0], data[1], data[2])
                connection.close()
                if(data[0] is False):
                    running = False
                break
        s.close()

    def random_robot_controller(self):
        from random import randint
        self.state = True
        while(self.state):
            # latest_state = self.states[self.t]
            # print(self.states)
            # print(self.t)
            lin_vel = 0.6 * (randint(0, 100) / 100.)
            ang_vel = 1.1 * (randint(0, 100) / 100.)
            tf_lin_vel = tf.constant([[[lin_vel]]], dtype=tf.float32)
            tf_ang_vel = tf.constant([[[ang_vel]]], dtype=tf.float32)
            message = tf.concat([tf_lin_vel, tf_ang_vel], 2)
            self.robot.send_commands(message)
            # random delay
            time.sleep(randint(0,100)/100.)