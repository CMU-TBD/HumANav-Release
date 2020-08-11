import tensorflow as tf
import socket
import threading
import multiprocessing
import threading
import time
import sys
import json
import os
from copy import deepcopy
from random import randint
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # for rendering without a display
import matplotlib.pyplot as plt
from utils.utils import print_colors, conn_recv, touch, save_to_gif, plot_agents
from params.robot_params import create_params
from params.renderer_params import get_path_to_humanav


class Joystick():
    def __init__(self):
        self.t = 0
        self.latest_state = None
        self.world_state = []
        self.environment = None
        self.params = create_params()
        # sockets for communication
        self.robot_sender_socket = None
        self.robot_running = False
        self.host = socket.gethostname()
        self.port_send = self.params.port  # port for sending commands to the robot
        self.port_recv = self.port_send + 1  # port for recieving commands from the robot
        self.frame_num = 0
        self.ready_to_send = False
        self.ready_to_req = False  # True whenever the joystick wants data about the world
        print("Initiated joystick at", self.host, self.port_send)

    def set_host(self, h):
        self.host = h

    def random_robot_joystick(self):
        repeat = 1          # number of times to send the same command to the robot
        sent_commands = 0
        self.robot_running = True
        while(self.robot_running is True):
            try:
                if(self.ready_to_send and self.environment is not None):
                    # robot can only more forwards
                    lin_command = (randint(10, 100) / 100.)
                    ang_command = (randint(-100, 100) / 100.)
                    for _ in range(repeat):
                        # TODO: remove robot_running stuff
                        message = (self.robot_running, time.clock(),
                                   lin_command, ang_command, self.ready_to_req)
                        self.send_to_robot(message)
                        print("sent", message)
                        sent_commands += 1
                    # now wait for robot to ping with "ready"
                    time.sleep(0.05)
                    self.ready_to_send = True
                # TODO: create a backlog of commands that were not sent bc the robot wasn't ready
            except KeyboardInterrupt:
                print(print_colors()[
                      "yellow"], "Joystick disconnected by user", print_colors()['reset'])
                # send message to turn off the robot
                self.send_to_robot((False, time.clock(), 0, 0, False))  # stop
                self.robot_running = False
                break

    def update(self):
        """ Independent process for a user (at a designated host:port) to recieve
        information from the simulation while also sending commands to the robot """
        listen_thread = threading.Thread(target=self.listen_to_robot)
        listen_thread.start()
        self.random_robot_joystick()
        listen_thread.join()
        # Close communication channel
        self.robot_sender_socket.close()
        # begin gif (movie) generation
        try:
            save_to_gif(os.path.join(get_path_to_humanav(), self.dirname))
        except:
            print("unable to render gif")

    def power_off(self):
        if(self.robot_running):
            print(print_colors()[
                  "red"], "Connection closed by robot", print_colors()['reset'])
            self.robot_running = False
            try:
                self.send_to_robot((False, time.clock(), 0, 0, False))  # stop
            except:
                pass

    """BEGIN socket utils"""

    def send_to_robot(self, commands):
        # Create a TCP/IP socket
        # TODO: make this use JSON rather than the current solution
        self.robot_sender_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        server_address = ((self.host, self.port_send))
        try:
            self.robot_sender_socket.connect(server_address)
        except ConnectionRefusedError:  # used to turn off the joystick
            self.power_off()
            return
        # Send data
        message = str(commands)
        self.robot_sender_socket.sendall(bytes(message, "utf-8"))
        self.robot_sender_socket.close()

    def listen_to_robot(self):
        self.robot_receiver_socket.listen(10)
        self.robot_running = True
        while(self.robot_running):
            connection, client = self.robot_receiver_socket.accept()
            data_b, response_len = conn_recv(connection)
            # quickly close connection to open up for the next input
            connection.close()
            print("received", response_len, "bytes from server")
            if(data_b is not None):
                self.ready_to_send = True  # has recieved a world state from the robot
                self.ready_to_req = False
                data_str = data_b.decode("utf-8")  # bytes to str
                self.world_state.append(json.loads(data_str))
                if(self.world_state[-1]['robot_on'] is True):
                    if(self.world_state[-1]['environment']):  # not empty
                        # notify the robot that the joystick received the environment
                        self.send_to_robot((True, -1, 0, 0, False))
                        # only update the environment if it is non-empty
                        self.environment = self.world_state[-1]['environment']
                        print("Updated environment from robot")
                    self.generate_frame(self.frame_num)
                else:
                    print("powering off joystick")
                    self.power_off()
                    break
            else:
                break
            # time.sleep(1)
            # this should be a separate thread
            self.ready_to_req = True

    def generate_frame(self, frame_count, plot_quiver=False):
        # extract the information from the world state
        environment = self.environment
        agents = self.world_state[-1]['agents']
        prerecs = self.world_state[-1]['prerecs']
        robots = self.world_state[-1]['robots']
        sim_time = self.world_state[-1]['sim_t']
        wall_time = self.world_state[-1]['wall_t']
        # process the information
        map_scale = eval(environment["map_scale"])  # float
        room_center = np.array(environment["room_center"])
        traversible = np.array(environment['traversibles'][0])
        # Compute the real_world extent (in meters) of the traversible
        extent = [0., traversible.shape[1], 0., traversible.shape[0]]
        extent = np.array(extent) * map_scale
        # plot the matplot imgs
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(1, 1, 1)
        # plot the img on the axis
        ax.imshow(traversible, extent=extent, cmap='gray',
                  vmin=-.5, vmax=1.5, origin='lower')

        # get number of pixels per meter based off the ax plot space
        img_scale = ax.transData.transform(
            (0, 1)) - ax.transData.transform((0, 0))
        # print(img_scale)
        ppm = img_scale[1]  # number of pixels per "meter" unit in the plot

        # Plot the camera (robots)
        plot_agents(ax, ppm, robots, json_key="current_config", label="Robot",
                    normal_color="bo", collided_color="ko", plot_trajectory=False, plot_quiver=plot_quiver)

        # plot all the simulated prerecorded agents
        plot_agents(ax, ppm, prerecs, json_key="current_config", label="Prerec",
                    normal_color="yo", collided_color="ro", plot_trajectory=False, plot_quiver=plot_quiver)

        # plot all the randomly generated simulated agents
        plot_agents(ax, ppm, agents, json_key="current_config", label="Agent",
                    normal_color="go", collided_color="ro", plot_trajectory=False, plot_quiver=plot_quiver)

        # save the axis to a file
        filename = "jview" + str(frame_count) + ".png"
        self.dirname = 'tests/socnav/joystick_movie'
        full_file_name = os.path.join(
            get_path_to_humanav(), self.dirname, filename)
        if(not os.path.exists(full_file_name)):
            touch(full_file_name)  # Just as the bash command
        fig.savefig(full_file_name, bbox_inches='tight', pad_inches=0)
        # clear matplot from memory
        fig.clear()
        plt.close(fig)
        del fig
        plt.clf()
        # update frame count
        self.frame_num += 1

    def establish_robot_sender_connection(self):
        """This is akin to a client connection (joystick is client)"""
        self.robot_sender_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        robot_address = ((self.host, self.port_send))
        try:
            self.robot_sender_socket.connect(robot_address)
        except:
            print(print_colors()[
                  "red"], "Unable to connect to robot", print_colors()['reset'])
            print("Make sure you have a simulation instance running")
            exit(1)
        print(print_colors()[
              "green"], "Joystick->Robot connection established", print_colors()['reset'])
        assert(self.robot_sender_socket is not None)

    def establish_robot_receiver_connection(self):
        """This is akin to a server connection (robot is server)"""
        self.robot_receiver_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        self.robot_receiver_socket.bind((self.host, self.port_recv))
        # wait for a connection
        self.robot_receiver_socket.listen(1)
        connection, client = self.robot_receiver_socket.accept()
        print(print_colors()[
              "green"], "Robot---->Joystick connection established", print_colors()['reset'])
        return connection, client

    """ END socket utils """
