import tensorflow as tf
import socket
import threading
import multiprocessing
import time
import sys
import json
import numpy as np
import matplotlib as mpl
mpl.use('Agg')  # for rendering without a display
import matplotlib.pyplot as plt
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
        self.port_send = self.params.port  # port for sending commands to the robot
        self.port_recv = self.port_send + 1  # port for recieving commands from the robot
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
                    # robot can only more forwards
                    lin_command = (randint(10, 100) / 100.)
                    ang_command = (randint(-100, 100) / 100.)
                    for _ in range(repeat):
                        # TODO: remove robot_running stuff
                        message = (self.robot_running, time.clock(),
                                   lin_command, ang_command)
                        self.send_to_robot(message)
                        print("sent", message)
                        sent_commands += 1
                    # now wait for robot to ping with "ready"
                    self.ready_to_send = False
                # TODO: create a backlog of commands that were not sent bc the robot wasn't ready
            except KeyboardInterrupt:
                print(print_colors()[
                      "yellow"], "Joystick disconnected by user", print_colors()['reset'])
                self.send_to_robot((False, time.clock(), 0, 0))  # stop signal
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
            print(print_colors()[
                  "red"], "Connection closed by robot", print_colors()['reset'])
            self.robot_running = False

    """BEGIN socket utils"""

    def send_to_robot(self, commands):
        # Create a TCP/IP socket
        # TODO: make this use JSON rather than the current jank solution
        self.robot_sender_socket = socket.socket(
            socket.AF_INET, socket.SOCK_STREAM)
        # Connect the socket to the port where the server is listening
        server_address = ((self.host, self.port_send))
        # print(self.host, self.port)
        try:
            self.robot_sender_socket.connect(server_address)
        except ConnectionRefusedError:  # used to turn off the joystick
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
            # NOTE: allow for buffered data, thus no limit
            chunks = []
            response_len = 0
            while True:
                chunk = connection.recv(1024)
                if chunk == b'':
                    break
                chunks.append(chunk)
                response_len += len(chunk)
            data = b''.join(chunks)
            # quickly close connection to open up for the next input
            connection.close()
            # NOTE: data is either true or false
            # TODO: use ast.literal_eval instead of eval to
            print("received", response_len, "from server")
            if(data is not None):
                data_str = data.decode("utf-8")  # bytes to str
                # TODO: only send a single instance of the map since it is MASSIVE
                world_state = json.loads(data_str)
                if(world_state['environment']):  # not empty
                    # only update the environment if it is non-empty
                    self.environment = world_state['environment']
                    print("received environment from robot")
                    # notify the robot that the joystick received the environment
                    self.send_to_robot((True, -1, 0, 0))
                self.agents = world_state['agents']
                self.prerecs = world_state['prerecs']
                self.robots = world_state['robots']
                # for lingering constants
                self.sim_t = world_state['sim_t']
                self.wall_t = world_state['wall_t']
                self.generate_world(self.environment, self.agents,
                                    self.prerecs, self.robots, self.sim_t, self.wall_t)
                exit(1)
                if(isinstance(data, tuple)):
                    self.ready_to_send = data[0]
                    self.world_state = data[1]
                else:
                    self.ready_to_send = data
                if(self.ready_to_send is False):
                    self.power_off()
                    break
            else:
                break

    def generate_world(self, environment, agents, prerecs, robots, sim_time, wall_time, plot_quiver=False):
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
        for i, r in enumerate(robots.values()):
            r_pos_3 = r["current_config"]
            # TODO: trajectory
            # r.get_trajectory().render(ax, freq=1, color=None, plot_quiver=False)
            color = 'bo'  # robots are blue and solid unless collided
            if(r["collided"]):
                color = 'ko'  # collided robots are drawn BLACK
            if i == 0:
                # only add label on first robot
                ax.plot(r_pos_3[0], r_pos_3[1], color,
                        markersize=10, label='Robot')
            else:
                ax.plot(r_pos_3[0], r_pos_3[1], color,
                        markersize=r["radius"] * ppm)
            # visual "bubble" around robot base to stay safe
            ax.plot(r_pos_3[0], r_pos_3[1], color,
                    alpha=0.2, markersize=r["radius"] * 2. * ppm)
            # this is the "camera" robot (with quiver)
            ax.quiver(r_pos_3[0], r_pos_3[1], np.cos(
                r_pos_3[2]), np.sin(r_pos_3[2]))

        # plot all the simulated prerecorded agents
        for i, a in enumerate(prerecs.values()):
            pos_3 = a["current_config"]
            # TODO: make colours of trajectories random rather than hardcoded
            # a.get_trajectory().render(ax, freq=1, color=None, plot_quiver=False)
            color = 'yo'  # agents are green and solid unless collided
            if(a["collided"]):
                color = 'ro'  # collided agents are drawn red
            if(i == 0):
                # Only add label on the first humans
                ax.plot(pos_3[0], pos_3[1],
                        color, markersize=10, label='Prerec')
            else:
                ax.plot(pos_3[0], pos_3[1], color,
                        markersize=a["radius"] * ppm)
            # TODO: use agent radius instead of hardcode
            ax.plot(pos_3[0], pos_3[1], color,
                    alpha=0.2, markersize=a["radius"] * 2.0 * ppm)
            if(plot_quiver):
                # Agent heading
                ax.quiver(pos_3[0], pos_3[1], np.cos(pos_3[2]), np.sin(pos_3[2]),
                          scale=2, scale_units='inches')

        # plot all the randomly generated simulated agents
        for i, a in enumerate(agents.values()):
            pos_3 = a["current_config"]
            # TODO: make colours of trajectories random rather than hardcoded
            # a.get_trajectory().render(ax, freq=1, color=None, plot_quiver=False)
            color = 'go'  # agents are green and solid unless collided
            if(a["collided"]):
                color = 'ro'  # collided agents are drawn red
            if(i == 0):
                # Only add label on the first humans
                ax.plot(pos_3[0], pos_3[1],
                        color, markersize=10, label='Agent')
            else:
                ax.plot(pos_3[0], pos_3[1], color,
                        markersize=a["radius"] * ppm)
            # TODO: use agent radius instead of hardcode
            ax.plot(pos_3[0], pos_3[1], color,
                    alpha=0.2, markersize=a["radius"] * 2. * ppm)
            if(plot_quiver):
                # Agent heading
                ax.quiver(pos_3[0], pos_3[1], np.cos(pos_3[2]), np.sin(pos_3[2]),
                          scale=2, scale_units='inches')

        # save the axis to a file
        fig.savefig("joystick_map.png", bbox_inches='tight', pad_inches=0)

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
