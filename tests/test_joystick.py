import threading
import socket
import time
from simulators.joystick import Joystick


def test_joystick():
    J = Joystick()
    J.establish_robot_sender_connection()
    J.establish_robot_receiver_connection()
    J.init_control_pipeline()
    # TODO: only run while the simulator has more episodes to run
    while(True):
        J.update(random_commands=False)
        # for other episode instance
        J.listen_thread = threading.Thread(target=J.listen_to_robot)
        J.listen_thread.start()
        # init control pipeline after recieved map from robot
        J.init_control_pipeline()
        J.await_env()


if __name__ == '__main__':
    test_joystick()
