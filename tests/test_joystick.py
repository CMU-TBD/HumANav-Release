import threading
import socket
import time
from simulators.joystick import Joystick


def test_joystick():
    J = Joystick()
    J.establish_robot_sender_connection()
    J.establish_robot_receiver_connection()
    J.await_episodes()
    episodes = J.get_episodes()
    for ep in episodes:
        J.await_env()
        # init control pipeline after episode from robot
        J.init_control_pipeline()
        J.update(random_commands=False)
        # for other episode instance


if __name__ == '__main__':
    test_joystick()
