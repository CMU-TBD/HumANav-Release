from dotmap import DotMap
import numpy as np
import os


def create_params():
    p = DotMap()

    # can be any valid port, this is an arbitrary choice
    p.port = 6000

    # in our case, the robot's length/width = 66.8 cm, radius is half of that
    # radius of robot, we are basing the drive of the robot off of a pr2 robot
    # more info here: https://robots.ieee.org/robots/pr2/
    p.radius: float = 0.668 / 2.0  # meters
    return p
