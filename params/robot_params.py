from dotmap import DotMap
import numpy as np
import os


def create_params():
    p = DotMap()

    # can be any valid port, this is an arbitrary choice
    p.port = 6000

    # radius of robot, we are basing the drive of the robot off of a pr2 robot
    # more info here: https://robots.ieee.org/robots/pr2/
    p.radius = 0.668  # meters
    return p
