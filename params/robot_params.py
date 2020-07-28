from dotmap import DotMap
import numpy as np
import os


def create_params():
    p = DotMap()
    
    # can be any valid port, this is an arbitrary choice
    p.port = 6000
    return p
