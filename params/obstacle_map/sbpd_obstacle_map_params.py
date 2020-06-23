from dotmap import DotMap
from obstacles.sbpd_map import SBPDMap
from params.renderer_params import create_params as create_renderer_params
import numpy as np

def create_params():
    p = DotMap()

    # Load the dependencies
    p.renderer_params = create_renderer_params()

    p.obstacle_map = SBPDMap

    # Size of map
    p.map_size_2 = np.array([521, 600]) # Same as for HumANav FMM Map of Area3

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    p.dx = 0.05

    # Origin is always 0,0 for SBPD
    p.map_origin_2 = [0, 0]

    # Threshold distance from the obstacles to sample the start and the goal positions.
    p.sampling_thres = 2

    # Number of grid steps around the start position to use for plotting
    p.plotting_grid_steps = 100
    return p
