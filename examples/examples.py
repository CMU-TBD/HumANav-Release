import matplotlib.pyplot as plt
import numpy as np
import os, sys
from dotmap import DotMap
from humanav.humanav_renderer import HumANavRenderer
from humanav.renderer_params import create_params as create_base_params
from humanav.renderer_params import get_surreal_texture_dir
from random import seed 
from random import random
from random import randint

def create_params():
    p = create_base_params()

	# Set any custom parameters
    p.building_name = 'area3'

    p.camera_params.width = 1024
    p.camera_params.height = 1024
    p.camera_params.fov_vertical = 75.
    p.camera_params.fov_horizontal = 75.

    # The camera is assumed to be mounted on a robot at fixed height
    # and fixed pitch. See humanav/renderer_params.py for more information

    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.camera_elevation_degree = -10

    p.camera_params.modalities = ['rgb', 'disparity']
    return p

def plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m, camera_pos_13, humans_pos_3, filename):

    # Compute the real_world extent (in meters) of the traversible
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent)*dx_m

    fig = plt.figure(figsize=(30, 10))

    # Plot the 5x5 meter occupancy grid centered around the camera
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(traversible, extent=extent, cmap='gray',
              vmin=-.5, vmax=1.5, origin='lower')

    # Plot the camera
    ax.plot(camera_pos_13[0, 0], camera_pos_13[0, 1], 'bo', markersize=10, label='Camera')
    ax.quiver(camera_pos_13[0, 0], camera_pos_13[0, 1], np.cos(camera_pos_13[0, 2]), np.sin(camera_pos_13[0, 2]))
    
    # Plot the humans (added support for multiple humans)
    for human_pos in humans_pos_3:
        ax.plot(human_pos[0], human_pos[1], 'ro', markersize=10, label='Human')
        ax.quiver(human_pos[0], human_pos[1], np.cos(human_pos[2]), np.sin(human_pos[2]))

    ax.legend()
    ax.set_xlim([camera_pos_13[0, 0]-5., camera_pos_13[0, 0]+5.])
    ax.set_ylim([camera_pos_13[0, 1]-5., camera_pos_13[0, 1]+5.])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Topview')

    # Plot the RGB Image
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('RGB')

    # Plot the Depth Image
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(depth_image_1mk1[0, :, :, 0].astype(np.uint8), cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Depth')

    fig.savefig(filename, bbox_inches='tight', pad_inches=0)
    print('\033[32m', "Successfully rendered image:", filename, '\033[0m')

def render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=True):
    # Convert from real world units to grid world units
    camera_grid_world_pos_12 = camera_pos_13[:, :2]/dx_m

    # Render RGB and Depth Images. The shape of the resulting
    # image is (1 (batch), m (width), k (height), c (number channels))
    rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=True)

    depth_image_1mk1, _, _ = r._get_depth_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], xy_resolution=.05, map_size=1500, pos_3=camera_pos_13[0, :3], human_visible=True)

    return rgb_image_1mk3, depth_image_1mk1

def generate_random_pos_3(center, xdiff = 3, ydiff = 3):
    # Generates a random position near the center within an elliptical radius of xdiff and ydiff
    offset_x = 2*xdiff * random() - xdiff #bound by (-xdiff, xdiff)
    offset_y = 2*ydiff * random() - ydiff #bound by (-ydiff, ydiff)
    offset_theta = 2 * np.pi * random()    #bound by (0, 2*pi)
    return np.add(center, np.array([offset_x, offset_y, offset_theta]))
        

def example1(num_humans):
    """
    Code for loading a random human into the environment
    and rendering topview, rgb, and depth images.
    """
    p = create_params() # used to instantiate the camera and its parameters

    r = HumANavRenderer.get_renderer(p)#get the renderer from the camera p
    dx_cm, traversible = r.get_config()#optain "resolution and traversible of building"

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    # Camera (robot) position modeled as (x, y, theta) in 2D array
    # Multiple entries yield multiple shots
    camera_pos_13 = np.array([[7.5, 12., -1.3], [5, 8, 0]]) 
    for i in range(np.shape(camera_pos_13)[0]):#(vertical dimensions)
        print("Rendering camera (robot) at", camera_pos_13[i])
    humans = []#tuple of all the below
    identity_rng = []
    mesh_rng = []
    human_pos_3 = []
    human_speed = []
    for i in range(num_humans):
        # Set the identity seed. This is used to sample a random human identity
        # (gender, texture, body shape)
        identity_rng.append(np.random.RandomState(randint(10, 100)))

        # Set the Mesh seed. This is used to sample the actual mesh to be loaded
        # which reflects the pose of the human skeleton.
        mesh_rng.append(np.random.RandomState(randint(10, 100)))

        # State of the camera and the human. 
        # Specified as [x (meters), y (meters), theta (radians)] coordinates
        human_pos_3.append(generate_random_pos_3(camera_pos_13[0]))
        
        print("Generating human", i, "at", human_pos_3[i])
        # Speed of the human in m/s
        human_speed.append(random())# random value from 0 to 1
        #humans.append(tuple([human_pos_3[i], human_speed[i], identity_rng[i], mesh_rng[i]]))
        # Load a random human at a specified state and speed
        r.add_human_at_position_with_speed(human_pos_3[i], human_speed[i], identity_rng[i], mesh_rng[i], i)

    # Get information about which mesh was loaded
    human_mesh_info = r.human_mesh_params
    rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=True)

    # Remove the human from the environment
    r.remove_human()

    # Plot the rendered images
    plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m, camera_pos_13, human_pos_3, 'example1.png')

def get_known_human_identity(r):
    """
    Specify a known human identity. An identity
    is a dictionary with keys {'human_gender', 'human_texture', 'body_shape'}
    """

    # Method 1: Set a seed and sample a random identity
    identity_rng = np.random.RandomState(48)
    human_identity = r.load_random_human_identity(identity_rng)

    # Method 2: If you know which human you want to load,
    # specify the params manually (or load them from a file)
    human_identity = {'human_gender': 'male', 'human_texture': [os.path.join(get_surreal_texture_dir(), 'train/male/nongrey_male_0110.jpg')], 'body_shape': 1320}
    return human_identity

def example2():
    """
    Code for loading a specified human identity into the environment
    and rendering topview, rgb, and depth images.
    Note: Example 2 is expected to produce the same output as Example1
    """
    p = create_params()

    r = HumANavRenderer.get_renderer(p)
    dx_cm, traversible = r.get_config()

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    human_identity = get_known_human_identity(r)

    # Set the Mesh seed. This is used to sample the actual mesh to be loaded
    # which reflects the pose of the human skeleton.
    mesh_rng = np.random.RandomState(20)

    # State of the camera and the human. 
    # Specified as [x (meters), y (meters), theta (radians)] coordinates
    camera_pos_13 = np.array([[7.5, 12., -1.3]])
    human_pos_3 = np.array([8.0, 9.75, np.pi/2.])

    # Speed of the human in m/s
    human_speed = 0.7

    # Load a random human at a specified state and speed
    r.add_human_with_known_identity_at_position_with_speed(human_pos_3, human_speed, mesh_rng, human_identity)

    # Get information about which mesh was loaded
    human_mesh_info = r.human_mesh_params

    rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=True)

    # Remove the human from the environment
    r.remove_human()

    # Plot the rendered images
    plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, dx_m,
                camera_pos_13, human_pos_3, 'example2.png')


if __name__ == '__main__':
    try:
        example1(20) 
        #example2() #not running example2 yet
    except:
        print('\033[31m', "Failed to render image", '\033[0m')
        sys.exit(1)
