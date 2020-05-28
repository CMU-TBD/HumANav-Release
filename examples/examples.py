import matplotlib.pyplot as plt
import numpy as np
import os, sys, math
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
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
def plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, human_traversible, dx_m, camera_pos_13, humans_pos_3, human_goal_3, human_speed, time, filename):

    # Compute the real_world extent (in meters) of the traversible
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent)*dx_m

    fig = plt.figure(figsize=(30, 10))

    # Plot the 5x5 meter occupancy grid centered around the camera
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(traversible, extent=extent, cmap='gray',
              vmin=-.5, vmax=1.5, origin='lower')

    # Plot the 5x5 meter human radius grid atop the environment traversible
    alphas = np.empty(np.shape(human_traversible))
    for y in range(human_traversible.shape[1]):
            for x in range(human_traversible.shape[0]):
                alphas[x][y] = not(human_traversible[x][y])
    ax.imshow(human_traversible, extent=extent, cmap='autumn_r',
              vmin=-.5, vmax=1.5, origin='lower', alpha = alphas)
    alphas = np.all(np.invert(human_traversible))
    # Plot the camera
    ax.plot(camera_pos_13[0, 0], camera_pos_13[0, 1], 'bo', markersize=10, label='Camera')
    ax.quiver(camera_pos_13[0, 0], camera_pos_13[0, 1], np.cos(camera_pos_13[0, 2]), np.sin(camera_pos_13[0, 2]))
    
    # Plot the humans (added support for multiple humans)
    for i, human_pos in enumerate(humans_pos_3):
        if(i == 0):
            ax.plot(human_pos[0], human_pos[1], 'ro', markersize=10, label='Human')
        else:
            ax.plot(human_pos[0], human_pos[1], 'ro', markersize=10) #no label
        ax.quiver(human_pos[0], human_pos[1], np.cos(human_pos[2]), np.sin(human_pos[2]), scale=(1-human_speed[i])*4+1, scale_units='inches')
    
    # Plot the human goals
    for i, pos_3 in enumerate(human_goal_3):
        if(i == 0):
            ax.plot(pos_3[0], pos_3[1], 'go', markersize=10, label='Goal')
        else:
            ax.plot(pos_3[0], pos_3[1], 'go', markersize=10) #no label
    
    # Drawing traversible (for debugging)
    debug_drawing = False
    if(debug_drawing):
        for y in range(int(traversible.shape[1]/2.)):
            for x in range(int(traversible.shape[0]/2.)):
                if(traversible[2*x][2*y]):
                    ax.plot(2*y*dx_m, 2*x*dx_m, 'go', markersize=2)
                else:
                    ax.plot(2*y*dx_m, 2*x*dx_m, 'ro', markersize=2)
    
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
    local_path = "time" + str(time)
    mkdir_p(local_path)
    fig.savefig(os.path.join(local_path, filename), bbox_inches='tight', pad_inches=0)
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

def within_traversible(new_pos, traversible, dx_m, radius = 1, stroked_radius = False):
    # Returns whether or not the position is in a valid spot in the traversible
    # the Radius input can determine how many surrounding spots must also be valid
    for i in range(2*radius):
        for j in range(2*radius):
            if(stroked_radius):
                if not((i == 0 or i == radius - 1 or j == 0 or j == radius - 1)):
                    continue;
            pos_x = int(new_pos[0]/dx_m) - radius + i
            pos_y = int(new_pos[1]/dx_m) - radius + j
            # Note: the traversible is mapped unintuitively, goes [y, x]
            if (not traversible[pos_y][pos_x]): # Looking for invalid spots
                return False
    return True

def example1(num_humans):
    """
    Code for loading a random human into the environment
    and rendering topview, rgb, and depth images.
    """
    p = create_params() # used to instantiate the camera and its parameters

    r = HumANavRenderer.get_renderer(p)#get the renderer from the camera p
    dx_cm, traversible = r.get_config()#obtain "resolution and traversible of building"
    human_traversible = np.empty(traversible.shape)
    human_traversible.fill(True) #initially all good
    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    # Camera (robot) position modeled as (x, y, theta) in 2D array
    # Multiple entries yield multiple shots
    camera_pos_13 = np.array([
        [7.5, 12., -1.3], # middle view
        [8, 9, 1.7],  # bottom-up view
        [5.5, 11.5, 0.1],       # left-right view
        [11, 11.5, 3.2]   # right-left view
    ])

    # Add surrounding boundary dots to camer's so generated humans won't interfere
    num_cameras = np.shape(camera_pos_13)[0]
    for i in range(num_cameras):
        num_dots = 5
        skip = 3
        for j in range(num_dots):
            for k in range(num_dots):
                if (j == 0 or j == num_dots - 1 or k == 0 or k == num_dots - 1):
                    camera_x = int(camera_pos_13[i][0]/dx_m) - int(skip/2.*num_dots) + skip*j
                    camera_y = int(camera_pos_13[i][1]/dx_m) - int(skip/2.*num_dots) + skip*k
                    traversible[camera_y][camera_x] = False
    # In order to print more readable arrays
    np.set_printoptions(precision = 2)

    # Output position of new camera renders
    for i in range(num_cameras):
        print("Rendering camera (robot) at", camera_pos_13[i])
    
    identity_rng = []
    mesh_rng = []
    human_pos_3 = []
    human_goal_3 = []
    human_speed = []
    time = 0
    for i in range(num_humans):
        # Set the identity seed. This is used to sample a random human identity
        # (gender, texture, body shape)
        identity_rng.append(np.random.RandomState(randint(10, 100)))

        # Set the Mesh seed. This is used to sample the actual mesh to be loaded
        # which reflects the pose of the human skeleton.
        mesh_rng.append(np.random.RandomState(randint(10, 100)))

        # State of the camera and the human. 
        # Specified as [x (meters), y (meters), theta (radians)] coordinates
        new_pos_3 = np.array([-1, -1, 0])# start far out of the traversible
        while(not within_traversible(new_pos_3, traversible, dx_m, 3) or 
              not within_traversible(new_pos_3, human_traversible, dx_m, 3)):
            new_pos_3 = generate_random_pos_3(camera_pos_13[0], 3, 3);
        human_pos_3.append(new_pos_3)

        # Generating new position as human's goal (endpoint)
        new_pos_3 = np.array([-1, -1, 0])# start far out of the traversible
        while(not within_traversible(new_pos_3, traversible, dx_m, 3) or 
              not within_traversible(new_pos_3, human_traversible, dx_m, 3)):
            new_pos_3 = generate_random_pos_3(human_pos_3[i], 1.5, 1.5);
        human_goal_3.append(new_pos_3)

        # Update human i's angle to point towards the goal
        diff_x = human_goal_3[i][0] - human_pos_3[i][0]
        diff_y = human_goal_3[i][1] - human_pos_3[i][1]
        human_pos_3[i][2] = math.atan2(diff_y, diff_x)

        # Generating random speed of the human in m/s
        human_speed.append(random())# random value from 0 to 1

        print("Human", i, "at", human_pos_3[i], "& goal", human_goal_3[i][:2], "& speed", round(human_speed[i], 3), "m/s")
        
        # Load a random human at a specified state and speed
        r.add_human_at_position_with_speed(human_pos_3[i], human_speed[i], identity_rng[i], mesh_rng[i], i)
        human_traversible = r.get_human_traversible() #update human traversible

    # Get information about which mesh was loaded
    human_mesh_info = r.human_mesh_params

    # Plotting an image for each camera location
    for i in range(num_cameras):
        rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, np.array([camera_pos_13[i]]), dx_m, human_visible=True)

        # Plot the rendered images
        plot_images(rgb_image_1mk3, depth_image_1mk1, traversible, human_traversible, dx_m, np.array([camera_pos_13[i]]), human_pos_3, human_goal_3, human_speed, time, "example1_v" + str(i) + ".png")
    # Remove the human from the environment
    r.remove_human()

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
    #try:
        example1(5) 
        #example2() #not running example2 yet
    #except:
    #    print('\033[31m', "Failed to render image", '\033[0m')
    #    sys.exit(1)
