import matplotlib.pyplot as plt
import numpy as np
import os, sys, math
from dotmap import DotMap
from random import seed, random, randint
from humanav import sbpd
from humanav.human import Human
from humanav.humanav_renderer_multi import HumANavRendererMulti
from humanav.renderer_params import create_params as create_base_params
from utils.utils import touch

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
    
def plot_images(p, rgb_image_1mk3, depth_image_1mk1, environment, camera_pos_13, humans, filename):

    map_scale = environment["map_scale"]
    traversible = environment["traversibles"][0] # Obstacles/building traversible
    human_traversible = environment["traversibles"][1]
    # Compute the real_world extent (in meters) of the traversible
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent) * map_scale

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
    for i, human in enumerate(humans):
        if(i == 0):
            ax.plot(human.pos_3[0], human.pos_3[1], 'ro', markersize=10, label='Human')
            ax.plot(human.goal_3[0], human.goal_3[1], 'go', markersize=10, label='Goal')
        else:
            ax.plot(human.pos_3[0], human.pos_3[1], 'ro', markersize=10)
            ax.plot(human.goal_3[0], human.goal_3[1], 'go', markersize=10)
        ax.quiver(human.pos_3[0], human.pos_3[1], np.cos(human.pos_3[2]), np.sin(human.pos_3[2]), scale=2, scale_units='inches')
    
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

    full_file_name = os.path.join(p.humanav_dir, 'tests/humanav', filename)
    if(not os.path.exists(full_file_name)):
        print('\033[31m', "Failed to find:", full_file_name, '\033[33m', "and therefore it will be created", '\033[0m')
        touch(full_file_name) # Just as the bash command
    fig.savefig(full_file_name, bbox_inches='tight', pad_inches=0)
    print('\033[32m', "Successfully rendered:", full_file_name, '\033[0m')

def render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=True):
    # Convert from real world units to grid world units
    camera_grid_world_pos_12 = camera_pos_13[:, :2]/dx_m

    # Render RGB and Depth Images. The shape of the resulting
    # image is (1 (batch), m (width), k (height), c (number channels))
    rgb_image_1mk3 = r._get_rgb_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=True)

    depth_image_1mk1, _, _ = r._get_depth_image(camera_grid_world_pos_12, camera_pos_13[:, 2:3], xy_resolution=.05, map_size=1500, pos_3=camera_pos_13[0, :3], human_visible=True)

    return rgb_image_1mk3, depth_image_1mk1

def test_1(num_humans):
    """
    Code for loading a random human into the environment
    and rendering topview, rgb, and depth images.
    """
    p = create_params() # used to instantiate the camera and its parameters

    r = HumANavRendererMulti.get_renderer(p)#get the renderer from the camera p

    # Get the surreal dataset for human generation
    surreal_data = r.d

    dx_cm, traversible = r.get_config()#obtain "resolution and traversible of building"
    human_traversible = np.empty(traversible.shape)
    human_traversible.fill(True) #initially all good
    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    # Camera (robot) position modeled as (x, y, theta) in 2D array
    # Multiple entries yield multiple shots
    camera_pos_13 = np.array([
        [10., 20., 0],   # middle of the corridor
        [15.,17., -np.pi]   # middle of the corridor

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
    
    humans = []

    # Create default environment which is a dictionary
    # containing ["map_scale", "traversibles"]
    # which is a constant and list of traversibles respectively
    environment = {}
    environment["map_scale"] = dx_m
    # obstacle traversible / human traversible
    environment["traversibles"] = (traversible, human_traversible) 
    for i in range(num_humans):
        # Generates a random human from the environment
        humans.append(Human.generate_random_human_from_environment(Human, surreal_data, environment, camera_pos_13[0]))

        # Load a random human at a specified state and speed
        r.add_human_at_position_with_speed(humans[i])
        environment["traversibles"] = (traversible, r.get_human_traversible()) #update human traversible

    # Get information about which mesh was loaded
    human_mesh_info = r.human_mesh_params
    
    # Plotting an image for each camera location
    for i in range(num_cameras):
        rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, np.array([camera_pos_13[i]]), dx_m, human_visible=True)

        # Plot the rendered images
        plot_images(p, rgb_image_1mk3, depth_image_1mk1, environment, np.array([camera_pos_13[i]]), humans, "example1_v" + str(i) + ".png")

    # Remove all the humans from the environment
    r.remove_all_humans()

def test_2(num_humans = 1):
    """
    Code for loading a specified human identity into the environment
    and rendering topview, rgb, and depth images.
    - Note: Example 2 is expected to produce the same output as Example1 
    from before the multi-human redux
    """
    p = create_params()

    r = HumANavRendererMulti.get_renderer(p)
    dx_cm, traversible = r.get_config()
    human_traversible = np.empty(traversible.shape)
    human_traversible.fill(True) #initially all good

    # Get the surreal dataset for human generation
    surreal_data = r.d

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    # Camera (robot) position modeled as (x, y, theta) in 2D array
    # Multiple entries yield multiple shots
    camera_pos_13 = np.array([
        [12., 18., -1.3],   # middle view
        [15, 25, 4.5],        # bottom-up view
    ])
    num_cameras = np.shape(camera_pos_13)[0]

    humans = []

    # Create default environment which is a dictionary
    # containing ["map_scale", "traversibles"]
    # which is a constant and list of traversibles respectively
    environment = {}
    environment["map_scale"] = dx_m
    # obstacle traversible / human traversible
    environment["traversibles"] = (traversible, human_traversible) 



    for i in range(num_humans):
            # generate new human from known identification/mesh information
            (name, gender, texture, shape) = Human.create_random_human_identity(Human, surreal_data, np.random.RandomState(randint(1, 1000)))
            humans.append(Human.generate_human_with_known_identity(Human, name, gender, texture, shape, 
            environment, camera_pos_13[0]))

            # Load a random human at a specified state and speed
            r.add_human_at_position_with_speed(humans[i])
            environment["traversibles"] = (traversible, r.get_human_traversible()) #update human traversible

    # Get information about which mesh was loaded
    human_mesh_info = r.human_mesh_params
    
    # Plotting an image for each camera location
    for i in range(num_cameras):
        rgb_image_1mk3, depth_image_1mk1 = render_rgb_and_depth(r, np.array([camera_pos_13[i]]), dx_m, human_visible=True)

        # Plot the rendered images
        plot_images(p, rgb_image_1mk3, depth_image_1mk1, environment, np.array([camera_pos_13[i]]), humans, "example2_v" + str(i) + ".png")

    # Remove the human from the environment
    r.remove_all_humans()

if __name__ == '__main__':
    test_1(2) 
    test_2(1)