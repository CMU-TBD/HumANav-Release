import matplotlib as mpl
mpl.use('Agg') # for rendering without a display
import matplotlib.pyplot as plt
import numpy as np
import os, sys, math, time
from dotmap import DotMap
from random import seed, random, randint
import tensorflow as tf
tf.enable_eager_execution()
# Humanav
from humanav import sbpd
from humans.human import Human
from humans.human_configs import HumanConfigs
from humans.human_appearance import HumanAppearance
from humanav.humanav_renderer_multi import HumANavRendererMulti
from simulators.agent import Agent
# Planner + Simulator:
from simulators.central_simulator import CentralSimulator
from planners.sampling_planner import SamplingPlanner
from params.planner_params import create_params as create_planner_params
from params.simulator.sbpd_simulator_params import create_params as create_sim_params
from params.renderer_params import create_params as create_base_params
# Other
from utils.utils import touch, print_colors

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

    if p.render_with_display:
        # Can only render rgb and depth then host pc has an available display
        p.camera_params.modalities = ['rgb', 'disparity']
    else:
        p.camera_params.modalities = ['occupancy_grid']

    return p

def plot_topview(ax, extent, traversible, human_traversible, camera_pos_13, 
                 humans, plot_quiver=False):
    ax.imshow(traversible, extent=extent, cmap='gray',
              vmin=-.5, vmax=1.5, origin='lower')

    if human_traversible is not None:
        # NOTE: the human radius is only available given the openGL human modeling
        # and rendering, thus p.render_with_display must be True
        # Plot the 5x5 meter human radius grid atop the environment traversible
        alphas = np.empty(np.shape(human_traversible))
        for y in range(human_traversible.shape[1]):
            for x in range(human_traversible.shape[0]):
                alphas[x][y] = not(human_traversible[x][y])
        ax.imshow(human_traversible, extent=extent, cmap='autumn_r',
                vmin=-.5, vmax=1.5, origin='lower', alpha=alphas)
        alphas = np.all(np.invert(human_traversible))

    # Plot the camera
    ax.plot(camera_pos_13[0], camera_pos_13[1],
            'bo', markersize=10, label='Camera')
    ax.quiver(camera_pos_13[0], camera_pos_13[1], np.cos(
        camera_pos_13[2]), np.sin(camera_pos_13[2]))

    # Plot the humans (added support for multiple humans) and their trajectories
    for i, human in enumerate(humans):
        human_pos_2 = human.get_start_config().position_nk2().numpy()[0][0]
        human_heading = (human.get_start_config().heading_nk1().numpy())[0][0]
        human_goal_2 = human.get_goal_config().position_nk2().numpy()[0][0]
        goal_heading = (human.get_goal_config().heading_nk1().numpy())[0][0]

        color = human.get_termination()
        human.get_trajectory().render(ax, freq=1, color=color, plot_quiver=False)
        if(i == 0):
            # Only add label on the first humans
            ax.plot(human_pos_2[0], human_pos_2[1],
                    'ro', markersize=10, label='Human')
            ax.plot(human_goal_2[0], human_goal_2[1], markerfacecolor="#FF7C00",
                    marker='o', markersize=10, label='Goal')
        else:
            ax.plot(human_pos_2[0], human_pos_2[1], 'ro', markersize=10)
            ax.plot(human_goal_2[0], human_goal_2[1],
                    markerfacecolor="#FF7C00", marker='o', markersize=10)
        if(plot_quiver):
            # human start quiver
            ax.quiver(human_pos_2[0], human_pos_2[1], np.cos(human_heading), np.sin(
                human_heading), scale=2, scale_units='inches')
            # goal quiver
            ax.quiver(human_goal_2[0], human_goal_2[1], np.cos(goal_heading), np.sin(
                goal_heading), scale=2, scale_units='inches')


def plot_images(p, rgb_image_1mk3, depth_image_1mk1, environment, room_center,
                camera_pos_13, humans, filename):

    map_scale = environment["map_scale"]
    # Obstacles/building traversible
    traversible = environment["traversibles"][0]
    human_traversible = None

    if len(environment["traversibles"]) > 1:
        human_traversible = environment["traversibles"][1]
    # Compute the real_world extent (in meters) of the traversible
    extent = [0., traversible.shape[1], 0., traversible.shape[0]]
    extent = np.array(extent) * map_scale

    num_frames = 2
    if rgb_image_1mk3 is not None:
        num_frames = num_frames + 1
    if depth_image_1mk1 is not None:
        num_frames = num_frames + 1
    
    img_size = 10
    fig = plt.figure(figsize=(num_frames * img_size, img_size))

    # Plot the 5x5 meter occupancy grid centered around the camera
    zoom = 5.5  # zoom in by a constant amount
    ax = fig.add_subplot(1, num_frames, 1)
    ax.set_xlim([room_center[0] - zoom, room_center[0] + zoom])
    ax.set_ylim([room_center[1] - zoom, room_center[1] + zoom])
    plot_topview(ax, extent, traversible, human_traversible,
                 camera_pos_13, humans, plot_quiver=True)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Topview (zoomed)')

    # Render entire map-view from the top
    # to keep square plot
    outer_zoom = min(traversible.shape[0], traversible.shape[1]) * map_scale
    ax = fig.add_subplot(1, num_frames, 2)
    ax.set_xlim(0., outer_zoom)
    ax.set_ylim(0., outer_zoom)
    plot_topview(ax, extent, traversible,
                 human_traversible, camera_pos_13, humans)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Topview')

    if rgb_image_1mk3 is not None:
        # Plot the RGB Image
        ax = fig.add_subplot(1, num_frames, 3)
        ax.imshow(rgb_image_1mk3[0].astype(np.uint8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('RGB')

    if depth_image_1mk1 is not None:
        # Plot the Depth Image
        ax = fig.add_subplot(1, num_frames, 4)
        ax.imshow(depth_image_1mk1[0, :, :, 0].astype(np.uint8), cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('Depth')

    full_file_name = os.path.join(p.humanav_dir, 'tests/socnav', filename)
    if(not os.path.exists(full_file_name)):
        print('\033[31m', "Failed to find:", full_file_name,
              '\033[33m', "and therefore it will be created", '\033[0m')
        touch(full_file_name)  # Just as the bash command
    fig.savefig(full_file_name, bbox_inches='tight', pad_inches=0)
    print('\033[32m', "Successfully rendered:", full_file_name, '\033[0m')


def render_rgb_and_depth(r, camera_pos_13, dx_m, human_visible=True):
    # Convert from real world units to grid world units
    camera_grid_world_pos_12 = camera_pos_13[:, :2]/dx_m

    # Render RGB and Depth Images. The shape of the resulting
    # image is (1 (batch), m (width), k (height), c (number channels))
    rgb_image_1mk3 = r._get_rgb_image(
        camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=True)

    depth_image_1mk1, _, _ = r._get_depth_image(
        camera_grid_world_pos_12, camera_pos_13[:, 2:3], xy_resolution=.05, 
        map_size=1500, pos_3=camera_pos_13[0, :3], human_visible=True)

    return rgb_image_1mk3, depth_image_1mk1


def test_socnav(num_humans):
    """
    Code for loading a random human into the environment
    and rendering topview, rgb, and depth images.
    """
    p = create_params()  # used to instantiate the camera and its parameters

    # get the renderer from the camera p
    r = HumANavRendererMulti.get_renderer(p)
    # Get the surreal dataset for human generation
    surreal_data = r.d

    # Update the Human's appearance classes to contain the dataset
    HumanAppearance.dataset = surreal_data

    # obtain "resolution and traversible of building"
    dx_cm, traversible = r.get_config()
    human_traversible = np.empty(traversible.shape)
    human_traversible.fill(True)  # initially all good
    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm/100.

    # Camera (robot) position modeled as (x, y, theta) in 2D array
    # Multiple entries yield multiple shots
    camera_pos_13 = np.array([
        [9., 22., -np.pi/4]
    ])

    # Add surrounding boundary dots to camer's so generated humans won't interfere
    num_cameras = np.shape(camera_pos_13)[0]

    # In order to print more readable arrays
    np.set_printoptions(precision=2)

    # Output position of new camera renders
    for i in range(num_cameras):
        print(" Rendering camera (robot) at", camera_pos_13[i])

    # Creating list of to-be humans that will partake in the scene
    human_list = []

    # Create default environment which is a dictionary
    # containing ["map_scale", "traversibles"]
    # which is a constant and list of traversibles respectively
    environment = {}
    environment["map_scale"] = dx_m
    # obstacle traversible / human traversible
    if p.render_with_display:
        environment["traversibles"] = [traversible, human_traversible]
    else:
        environment["traversibles"] = np.array([traversible])
    """
    Creating planner, simulator, and control pipelines for the framework
    of a human trajectory and pathfinding. 
    """

    # Create planner parameters
    # planner_params = create_planner_params()
    sim_params = create_sim_params()
    simulator = CentralSimulator(sim_params, environment, p.humanav_dir, renderer=r)

    """
    Generate the humans and run the simulation on every human
    """
    # Generate the ~center~ of area3 when scaled up 50%
    room_center = np.array([12., 17., 0.])
    for i in range(num_humans):
        # Generates a random human from the environment
        new_human_i = Human.generate_random_human_from_environment(
            Human, environment, room_center, 
            generate_appearance=p.render_with_display, radius=4)
        # Or specify a human's initial configs with a HumanConfig instance
        # Human.generate_human_with_configs(Human, fixed_start_goal)
        human_list.append(new_human_i)

        # Load a random human at a specified state and speed
        # update human traversible
        if p.render_with_display:
            r.add_human_at_position_with_speed(human_list[i])
            environment["traversibles"] = np.array([traversible, r.get_human_traversible()])  
        else:
            environment["traversibles"] = np.array([traversible]) 
        # Input human fields into simulator
        simulator.add_agent(Agent.human_to_agent(Agent, new_human_i))

    # run simulation
    wall_clock_pre_sim = time.clock()
    simulator.simulate()
    # get wall clock time
    wall_clock_time = time.clock() - wall_clock_pre_sim

    print("Simulation took", wall_clock_time, "seconds of wall clock time")
    for i in range(len(human_list)):
        human_list[i].update_trajectory(simulator.agents[i].vehicle_trajectory)
        human_list[i].update_termination(simulator.agents[i].termination_cause)

    # Plotting an image for each camera location
    for i in range(num_cameras):
        rgb_image_1mk3 = None
        depth_image_1mk1 = None
        if p.render_with_display: # only when rendering with opengl
            rgb_image_1mk3, depth_image_1mk1 = \
                render_rgb_and_depth(r, np.array([camera_pos_13[i]]), dx_m, human_visible=True)
        # Plot the rendered images
        plot_images(p, rgb_image_1mk3, depth_image_1mk1, environment, room_center,
                    camera_pos_13[i], human_list, "example1_v" + str(i) + ".png")

    # Remove all the humans from the environment
    if p.render_with_display: # only when rendering with opengl
        r.remove_all_humans()


if __name__ == '__main__':
    test_socnav(2)  # run basic room test with variable # of human
