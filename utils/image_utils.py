import os
import numpy as np
from utils.utils import *


def plot_image_observation(ax, img_mkd, size=None):
    """
    Plot an image observation (occupancy_grid, rgb, or depth).
    The image to be plotted is img_mkd an mxk image with d channels.
    """
    if img_mkd.shape[2] == 1:  # plot an occupancy grid image
        ax.imshow(img_mkd[:, :, 0], cmap='gray', extent=(
            0, size, -1 * size / 2.0, size / 2.0))
    elif img_mkd.shape[2] == 3:  # plot an rgb image
        ax.imshow(img_mkd.astype(np.int32))
        ax.grid(False)
    else:
        raise NotImplementedError


def plot_agents(ax, ppm: float, agents_dict: dict, json_key: str = None, label='Agent', normal_color='bo', collided_color='ro',
                plot_trajectory=True, plot_quiver=False, plot_start_goal=False, start_3=None, goal_3=None):
    # plot all the simulated prerecorded gen_agents
    for i, a in enumerate(agents_dict.values()):
        if(json_key is not None):
            # when plotting from JSON serialized gen_agents
            collided = a["collided"]
            markersize = a["radius"] * ppm
            pos_3 = a[json_key]
            traj_col = a["color"]
        else:
            collided = a.get_collided()
            markersize = a.get_radius() * ppm
            pos_3 = a.get_current_config().to_3D_numpy()
            traj_col = a.get_color()
            if(plot_start_goal):
                start_3 = a.get_start_config().to_3D_numpy()
                goal_3 = a.get_goal_config().to_3D_numpy()
        if(plot_start_goal):
            assert(start_3 is not None)
            assert(goal_3 is not None)
        start_goal_markersize = markersize * 0.7
        if(plot_trajectory):
            a.get_trajectory().render(ax, freq=1, color=traj_col, plot_quiver=False)
        color = normal_color  # gen_agents are green and solid unless collided
        start_goal_col = 'wo'  # white circle
        if(collided):
            color = collided_color  # collided gen_agents are drawn red
        if(i == 0):
            # Only add label on the first humans
            ax.plot(pos_3[0], pos_3[1], color,
                    markersize=markersize, label=label)
            if(plot_start_goal):
                ax.plot(start_3[0], start_3[1], start_goal_col,
                        markersize=start_goal_markersize, label=label + " start")
                ax.plot(goal_3[0], goal_3[1], start_goal_col,
                        markersize=start_goal_markersize, label=label + " goal")
        else:
            ax.plot(pos_3[0], pos_3[1], color,
                    markersize=markersize)
            if(plot_start_goal):
                ax.plot(start_3[0], start_3[1], start_goal_col,
                        markersize=start_goal_markersize)
                ax.plot(goal_3[0], goal_3[1], start_goal_col,
                        markersize=start_goal_markersize)
        # plot the surrounding "force field" around the agent
        ax.plot(pos_3[0], pos_3[1], color,
                alpha=0.2, markersize=2. * markersize)
        if(plot_quiver):
            # Agent heading
            ax.quiver(pos_3[0], pos_3[1], np.cos(pos_3[2]), np.sin(pos_3[2]),
                      scale=int(0.066 * ppm), scale_units='inches')
            if(plot_start_goal):
                ax.quiver(start_3[0], start_3[1], np.cos(start_3[2]), np.sin(start_3[2]),
                          scale=int(0.07 * ppm), scale_units='inches')
                ax.quiver(goal_3[0], goal_3[1], np.cos(goal_3[2]), np.sin(goal_3[2]),
                          scale=int(0.07 * ppm), scale_units='inches')


def render_rgb_and_depth(r, camera_pos_13, dx_m: float, human_visible=True):
    """render the rgb and depth images from the openGL renderer

    Args:
        r: the openGL renderer object
        camera_pos_13: the 3D (x, y, theta) position of the camera
        dx_m (float): the delta_x in meters between real world and grid units
        human_visible (bool, optional): Whether or not the humans are drawn. Defaults to True.

    Returns:
        rgb_image_1mk3, depth_image_1mk1: the rgb and depth images respectively
    """

    # Convert from real world units to grid world units
    camera_grid_world_pos_12 = camera_pos_13[:, :2] / dx_m

    # Render RGB and Depth Images. The shape of the resulting
    # image is (1 (batch), m (width), k (height), c (number channels))
    rgb_image_1mk3 = r._get_rgb_image(
        camera_grid_world_pos_12, camera_pos_13[:, 2:3], human_visible=True)

    depth_image_1mk1, _, _ = r._get_depth_image(
        camera_grid_world_pos_12, camera_pos_13[:, 2:3], xy_resolution=.05,
        map_size=1500, pos_3=camera_pos_13[0, :3], human_visible=True)

    return rgb_image_1mk3, depth_image_1mk1


def save_to_gif(IMAGES_DIR, duration=0.05, gif_filename="movie", clear_old_files=True, verbose=False):
    """Takes the image directory and naturally sorts the images into a singular movie.gif"""
    images = []
    if(not os.path.exists(IMAGES_DIR)):
        print('\033[31m', "ERROR: Failed to image directory at",
              IMAGES_DIR, '\033[0m')
        os._exit(1)  # Failure condition
    files = natural_sort(glob.glob(os.path.join(IMAGES_DIR, '*.png')))
    num_images = len(files)
    for i, filename in enumerate(files):
        if(verbose):
            print("appending", filename)
        try:
            images.append(imageio.imread(filename))
        except:
            print("%sUnable to read file:" % (color_red), filename,
                  "Try clearing the directory of old files and rerunning%s" % (color_reset))
            exit(1)
        print("Movie progress:", i, "out of", num_images, "%.3f" %
              (i / num_images), "\r", end="")
    output_location = os.path.join(IMAGES_DIR, gif_filename + ".gif")
    kargs = {'duration': duration}  # 1/fps
    imageio.mimsave(output_location, images, 'GIF', **kargs)
    print("%sRendered gif at" % (color_green), output_location, '\033[0m')
    # Clearing remaining files to not affect next render
    if clear_old_files:
        for f in files:
            os.remove(f)
