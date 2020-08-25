import os
import json
import copy
import numpy as np
import shutil
from dotmap import DotMap
from random import seed, random, randint
import string
import random
import glob
import imageio
import socket
from trajectory.trajectory import SystemConfig

color_orange = '\033[33m'
color_green = '\033[32m'
color_red = '\033[31m'
color_blue = '\033[36m'
color_yellow = '\033[35m'
color_reset = '\033[00m'


def ensure_odd(integer):
    if integer % 2 == 0:
        integer += 1
    return integer


def render_angle_frequency(p):
    """Returns a render angle frequency
    that looks heuristically nice on plots."""
    return int(p.episode_horizon / 25)


def log_dict_as_json(params, filename):
    """Save params (either a DotMap object or a python dictionary) to a file in json format"""
    with open(filename, 'w') as f:
        if isinstance(params, DotMap):
            params = params.toDict()
        param_dict_serializable = _to_json_serializable_dict(
            copy.deepcopy(params))
        json.dump(param_dict_serializable, f, indent=4, sort_keys=True)


def _to_json_serializable_dict(param_dict):
    """ Converts params_dict to a json serializable dict."""
    def _to_serializable_type(elem):
        """ Converts an element to a json serializable type. """
        if isinstance(elem, np.int64) or isinstance(elem, np.int32):
            return int(elem)
        if isinstance(elem, np.ndarray):
            return elem.tolist()
        if isinstance(elem, dict):
            return _to_json_serializable_dict(elem)
        if type(elem) is type:  # elem is a class
            return str(elem)
        else:
            return str(elem)
    for key in param_dict.keys():
        param_dict[key] = _to_serializable_type(param_dict[key])
    return param_dict


def euclidean_dist2(p1, p2):
    diff_x = p1[0] - p2[0]
    diff_y = p1[1] - p2[1]
    return np.sqrt(diff_x**2 + diff_y**2)


def touch(path):
    basedir = os.path.dirname(path)
    if not os.path.exists(basedir):
        os.makedirs(basedir)
    with open(path, 'a'):
        os.utime(path, None)


def natural_sort(l):
    import re
    def convert(text): return int(text) if text.isdigit() else text.lower()
    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def generate_name(max_chars):
    return "".join([
        random.choice(string.ascii_letters + string.digits)
        for n in range(max_chars)
    ])


def conn_recv(connection, buffr_amnt=1024):
    # NOTE: allow for buffered data, thus no limit
    chunks = []
    response_len = 0
    while True:
        chunk = connection.recv(buffr_amnt)
        if chunk == b'':
            break
        chunks.append(chunk)
        response_len += len(chunk)
    data = b''.join(chunks)
    return data, response_len


def plot_agents(ax, ppm, agents_dict, json_key=None, label='Agent', normal_color='bo', collided_color='ro',
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
                      scale=2, scale_units='inches')
            if(plot_start_goal):
                ax.quiver(start_3[0], start_3[1], np.cos(start_3[2]), np.sin(start_3[2]),
                          scale=3, scale_units='inches')
                ax.quiver(goal_3[0], goal_3[1], np.cos(goal_3[2]), np.sin(goal_3[2]),
                          scale=3, scale_units='inches')


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


def mkdir_if_missing(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def delete_if_exists(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def check_dotmap_equality(d1, d2):
    """Check equality on nested map objects that all keys and values match."""
    assert(len(set(d1.keys()).difference(set(d2.keys()))) == 0)
    equality = [True] * len(d1.keys())
    for i, key in enumerate(d1.keys()):
        d1_attr = getattr(d1, key)
        d2_attr = getattr(d2, key)
        if type(d1_attr) is DotMap:
            equality[i] = check_dotmap_equality(d1_attr, d2_attr)
    return np.array(equality).all()


def configure_plotting():
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')


def subplot2(plt, Y_X, sz_y_sz_x=(10, 10), space_y_x=(0.1, 0.1), T=False):
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    hspace, wspace = space_y_x
    plt.rcParams['figure.figsize'] = (X * sz_x, Y * sz_y)
    fig, axes = plt.subplots(Y, X, squeeze=False)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if T:
        axes_list = axes.T.ravel()[::-1].tolist()
    else:
        axes_list = axes.ravel()[::-1].tolist()
    return fig, axes, axes_list


""" BEGIN configs functions """


def generate_config_from_pos_3(pos_3, dt=0.1, speed=0):
    pos_n11 = np.array([[[pos_3[0], pos_3[1]]]], dtype=np.float32)
    heading_n11 = np.array([[[pos_3[2]]]], dtype=np.float32)
    speed_nk1 = np.ones((1, 1, 1), dtype=np.float32) * speed
    return SystemConfig(dt, 1, 1,
                        position_nk2=pos_n11,
                        heading_nk1=heading_n11,
                        speed_nk1=speed_nk1,
                        variable=False)


def generate_random_config(environment, dt=0.1,
                           max_vel=0.6, radius=5.):
    pos_3 = generate_random_pos_in_environment(environment, radius)
    return generate_config_from_pos_3(pos_3, dt=dt, speed=max_vel)

# For generating positional arguments in an environment


def generate_random_pos_3(center, xdiff=3, ydiff=3):
    """
    Generates a random position near the center within an elliptical radius of xdiff and ydiff
    """
    offset_x = 2 * xdiff * random.random() - xdiff  # bound by (-xdiff, xdiff)
    offset_y = 2 * ydiff * random.random() - ydiff  # bound by (-ydiff, ydiff)
    offset_theta = 2 * np.pi * random.random()  # bound by (0, 2*pi)
    return np.add(center, np.array([offset_x, offset_y, offset_theta]))


def within_traversible(new_pos: np.array, traversible: np.array, map_scale: float,
                       stroked_radius: bool = False):
    """
    Returns whether or not the position is in a valid spot in the
    traversible
    """
    pos_x = int(new_pos[0] / map_scale)
    pos_y = int(new_pos[1] / map_scale)
    # Note: the traversible is mapped unintuitively, goes [y, x]
    if (not traversible[pos_y][pos_x]):  # Looking for invalid spots
        return False
    return True


def within_traversible_with_radius(new_pos: np.array, traversible: np.array, map_scale: float, radius: int = 1,
                                   stroked_radius: bool = False):
    """
    Returns whether or not the position is in a valid spot in the
    traversible the Radius input can determine how many surrounding
    spots must also be valid
    """
    for i in range(2 * radius):
        for j in range(2 * radius):
            if(stroked_radius):
                if not((i == 0 or i == radius - 1 or j == 0 or j == radius - 1)):
                    continue
            pos_x = int(new_pos[0] / map_scale) - radius + i
            pos_y = int(new_pos[1] / map_scale) - radius + j
            # Note: the traversible is mapped unintuitively, goes [y, x]
            if (not traversible[pos_y][pos_x]):  # Looking for invalid spots
                return False
    return True


def generate_random_pos_in_environment(environment: dict, radius: int = 5):
    """
    Generate a random position (x : meters, y : meters, theta : radians)
    and near the 'center' with a nearby valid goal position.
    - Note that the obstacle_traversible and human_traversible are both
    checked to generate a valid pos_3.
    - Note that the "environment" holds the map scale and all the
    individual traversibles
    - Note that the map_scale primarily refers to the traversible's level
    of precision, it is best to use the dx_m provided in examples.py
    """
    map_scale = float(environment["map_scale"])
    center = np.array(environment["room_center"])
    # Combine the occupancy information from the static map
    # and the human
    if len(environment["traversibles"]) > 1:
        global_traversible = np.empty(environment["traversibles"][0].shape)
        global_traversible.fill(True)
        for t in environment["traversibles"]:
            # add 0th and all others that match shape
            if t.shape == environment["traversibles"][0].shape:
                global_traversible = np.stack([global_traversible, t], axis=2)
                global_traversible = np.all(global_traversible, axis=2)
    else:
        global_traversible = environment["traversibles"][0]

    # Generating new position as human's position
    pos_3 = np.array([-1, -1, 0])  # start far out of the traversible

    # continuously generate random positions near the center until one is valid
    while not within_traversible(pos_3, global_traversible, map_scale):
        pos_3 = generate_random_pos_3(center, radius, radius)

    # Random theta from 0 to pi
    pos_3[2] = random.random() * 2 * np.pi

    return pos_3


""" END configs functions """
