import os
import json
import copy
import numpy as np
import tensorflow as tf
import dotmap
import shutil
from dotmap import DotMap
from random import seed, random, randint
import string
import random
import glob
import imageio
import socket

color_orange = '\033[33m'
color_green = '\033[32m'
color_red = '\033[31m'
color_blue = '\033[36m'
color_yellow = '\033[35m'
color_reset = '\033[00m'


def tf_session_config():
    config = tf.ConfigProto()

    # Allows for memory growth so the process only uses the amount of memory it needs
    config.gpu_options.allow_growth = True

    # Allows for tensors to be copied onto cpu when no cuda gpu kernel is available
    device_policy = tf.contrib.eager.DEVICE_PLACEMENT_SILENT

    tf_config = {'config': config,
                 'device_policy': device_policy}
    return tf_config


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
        if isinstance(params, dotmap.DotMap):
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
        if isinstance(elem, tf.Tensor):
            return elem.numpy().tolist()
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


def euclidean_dist(p1, p2):
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
                plot_trajectory=True, plot_quiver=False):
    # plot all the simulated prerecorded agents
    for i, a in enumerate(agents_dict.values()):
        if(json_key is not None):
            collided = a["collided"]
            markersize = a["radius"] * ppm
            pos_3 = a[json_key]
        else:
            collided = a.get_collided()
            markersize = a.get_radius() * ppm
            pos_3 = a.get_current_config().to_3D_numpy()
        if(plot_trajectory):
            # TODO: make colours of trajectories random rather than hardcoded
            a.get_trajectory().render(ax, freq=1, color=None, plot_quiver=False)
        color = normal_color  # agents are green and solid unless collided
        if(collided):
            color = collided_color  # collided agents are drawn red
        if(i == 0):
            # Only add label on the first humans
            ax.plot(pos_3[0], pos_3[1],
                    color, markersize=markersize, label=label)
        else:
            ax.plot(pos_3[0], pos_3[1], color,
                    markersize=markersize)
        # TODO: use agent radius instead of hardcode
        ax.plot(pos_3[0], pos_3[1], color,
                alpha=0.2, markersize=2. * markersize)
        if(plot_quiver):
            # Agent heading
            ax.quiver(pos_3[0], pos_3[1], np.cos(pos_3[2]), np.sin(pos_3[2]),
                      scale=2, scale_units='inches')


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
    """Check equality on nested dotmap objects that all keys and values match."""
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
