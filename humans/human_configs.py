from random import seed, random, randint
import random
import string
import math
import numpy as np
import sys
import os
import pickle
import tensorflow as tf
# tf.enable_eager_execution()
from humanav import sbpd
from humanav import depth_utils as du
from humanav import utils
from params.renderer_params import get_surreal_texture_dir
from mp_env import map_utils as mu
from mp_env.render import swiftshader_renderer as sr

from trajectory.trajectory import SystemConfig


class HumanConfigs():

    def __init__(self, start_config, current_config, goal_config):
        self.start_config = start_config
        # note that for generation/initialization functions, the current
        # config will likely be the same as the start_config
        self.current_config = current_config
        self.goal_config = goal_config

    # Getters for the HumanConfigs class
    def get_start_config(self):
        return self.start_config
    
    def get_current_config(self):
        return self.current_config

    def set_current_config(self, current):
        self.current_config = current

    def get_goal_config(self):
        return self.goal_config

    def generate_human_config(self, start_config, current_config, goal_config):
        """
        Sample a new random human from all required features
        return HumanConfigs(start_config, goal_config)
        """
        return HumanConfigs(start_config, current_config, goal_config)

    def generate_config_from_pos_3(self, pos_3, dt=0.1, speed=0):
        pos_n11 = tf.constant([[[pos_3[0], pos_3[1]]]], dtype=tf.float32)
        # range of speed from [0, 0.6)
        initial_linear_velocity = random.random() * speed
        heading_n11 = tf.constant([[[pos_3[2]]]], dtype=tf.float32)
        speed_nk1 = tf.ones((1, 1, 1), dtype=tf.float32) * \
            initial_linear_velocity
        return SystemConfig(dt, 1, 1,
                            position_nk2=pos_n11,
                            heading_nk1=heading_n11,
                            speed_nk1=speed_nk1,
                            variable=False)

    def generate_random_config(self, environment, dt=0.1, 
                               center=np.array([0., 0., 0.]), 
                               max_vel=0.6, radius=5.):
        pos_3 = self.generate_random_pos_in_environment(
            self, center, environment, radius)
        return self.generate_config_from_pos_3(self, pos_3, dt=dt, speed=max_vel)

    def generate_random_human_config_from_start(self, start_config, 
                                                environment, 
                                                center=np.array([0, 0, 0])):
        """
        Generate a human with a random goal config given a known start
        config. The generated start config will be near center by a threshold
        """
        goal_config = self.generate_random_config(
            self, environment, center=center)
        return self.generate_human_config(self, start_config, start_config, goal_config)

    def generate_random_human_config_with_goal(self, goal_config, 
                                               environment, 
                                               center=np.array([0, 0, 0])):
        """
        Generate a human with a random start config given a known goal
        config. The generated start config will be near center by a threshold
        """
        start_config = self.generate_random_config(
            self, environment, center=center)
        return self.generate_human_config(self, start_config, start_config, goal_config)

    def generate_random_human_config(self, environment, 
                                     center=np.array([0, 0, 0]), radius=5.):
        """
        Generate a random human config (both start and goal configs) from
        the given environment
        """
        start_config = self.generate_random_config(
            self, environment, center=center, radius=radius)
        goal_config = self.generate_random_config(
            self, environment, center=center, radius=radius)
        return self.generate_human_config(self, start_config, start_config, goal_config)

    # For generating positional arguments in an environment
    def generate_random_pos_3(self, center, xdiff=3, ydiff=3):
        """
        Generates a random position near the center within an elliptical radius of xdiff and ydiff
        """
        offset_x = 2*xdiff * random.random() - xdiff  # bound by (-xdiff, xdiff)
        offset_y = 2*ydiff * random.random() - ydiff  # bound by (-ydiff, ydiff)
        offset_theta = 2 * np.pi * random.random()  # bound by (0, 2*pi)
        return np.add(center, np.array([offset_x, offset_y, offset_theta]))

    def within_traversible(self, new_pos, traversible, map_scale, radius=1,
                           stroked_radius=False):
        """
        Returns whether or not the position is in a valid spot in the 
        traversible the Radius input can determine how many surrounding 
        spots must also be valid
        """
        for i in range(2*radius):
            for j in range(2*radius):
                if(stroked_radius):
                    if not((i == 0 or i == radius - 1 or j == 0 or j == radius - 1)):
                        continue
                pos_x = int(new_pos[0] / map_scale) - radius + i
                pos_y = int(new_pos[1] / map_scale) - radius + j
                # Note: the traversible is mapped unintuitively, goes [y, x]
                if (not traversible[pos_y][pos_x]):  # Looking for invalid spots
                    return False
        return True

    def generate_random_pos_in_environment(self, center, environment, radius=5):
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
        map_scale = environment["map_scale"]

        # Combine the occupancy information from the static map
        # and the human
        global_traversible = np.empty(environment["traversibles"][0].shape)
        global_traversible.fill(True)
        if len(environment["traversibles"]) > 1:
            for t in environment["traversibles"]:
                # add 0th and all others that match shape
                if(t.shape == environment["traversibles"][0].shape):
                    global_traversible = np.stack([global_traversible, t], axis=2)
                    global_traversible = np.all(global_traversible, axis=2)
        else:
            global_traversible = environment["traversibles"][0]

        # Generating new position as human's position
        pos_3 = np.array([-1, -1, 0])  # start far out of the traversible
        while(not self.within_traversible(self, pos_3, global_traversible, map_scale, radius=3)):
            pos_3 = self.generate_random_pos_3(self, center, radius, radius)

        # Random theta from 0 to pi
        pos_3[2] = random.random() * 2 * np.pi

        return pos_3
