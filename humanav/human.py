from humanav.render import swiftshader_renderer as sr
from humanav import sbpd, map_utils as mu
from humanav import depth_utils as du
from humanav import utils
import random, string, math
import numpy as np
import sys
import os
import pickle


class Human():
    def __init__(self, pos_3, goal_3, speed, name, gender, texture, shape, mesh_rng):
        self.name = name
        self.gender = gender
        self.shape = shape
        self.texture = texture
        self.mesh_rng = mesh_rng    
        self.pos_3 = pos_3
        self.goal_3 = goal_3
        self.speed = speed

    def load_random_human_identity(self, identity_rng):
        """
        Sample a new human identity, but don't load it into
        memory, note that identity's consist of a human "name"
        "gender" "texture" and "body shape" 
        """
        # The human's name will be H_xxxxxxxxxxxx which is randomly generated
        # with 20 characters of the lowercase and uppercase alphabet
        human_name = "H_".join(random.choice(string.ascii_letters) for i in range(20))

        # Using the SBPD dataset to generate a random gender, texture, and body shape
        human_gender, human_texture, body_shape = \
            sbpd.get_random_human_gender_texture_and_body_shape(identity_rng, load_materials=False)
        
        identity = {'human_name': human_name,
                    'human_gender': human_gender,
                    'human_texture': human_texture,
                    'body_shape': body_shape}
        return identity

    def generate_random_human(self, pos_3, goal_3, speed, identity_rng, mesh_rng):
        """
        Sample a new random human from a given position, goal,
        speed, and identity/mesh seeds
        """
        # Generate a random set of identifying features for the human
        name, gender, texture, shape = load_random_human_identity(identity_rng)
        return Human(pos_3, goal_3, speed, name, gender, texture, shape, mesh_rng)

    def generate_random_pos_3(self, center, xdiff = 3, ydiff = 3):
        # Generates a random position near the center within an elliptical radius of xdiff and ydiff
        offset_x = 2*xdiff * random() - xdiff #bound by (-xdiff, xdiff)
        offset_y = 2*ydiff * random() - ydiff #bound by (-ydiff, ydiff)
        offset_theta = 2 * np.pi * random()    #bound by (0, 2*pi)
        return np.add(center, np.array([offset_x, offset_y, offset_theta]))

    def within_traversible(self, new_pos, traversible, dx_m, radius = 1, stroked_radius = False):
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

    def generate_random_pos_and_goal(self, center, obstacle_traversible, human_traversible, map_scale = 1):
        """
        Generate a random position (x : meters, y : meters, theta : radians) 
        and near the 'center' with a nearby valid goal position. 
        - Note that the obstacle_traversible and human_traversible are both 
        checked to generate a valid pos_3. 
        - Note that the map_scale primarily refers to the traversible's level
        of precision, it is best to use the dx_m provided in examples.py
        """
        # State of the center and the human. 
        # Specified as [x (meters), y (meters), theta (radians)] coordinates
        human_pos_3 = np.array([-1, -1, 0])# start far out of the traversible
        while(not within_traversible(human_pos_3, obstacle_traversible, map_scale, radius=3) or 
              not within_traversible(human_pos_3, human_traversible, map_scale, radius=3)):
            human_pos_3 = generate_random_pos_3(center, 3, 3);

        # Generating new position as human's goal (endpoint)
        goal_pos_3 = np.array([-1, -1, 0])# start far out of the traversible
        while(not within_traversible(goal_pos_3, obstacle_traversible, map_scale, radius=3) or 
              not within_traversible(goal_pos_3, human_traversible, map_scale, radius=3)):
            new_pos_3 = generate_random_pos_3(human_pos_3, 1.5, 1.5);

        # Update human i's angle to point towards the goal
        diff_x = goal_pos_3[0] - human_pos_3[0]
        diff_y = goal_pos_3[1] - human_pos_3[1]
        # Update theta
        human_pos_3[2] = math.atan2(diff_y, diff_x)
        return human_pos_3, goal_pos_3

