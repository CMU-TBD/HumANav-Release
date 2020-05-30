from humanav.render import swiftshader_renderer as sr
from humanav import sbpd, map_utils as mu
from humanav import depth_utils as du
from humanav import utils
from random import seed, random, randint
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
        # Hashable Tuple for dictionaries
        self.identity = (self.name, self.gender, self.shape)
        self.mesh_rng = mesh_rng    
        self.pos_3 = pos_3
        self.goal_3 = goal_3
        self.speed = speed

    def create_random_human_identity(self, identity_rng, dataset):
        """
        Sample a new human identity, but don't load it into
        memory, note that identity's consist of a human "name"
        "gender" "texture" and "body shape" 
        """
        # The human's name will be H_abc...xyz which is randomly generated
        # with 20 characters of the lowercase and uppercase alphabet
        human_name = "".join([random.choice(string.ascii_letters + string.digits) for n in range(20)])

        # Using the SBPD dataset to generate a random gender, texture, and body shape
        human_gender, human_texture, body_shape = \
            dataset.get_random_human_gender_texture_and_body_shape(identity_rng)
        return human_name, human_gender, human_texture, body_shape

    def generate_random_human_at_pos_with_goal(self, pos_3, goal_3, speed, identity_rng, mesh_rng, dataset):
        """
        Sample a new random human from a given position, goal,
        speed, and identity/mesh seeds (dataset from surreal)
        """
        # Generate a random set of identifying features for the human
        name, gender, texture, shape = self.create_random_human_identity(identity_rng, dataset)
        return Human(pos_3, goal_3, speed, name, gender, texture, shape, mesh_rng)

    def generate_random_human(self, environment, dataset, center = np.array([0,0,0]), radius_to_center = 3):
        """
        Sample a new random human from nothing
        - Optionally, can include a center to generate the human nearby
        """
        # Set the identity seed. This is used to sample a random human identity
        # (name, gender, texture, body shape)
        identity_rng = np.random.RandomState(randint(10, 100))
        name, gender, texture, shape = self.create_random_human_identity(self, identity_rng, dataset)

        # Set the Mesh seed. This is used to sample the actual mesh to be loaded
        # which reflects the pose of the human skeleton.
        mesh_rng = np.random.RandomState(randint(10, 100))

        # State of the center and the human. 
        # Specified as [x (meters), y (meters), theta (radians)] coordinates
        pos_3, goal_3 = self.generate_random_pos_and_goal(self, center, environment)

        # Generating random speed of the human in m/s
        speed = random.random() #   random value from 0 to 1

        print("Human at", pos_3, "& goal", goal_3[:2], "& speed", round(speed, 3), "m/s")
        # Generate a random set of identifying features for the human
        return Human(pos_3, goal_3, speed, name, gender, texture, shape, mesh_rng)

    def generate_random_pos_3(self, center, xdiff = 3, ydiff = 3):
        # Generates a random position near the center within an elliptical radius of xdiff and ydiff
        offset_x = 2*xdiff * random.random() - xdiff #bound by (-xdiff, xdiff)
        offset_y = 2*ydiff * random.random() - ydiff #bound by (-ydiff, ydiff)
        offset_theta = 2 * np.pi * random.random()    #bound by (0, 2*pi)
        return np.add(center, np.array([offset_x, offset_y, offset_theta]))

    def within_traversible(self, new_pos, traversible, map_scale, radius = 1, stroked_radius = False):
        # Returns whether or not the position is in a valid spot in the traversible
        # the Radius input can determine how many surrounding spots must also be valid
        for i in range(2*radius):
            for j in range(2*radius):
                if(stroked_radius):
                    if not((i == 0 or i == radius - 1 or j == 0 or j == radius - 1)):
                        continue;
                pos_x = int(new_pos[0] / map_scale) - radius + i
                pos_y = int(new_pos[1] / map_scale) - radius + j
                # Note: the traversible is mapped unintuitively, goes [y, x]
                if (not traversible[pos_y][pos_x]): # Looking for invalid spots
                    return False
        return True

    def generate_random_pos_and_goal(self, center, environment):
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
        for t in environment["traversibles"]:
            if(t.shape == environment["traversibles"][0].shape): #add 0th and all others that match shape
                global_traversible = np.stack([global_traversible, t], axis=2)
                global_traversible = np.all(global_traversible, axis=2)
        
        # Generating new position as human's position
        human_pos_3 = np.array([-1, -1, 0])# start far out of the traversible
        while(not self.within_traversible(self, human_pos_3, global_traversible, map_scale, radius=3)):
            human_pos_3 = self.generate_random_pos_3(self, center, 3, 3);

        # Generating new position as human's goal (endpoint)
        goal_pos_3 = np.array([-1, -1, 0])# start far out of the traversible
        while(not self.within_traversible(self, goal_pos_3, global_traversible, map_scale, radius=3)):
            goal_pos_3 = self.generate_random_pos_3(self, human_pos_3, 1.5, 1.5);

        # Update human i's angle to point towards the goal
        diff_x = goal_pos_3[0] - human_pos_3[0]
        diff_y = goal_pos_3[1] - human_pos_3[1]

        # Update theta
        human_pos_3[2] = math.atan2(diff_y, diff_x)
        
        return human_pos_3, goal_pos_3