from humanav.render import swiftshader_renderer as sr
from humanav import sbpd, map_utils as mu
from humanav import depth_utils as du
from humanav import utils
from humanav.renderer_params import get_surreal_texture_dir
from random import seed, random, randint
import random, string, math
import numpy as np
import sys
import os
import pickle


class Human():
    name = None
    gender = None
    shape = None
    texture = None
    identity = None
    mesh_rng = None
    pos_3 = None
    goal_3 = None
    speed = None

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

    # Getters for the Human class
    def get_name(self):
        return self.name
    def get_gender(self):
        return self.gender
    def get_shape(self):
        return self.shape
    def get_texture(self):
        return self.texture
    def get_identity(self):
        return self.identity
    def get_mesh_rng(self):
        return self.mesh_rng
    def get_pos_3(self):
        return self.pos_3
    def get_goal_3(self):
        return self.goal_3
    def get_speed(self):
        return self.speed


    def generate_name(self, max_chars):
        return "".join([random.choice(string.ascii_letters + string.digits) for n in range(max_chars)])

    def create_random_human_identity(self, dataset, identity_rng):
        """
        Sample a new human identity, but don't load it into
        memory, note that identity's consist of a human "name"
        "gender" "texture" and "body shape" 
        """
        # The human's name will be H_abc...xyz which is randomly generated
        # with 20 characters of the lowercase and uppercase alphabet
        human_name = self.generate_name(self,20)

        # Using the SBPD dataset to generate a random gender, texture, and body shape
        human_gender, human_texture, body_shape = \
            dataset.get_random_human_gender_texture_and_body_shape(identity_rng)
        return human_name, human_gender, human_texture, body_shape

    def get_known_human_identity(self):
        """
        Specify a known human identity. An identity
        is a Tuple with elements (name, gender, texture, shape)
        - Note: this is primarily for testing purposes and will
        generate the same human every time
        """

        # If you know which human you want to load,
        # specify the params manually (or load them from a file)
        human_identity = (self.generate_name(self,20), 'male', [os.path.join(get_surreal_texture_dir(), 'train/male/nongrey_male_0110.jpg')], 1320)

        return human_identity

    def generate_human(self, name, gender, texture, shape, mesh_rng, pos_3, goal_3, speed):
        """
        Sample a new random human from all required features
        """
        # In order to print more readable arrays
        np.set_printoptions(precision = 2)
        print("Human at", pos_3, "& goal", goal_3[:2], "& speed", round(speed, 3), "m/s")
        return Human(pos_3, goal_3, speed, name, gender, texture, shape, mesh_rng)

    def generate_random_human_at_pos_with_goal(self, dataset, pos_3, goal_3, speed, identity_rng, mesh_rng):
        """
        Sample a new random human from a given position, goal,
        speed, and identity/mesh seeds
        **useful if known the rng's and localizatoin arguments**
        """
        # Generate a random set of identifying features for the human
        name, gender, texture, shape = self.create_random_human_identity(self, dataset, identity_rng)
        return self.generate_human(self, name, gender, texture, shape, mesh_rng, pos_3, goal_3, speed)
    
    def generate_human_with_known_identity(self, name, gender, texture, shape, environment, center):
        """
        Sample a new human from known identity features, but unknown 
        positional/speed arguments (and mesh rng)
        **useful if identity/meshes are known, but position is not**
        """
        # Set the Mesh seed. This is used to sample the actual mesh to be loaded
        # which reflects the pose of the human skeleton.
        mesh_rng = np.random.RandomState(randint(1, 1000))

        # State of the center and the human. 
        # Specified as [x (meters), y (meters), theta (radians)] coordinates
        pos_3, goal_3 = self.generate_random_pos_and_goal(self, center, environment)

        # generate a random speed (in m/s) from [0 to 1)
        speed = random.random()
        return self.generate_human(self, name, gender, texture, shape, mesh_rng, pos_3, goal_3, speed)

    def generate_random_human_at_pos_3(self, dataset, pos_3, goal_3 = np.array([0,0,0]), speed = random.random()):
        """
        Sample a new random human from simply a position
        - Note that the human_goal_3 will default to be [0,0,0]
        unless otherwise specified
        **useful if mesh/ID rng's are unknown but positions/speed are known**
        """
        # Set the identity seed. This is used to sample a random human identity
        # (name, gender, texture, body shape)
        identity_rng = np.random.RandomState(randint(1, 1000))

        # Set the Mesh seed. This is used to sample the actual mesh to be loaded
        # which reflects the pose of the human skeleton.
        mesh_rng = np.random.RandomState(randint(1, 1000))

        return self.generate_random_human_at_pos_with_goal(self, dataset, pos_3, goal_3, speed, identity_rng, mesh_rng)

    def generate_random_human_from_environment(self, dataset, environment, center = np.array([0,0,0])):
        """
        Sample a new random human from the environment
        - Optionally, can include a center to generate the human nearby
        and will generate a nearby goal that will be used in the pos_3 function
        **Useful if only prior knowledge of a human is their environment and dataset**
        """
        # State of the center and the human. 
        # Specified as [x (meters), y (meters), theta (radians)] coordinates
        pos_3, goal_3 = self.generate_random_pos_and_goal(self, center, environment)

        # generate a random speed (in m/s) from [0 to 1)
        speed = random.random()

        return self.generate_random_human_at_pos_3(self, dataset, pos_3, goal_3, speed)

    # For generating positional arguments in an environment
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