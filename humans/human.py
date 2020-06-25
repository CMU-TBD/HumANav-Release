from humanav.render import swiftshader_renderer as sr
from humanav import sbpd, map_utils as mu
from humanav import depth_utils as du
from humanav import utils
from humanav.renderer_params import get_surreal_texture_dir
from humans.human_appearance import HumanAppearance
from humans.human_configs import HumanConfigs
from random import seed, random, randint
import random, string, math
import numpy as np
import sys
import os
import pickle


class Human():
    name = None
    identity = None
    appearance = None
    configs = None
    trajectory = None
    def __init__(self, name, appearance, configs, trajectory=None):
        self.name = name
        self.appearance = appearance
        # Identity is a hashable tuple of a human's name, gender, and shape
        self.identity = (name, appearance.gender, appearance.shape)
        self.configs = configs
        self.trajectory = trajectory

    # Getters for the Human class
    def get_name(self):
        return self.name
    def get_identity(self):
        return self.identity
    def get_appearance(self):
        return self.appearance
    def get_start_config(self):
        return self.configs.get_start_config()
    def get_goal_config(self):
        return self.configs.get_goal_config()
    def update_trajectory(self, trajectory):
        self.trajectory = trajectory
    def get_trajectory(self):
        return self.trajectory

    def _generate_name(self, max_chars):
        return "".join([random.choice(string.ascii_letters + string.digits) for n in range(max_chars)])

    def generate_human(self, appearance, configs, max_chars = 20):
        """
        Sample a new random human from all required features
        """
        # In order to print more readable arrays
        name = self._generate_name(self, max_chars)
        np.set_printoptions(precision = 2)
        pos_2 = (configs.get_start_config().position_nk2().numpy())[0][0]
        goal_2 = (configs.get_goal_config().position_nk2().numpy())[0][0]
        print('\033[35m', "Human", name, "at", pos_2 ,"with goal", goal_2, '\033[0m')
        return Human(name, appearance, configs)

        
    def generate_human_with_appearance(self, appearance, environment, center = np.array([0.,0.,0.])):
        """
        Sample a new human with a known appearance at a random 
        config with a random goal config.
        """
        configs = HumanConfigs.generate_random_human_config(HumanConfigs, environment, center)
        return generate_human(self, appearance, configs)
    
    def generate_human_with_configs(self, configs, dataset):
        """
        Sample a new random from known configs and a randomized
        appearance, if any of the configs are None they will be generated
        """
        appearance = HumanAppearance.generate_random_human_appearance(HumanAppearance, dataset)
        return generate_human(self, appearance, configs)
        
    def generate_random_human_from_environment(self, dataset, environment, center = np.array([0.,0.,0.]), radius = 5.):
        """
        Sample a new human without knowing any configs or appearance fields
        NOTE: needs environment to produce valid configs, and needs a dataset
        to produce an appearance
        """
        appearance = HumanAppearance.generate_random_human_appearance(HumanAppearance, dataset)
        configs = HumanConfigs.generate_random_human_config(HumanConfigs, environment, center, radius=radius)
        return self.generate_human(self, appearance, configs)
        

    