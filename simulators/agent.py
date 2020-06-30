from humanav.renderer_params import get_surreal_texture_dir
from humans.human import Human
from random import seed, random, randint
from utils.utils import print_colors
import random, string, math
import numpy as np
import sys
import os
import pickle


class Agent():

    def __init__(self, name, appearance, configs, trajectory=None):
        self.start_config = None
        self.goal_config = None
        self.end_episode = False
        self.vehicle_trajectory = None
        self.vehicle_data = None
        self.vehicle_data_last_step = None
        self.last_step_data_valid = None 
        self.episode_type = None 
        self.valid_episode = None 
        self.commanded_actions_1kf = None 
        self.obj_val = None 

    def human_to_agent(self, appearance, configs, max_chars = 20):
        """
        Sample a new random human from all required features
        """
        # In order to print more readable arrays
        name = self._generate_name(self, max_chars)
        np.set_printoptions(precision = 2)
        pos_2 = (configs.get_start_config().position_nk2().numpy())[0][0]
        goal_2 = (configs.get_goal_config().position_nk2().numpy())[0][0]
        print(print_colors()["yellow"], "Human", name, "at", pos_2 ,"with goal", goal_2, print_colors()["reset"])
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
        

    