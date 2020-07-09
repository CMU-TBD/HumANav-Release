from humans.human_appearance import HumanAppearance
from humans.human_configs import HumanConfigs
from utils.utils import print_colors, generate_name
import math, sys, os, pickle
import numpy as np


class Human():
    all_humans = {}
    def __init__(self, name, appearance, configs, trajectory=None):
        self.name = name
        self.appearance = appearance
        # Identity is a hashable tuple of a human's name, gender, and shape
        if appearance is None:
            self.identity = (name)
        else:
            self.identity = (name, appearance.gender, appearance.shape)
        self.configs = configs
        self.trajectory = trajectory
        self.termination = None

    # Getters for the Human class

    def get_name(self):
        return self.name

    def get_identity(self):
        return self.identity

    def get_appearance(self):
        return self.appearance

    def get_start_config(self):
        return self.configs.get_start_config()

    def get_current_config(self):
        return self.configs.get_current_config()

    def get_goal_config(self):
        return self.configs.get_goal_config()

    def update_trajectory(self, trajectory):
        self.trajectory = trajectory

    def get_trajectory(self):
        return self.trajectory

    def update_termination(self, cause):
        self.termination = cause

    def get_termination(self):
        return self.termination

    def generate_human(self, appearance, configs, name=None, max_chars=20, verbose=False):
        """
        Sample a new random human from all required features
        """
        human_name = None
        if(name is None):
            human_name = generate_name(max_chars)
        else:
            human_name = name
        # In order to print more readable arrays
        np.set_printoptions(precision=2)
        pos_2 = (configs.get_start_config().position_nk2().numpy())[0][0]
        goal_2 = (configs.get_goal_config().position_nk2().numpy())[0][0]
        if(verbose):
            print(" Human", human_name, "at", pos_2, "with goal", goal_2)
        new_human = Human(human_name, appearance, configs)
        # update knowledge of all other humans in the scene
        Human.all_humans[new_human.get_name()] = new_human
        return new_human

    def generate_human_with_appearance(self,
                                       appearance,
                                       environment,
                                       center=np.array([0., 0., 0.])):
        """
        Sample a new human with a known appearance at a random 
        config with a random goal config.
        """
        configs = HumanConfigs.generate_random_human_config( HumanConfigs, environment, center)
        return self.generate_human(self, appearance, configs)

    def generate_human_with_configs(self, configs, name=None, verbose=False):
        """
        Sample a new random from known configs and a randomized
        appearance, if any of the configs are None they will be generated
        """
        appearance = HumanAppearance.generate_random_human_appearance(HumanAppearance)
        return self.generate_human(self, appearance, configs, verbose=verbose, name=name)

    def update_human_with_name(self, name, configs):
        """
        Update an existing human and return them
        """
        updated_human = Human.all_humans[name]
        updated_human.configs = configs
        return updated_human

    def generate_random_human_from_environment(self,
                                               environment,
                                               center=np.array([0., 0., 0.]),
                                               radius=5.,
                                               generate_appearance=False):
        """
        Sample a new human without knowing any configs or appearance fields
        NOTE: needs environment to produce valid configs
        """
        appearance = None
        if generate_appearance:
            appearance = HumanAppearance.generate_random_human_appearance(HumanAppearance)
        configs = HumanConfigs.generate_random_human_config(HumanConfigs,
                                                            environment,
                                                            center,
                                                            radius=radius)
        return self.generate_human(self, appearance, configs)
