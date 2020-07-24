from utils.utils import print_colors, generate_name
from simulators.agent import Agent
from humans.human_configs import HumanConfigs
from trajectory.trajectory import SystemConfig
import numpy as np
import socket, time, threading

class PrerecordedAgent(Agent):
    def __init__(self, record_data, name=None):
        if name is None:
            self.name = generate_name(20)
        else:
            self.name = name
        self.record_data = record_data
        self.start_config = HumanConfigs.generate_config_from_pos_3(record_data[0])
        self.goal_config = HumanConfigs.generate_config_from_pos_3(record_data[-1])
    
    def execute(self, state):
        self.current_config = HumanConfigs.generate_config_from_pos_3(state)
        # TODO: perhaps make the control loop run multiple commands rather than one
        command = np.array([[[0,0]]], dtype=np.float32)
        # NOTE: the format for the acceleration commands to the open loop for the robot is:
        # np.array([[[L, A]]], dtype=np.float32) where L is linear, A is angular
        t_seg, actions_nk2 = self.apply_control_open_loop(self.current_config,   
                                                        command, 1, sim_mode='ideal'
                                                        )
        self.vehicle_trajectory.append_along_time_axis(t_seg)

    def update(self, sim_state=None):
        last_act_t = 0
        for r in self.record_data:
            # NOTE: this is assuming the execute is INSTANT (which it probably isnt)
            self.execute(r)
            delay = r[-1] - last_act_t
            last_act_t = r[-1]
            time.sleep(delay)