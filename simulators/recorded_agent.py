from utils.utils import print_colors, generate_name
from simulators.agent import Agent
from humans.human_configs import HumanConfigs
from trajectory.trajectory import SystemConfig, Trajectory
import numpy as np
import socket, time, threading

class PrerecordedAgent(Agent):
    def __init__(self, record_data, name=None):
        if name is None:
            self.name = generate_name(20)
        else:
            self.name = name
        self.record_data = record_data
        self.t = 0
        start = HumanConfigs.generate_config_from_pos_3(record_data[0])
        goal = HumanConfigs.generate_config_from_pos_3(record_data[-1])
        super().__init__(start, goal, name)

        # print(self.record_data)
        # print("prerecorded agent start:", self.start_config.to_3D_numpy(), "goal:", self.goal_config.to_3D_numpy())
    
    def simulation_init(self, sim_params, sim_map, with_planner=True):
        """ Initializes important fields for the CentralSimulator"""
        self.params = sim_params
        self.obstacle_map = sim_map
        # Initialize system dynamics and planner fields
        self.system_dynamics = self._init_system_dynamics()
        self.vehicle_trajectory = Trajectory(dt=self.params.dt, n=1, k=0)

    def get_appearance(self):
        return None

    def execute(self, state):
        self.set_current_config(HumanConfigs.generate_config_from_pos_3(state[:3]))
        print(self.get_current_config().to_3D_numpy())
        # TODO: perhaps make the control loop run multiple commands rather than one
        command = np.array([[[0,0]]], dtype=np.float32)
        # NOTE: the format for the acceleration commands to the open loop for the robot is:
        # np.array([[[L, A]]], dtype=np.float32) where L is linear, A is angular
        # t_seg, actions_nk2 = self.apply_control_open_loop(self.current_config,   
        #                                                 command, 1, sim_mode='ideal'
        #                                                 )
        # self.vehicle_trajectory.append_along_time_axis(t_seg)

    def update_time(self, t):
        self.t = t

    def update(self, sim_state=None):
        most_recent_indx = 0
        while(True):
            r = self.record_data[most_recent_indx]
            trace_time = r[-1] # last element
            if(self.t >= trace_time):
                self.execute(r)
                most_recent_indx += 1
                if(most_recent_indx == len(self.record_data)):
                    break
            else:
                time.sleep(0.01) # TODO: fix hardcoded delay
        
