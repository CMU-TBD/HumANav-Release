import numpy as np
import os
from time import sleep
from random import randint
from joystick.joystick_base import JoystickBase
from params.central_params import create_agent_params


class JoystickRandom(JoystickBase):
    def __init__(self):
        # planner variables
        self.commanded_actions = []  # the list of commands sent to the robot to execute
        super().__init__()

    def _init_obstacle_map(self, renderer=0):
        """ Initializes the sbpd map."""
        p = self.agent_params.obstacle_map_params
        env = self.current_ep.get_environment()
        return p.obstacle_map(p, renderer,
                              res=float(env["map_scale"]) * 100.,
                              trav=np.array(env["traversibles"][0])
                              )

    def init_control_pipeline(self):
        self.agent_params = create_agent_params(with_obstacle_map=True)
        self.obstacle_map = self._init_obstacle_map()
        # TODO: establish explicit limits of freedom for users to use this code
        # self.obj_fn = Agent._init_obj_fn(self, params=self.agent_params)
        # self.obj_fn.add_objective(Agent._init_psc_objective(params=self.agent_params))

        # Initialize Fast-Marching-Method map for agent's pathfinding
        # self.fmm_map = Agent._init_fmm_map(self, params=self.agent_params)
        # Agent._update_fmm_map(self)

        # Initialize system dynamics and planner fields
        # self.planner = Agent._init_planner(self, params=self.agent_params)
        # self.vehicle_data = self.planner.empty_data_dict()
        # self.system_dynamics = Agent._init_system_dynamics(self, params=self.agent_params)
        # self.vehicle_trajectory = Trajectory(dt=self.agent_params.dt, n=1, k=0)

    def random_inputs(self, amnt: int, pr: int = 100):
        # TODO: get these from params
        v_bounds = [0, 1.2]
        w_bounds = [-1.2, 1.2]
        v_cmds = []
        w_cmds = []
        for _ in range(amnt):
            # add a random linear velocity command to send
            rand_v_cmd = randint(v_bounds[0] * pr, v_bounds[1] * pr) / pr
            v_cmds.append(rand_v_cmd)

            # also add a random angular velocity command
            rand_w_cmd = randint(w_bounds[0] * pr, w_bounds[1] * pr) / pr
            w_cmds.append(rand_w_cmd)
        # send the data in lists based off the simulator/joystick refresh rate
        self.send_cmds(v_cmds, w_cmds)

    def joystick_sense(self):
        # ping's the robot to request a sim state
        self.send_to_robot("sense")

        # listen to the robot's reply
        if(not self.listen_once()):
            # occurs if the robot is unavailable or it finished
            self.joystick_on = False

        # NOTE: at this point, self.sim_state_now is updated with the
        # most up-to-date simulation information

    def joystick_plan(self):
        pass

    def joystick_act(self):
        if(self.joystick_on):
            num_actions_per_dt = \
                int(np.floor(self.sim_delta_t / self.agent_params.dt))
            # send a random to the robot
            self.random_inputs(num_actions_per_dt)

    def update_loop(self):
        assert(self.sim_delta_t)
        print("simulator's refresh rate = %.4f" % self.sim_delta_t)
        print("joystick's refresh rate  = %.4f" % self.agent_params.dt)
        self.robot_receiver_socket.listen(1)  # init listener thread
        self.joystick_on = True
        while(self.joystick_on):
            # gather information about the world state based off the simulator
            self.joystick_sense()

            # create a plan for the next steps of the trajectory
            self.joystick_plan()

            # send a command to the robot
            self.joystick_act()

        self.finish_episode()


""" BEGIN PLANNED JOYSTICK """
from trajectory.trajectory import Trajectory, SystemConfig
from utils.utils import generate_config_from_pos_3
from simulators.agent import Agent


class JoystickWithPlanner(JoystickBase):
    def __init__(self):
        # planner variables
        self.commanded_actions = []  # the list of commands sent to the robot to execute
        self.num_cmds_sent = 0
        super().__init__()

    def _init_obstacle_map(self, renderer=0):
        """ Initializes the sbpd map."""
        p = self.agent_params.obstacle_map_params
        env = self.current_ep.get_environment()
        return p.obstacle_map(p, renderer,
                              res=float(env["map_scale"]) * 100.,
                              trav=np.array(env["traversibles"][0])
                              )

    def init_control_pipeline(self):
        self.agent_params = create_agent_params(with_obstacle_map=True)
        self.obstacle_map = self._init_obstacle_map()
        self.obj_fn = Agent._init_obj_fn(self, params=self.agent_params)
        self.obj_fn.add_objective(
            Agent._init_psc_objective(params=self.agent_params))

        # Initialize Fast-Marching-Method map for agent's pathfinding
        self.fmm_map = Agent._init_fmm_map(self, params=self.agent_params)
        Agent._update_fmm_map(self)

        # Initialize system dynamics and planner fields
        self.planner = Agent._init_planner(self, params=self.agent_params)
        self.vehicle_data = self.planner.empty_data_dict()
        self.system_dynamics = \
            Agent._init_system_dynamics(self, params=self.agent_params)
        self.vehicle_trajectory = Trajectory(dt=self.agent_params.dt, n=1, k=0)

    def joystick_sense(self):
        # ping's the robot to request a sim state
        self.send_to_robot("sense")

        # listen to the robot's reply
        if(not self.listen_once()):
            # occurs if the robot is unavailable or it finished
            self.joystick_on = False

        # NOTE: at this point, self.sim_state_now is updated with the
        # most up-to-date simulation information

    def joystick_plan(self):
        """ Runs the planner for one step from config to generate a
        subtrajectory, the resulting robot config after the robot executes
        the subtrajectory, and relevant planner data
        - Access to sim_states from the self.current_world
        """
        robot_config = generate_config_from_pos_3(self.robot_current)
        robot_goal_config = generate_config_from_pos_3(self.get_robot_goal())
        self.planner_data = \
            self.planner.optimize(robot_config,
                                  robot_goal_config,
                                  self.sim_states)

        # LQR feedback control loop
        t_seg = Trajectory.new_traj_clip_along_time_axis(self.planner_data['trajectory'],
                                                         self.agent_params.control_horizon,
                                                         repeat_second_to_last_speed=True)
        _, commanded_actions_nkf = self.system_dynamics.parse_trajectory(
            t_seg)
        # NOTE: the format for the velocity commands to the open loop for the robot is:
        # np.array([[[L, A]]], dtype=np.float32) where L is linear, A is angular
        self.planned_next_config = \
            SystemConfig.init_config_from_trajectory_time_index(
                t_seg,
                t=-1
            )
        self.vehicle_trajectory.append_along_time_axis(
            t_seg,
            track_trajectory_acceleration=self.agent_params.planner_params.track_accel
        )
        self.commanded_actions.extend(commanded_actions_nkf[0])
        self.current_config = \
            SystemConfig.init_config_from_trajectory_time_index(
                self.vehicle_trajectory, t=-1)

    def joystick_act(self):
        if(self.joystick_on):
            num_actions_per_dt = \
                int(np.floor(self.sim_delta_t / self.agent_params.dt))
            # initialize the command containers
            v_cmds, w_cmds = [], []
            for _ in range(num_actions_per_dt):
                command = self.commanded_actions[self.num_cmds_sent]
                lin = command[0]
                ang = command[1]
                v_cmds.append(float(lin))
                w_cmds.append(float(ang))
                if len(v_cmds) >= num_actions_per_dt:
                    self.send_cmds(v_cmds, w_cmds)
                self.num_cmds_sent += 1

    def update_loop(self):
        assert(self.sim_delta_t)
        print("simulator's refresh rate = %.4f" % self.sim_delta_t)
        print("joystick's refresh rate  = %.4f" % self.agent_params.dt)
        self.robot_receiver_socket.listen(1)  # init listener thread
        self.joystick_on = True
        while(self.joystick_on):
            # gather information about the world state based off the simulator
            self.joystick_sense()

            # create a plan for the next steps of the trajectory
            self.joystick_plan()

            # send a command to the robot
            self.joystick_act()

        self.finish_episode()
