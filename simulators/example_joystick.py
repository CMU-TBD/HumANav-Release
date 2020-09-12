import os
from time import sleep
from utils.image_utils import *
from joystick.joystick_base import JoystickBase
from params.central_params import create_agent_params


class JoystickExample(JoystickBase):
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
            rand_v_cmd = \
                random.randint(v_bounds[0] * pr, v_bounds[1] * pr) / pr
            v_cmds.append(rand_v_cmd)

            # also add a random angular velocity command
            rand_w_cmd = \
                random.randint(w_bounds[0] * pr, w_bounds[1] * pr) / pr
            w_cmds.append(rand_w_cmd)
        # send the data in lists based off the simulator/joystick refresh rate
        self.robot_input(v_cmds, w_cmds, sense=True)

    def update_loop(self):
        assert(self.sim_delta_t)  # obtained from the second J.listen_once()
        print("simulator's refresh rate = %.4f" % self.sim_delta_t)
        print("joystick's refresh rate  = %.4f" % self.agent_params.dt)
        self.robot_receiver_socket.listen(1)  # init listener thread
        self.joystick_on = True
        while(self.joystick_on):
            # send a command to the robot
            num_actions_per_dt = \
                int(np.floor(self.sim_delta_t / self.agent_params.dt))
            self.random_inputs(num_actions_per_dt)
            # listen to the robot's reply
            if(not self.listen_once()):
                # occurs if the robot is unavailable or it finished
                self.power_off()
                break
        self.finish_episode()
