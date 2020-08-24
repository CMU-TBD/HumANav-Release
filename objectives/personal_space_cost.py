import numpy as np
from objectives.objective_function import Objective
from simulators.sim_state import *
from metrics.cost_functions import *

class PersonalSpaceCost(Objective):
    """
    Compute the cost of being in non ego gen_agents' path.
    """

    def __init__(self, params):
        self.p = params
        self.tag = 'personal_space_cost_per_nonego_agent'

    def evaluate_objective(self, trajectory, sim_state: SimState):
        # get ego agent position
        ego_pos3 = trajectory.position_and_heading_nk3()  # (x,y,th)_self

        # iterate through every non ego agent

        agents = sim_state.get_all_agents()

        personal_space_cost = 0

        for agent_name, agent_vals in agents:
            agent_pos3 = agent_vals.get_pos3()  # (x,y,th)
            theta = agent_pos3[2]
            # gaussian centered around the non ego agent
            # TODO actually account for velocity here
            personal_space_cost += asym_gauss_from_vel(x=ego_pos3[0], y=ego_pos3[1],
                                       velx=np.cos(theta), vely=np.sin(theta),
                                       xc=agent_pos3[0], yc=agent_pos3[1])

        return personal_space_cost
