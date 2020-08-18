import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_eager_execution()

from obstacles.sbpd_map import SBPDMap
from objectives.obstacle_avoidance import ObstacleAvoidance
from planners.planner import Planner
from simulators.sbpd_simulator import SBPDSimulator
from trajectory.trajectory import Trajectory
from trajectory.trajectory import SystemConfig
from params.planner_params import create_params as create_planner_params
from params.simulator.sbpd_simulator_params import create_params as create_sim_params
from planners.sampling_planner import SamplingPlanner
from dotmap import DotMap

def test_planner():
    # Create parameters


    # Create planner parameters
    planner_params = create_planner_params()
    sim_params = create_sim_params()
    # Define the objective
    # objective = ObstacleAvoidance(params=p.avoid_obstacle_objective,
    #                               obstacle_map=obstacle_map)
    sim = SBPDSimulator(sim_params)
    splanner = SamplingPlanner(sim, planner_params)

    # Spline trajectory params
    n = 1
    dt = 0.1
 
    # Goal states and initial speeds
    goal_pos_n11 = tf.constant([[[8., 12.5]]]) # Goal position (must be 1x1x2 array)
    goal_heading_n11 = tf.constant([[[np.pi/2.]]])
    # Start states and initial speeds
    start_pos_n11 = tf.constant([[[22., 16.5]]]) # Goal position (must be 1x1x2 array)
    start_heading_n11 = tf.constant([[[-np.pi]]])
    start_speed_nk1 = tf.ones((1, 1, 1), dtype=tf.float32) * 0.0
    # Define start and goal configurations
    # start_config = SystemConfig(dt, n, 1, speed_nk1=start_speed_nk1, variable=False)
    start_config = SystemConfig(dt, n,
                               k=1,
                               position_nk2=start_pos_n11,
                               heading_nk1=start_heading_n11,
                               speed_nk1=start_speed_nk1,
                               variable=False)
    goal_config = SystemConfig(dt, n,
                               k=1,
                               position_nk2=goal_pos_n11,
                               heading_nk1=goal_heading_n11,
                               variable=True)
    #  waypts, horizons, trajectories_lqr, trajectories_spline, controllers = controller.plan(start_config, goal_config)
    splanner.simulator.reset_with_start_and_goal(start_config, goal_config)
    splanner.optimize(start_config)
    splanner.simulator.simulate()
    # Visualization
    fig = plt.figure()

    # obstacle_map.render(ax)
    # ax.plot(pos_nk2[0, :, 0].numpy(), pos_nk2[0, :, 1].numpy(), 'r.')
    # ax.plot(trajectory._position_nk2[0, :, 0],trajectory._position_nk2[0, :, 1], 'r-')
    ax = fig.add_subplot(1,3,1)
    splanner.simulator.render(ax)
    ax = fig.add_subplot(1,3,2)
    splanner.simulator.render(ax, zoom=4)
    #ax.plot(objective[0, 0], objective[0, 1], 'k*')
    # ax.set_title('obstacle map')
    ax = fig.add_subplot(1,3,3)
    splanner.simulator.vehicle_trajectory.render(ax, plot_quiver=True)
    splanner.simulator._render_waypoints(ax,plot_quiver=True, text_offset=(-1.5, 0.1))


    # ax = fig.add_subplot(1,3,3)
    # ax.plot(opt_traj._position_nk2[0, :, 0],opt_traj._position_nk2[0, :, 1], 'r-')
    # ax.set_title('opt_traj')
    # Plotting the "distance map"
    # ax = fig.add_subplot(1,2,2)
    # ax.imshow(distance_map, origin='lower')
    # ax.set_title('distance map')

    fig.savefig('./tests/planner/test_planner.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    test_planner()
