import numpy as np

from obstacles.sbpd_map import SBPDMap
from objectives.goal_distance import GoalDistance
from trajectory.trajectory import Trajectory
from utils.fmm_map import FmmMap
from params.map import Map
from utils.utils import *


def create_renderer_params():
    from params.central_params import get_traversible_dir, get_sbpd_data_dir
    p = Map()
    p.dataset_name = 'sbpd'
    p.building_name = 'area3'
    p.flip = False

    p.camera_params = Map(modalities=['occupancy_grid'],  # occupancy_grid, rgb, or depth
                          width=64,
                          height=64)

    # The robot is modeled as a solid cylinder
    # of height, 'height', with radius, 'radius',
    # base at height 'base' above the ground
    # The robot has a camera at height
    # 'sensor_height' pointing at
    # camera_elevation_degree degrees vertically
    # from the horizontal plane.
    p.robot_params = Map(radius=18,
                         base=10,
                         height=100,
                         sensor_height=80,
                         camera_elevation_degree=-45,  # camera tilt
                         delta_theta=1.0)

    # Traversible dir
    p.traversible_dir = get_traversible_dir()

    # SBPD Data Directory
    p.sbpd_data_dir = get_sbpd_data_dir()

    return p


def create_params():
    p = Map()
    # Goal Distance parameters
    p.goal_distance_objective = Map(power=2,
                                    goal_cost=25.0,
                                    goal_margin=0.0)
    p.obstacle_map_params = Map(obstacle_map=SBPDMap,
                                map_origin_2=[0., 0.],
                                sampling_thres=2,
                                plotting_grid_steps=100)
    p.obstacle_map_params.renderer_params = create_renderer_params()

    return p


def test_goal_distance():
    # Create parameters
    p = create_params()

    # Create an SBPD Map
    from humanav.humanav_renderer_multi import HumANavRendererMulti
    r = HumANavRendererMulti.get_renderer(
        p.obstacle_map_params.renderer_params, deepcpy=False)
    # obtain "resolution and traversible of building"
    dx_cm, traversible = r.get_config()

    obstacle_map = SBPDMap(p.obstacle_map_params,
                           renderer=0, res=dx_cm, trav=traversible)
    # obstacle_map = SBPDMap(p.obstacle_map_params)
    obstacle_occupancy_grid = obstacle_map.create_occupancy_grid_for_map()
    map_size_2 = obstacle_occupancy_grid.shape[::-1]

    # Define a goal position and compute the corresponding fmm map
    goal_pos_n2 = np.array([[9., 15.]])
    fmm_map = FmmMap.create_fmm_map_based_on_goal_position(goal_positions_n2=goal_pos_n2,
                                                           map_size_2=map_size_2,
                                                           dx=0.05,
                                                           map_origin_2=[
                                                               0., 0.],
                                                           mask_grid_mn=obstacle_occupancy_grid)

    # Define the objective
    objective = GoalDistance(params=p.goal_distance_objective, fmm_map=fmm_map)

    # Define a set of positions and evaluate objective
    pos_nk2 = np.array(
        [[[8., 16.], [8., 12.5], [18., 16.5]]], dtype=np.float32)
    trajectory = Trajectory(dt=0.1, n=1, k=3, position_nk2=pos_nk2)

    # Compute the objective
    objective_values_13 = objective.evaluate_objective(trajectory)
    assert objective_values_13.shape == (1, 3)

    # Expected objective values
    distance_map = fmm_map.fmm_distance_map.voxel_function_mn
    idxs_xy_n2 = pos_nk2[0] / .05
    idxs_yx_n2 = idxs_xy_n2[:, ::-1].astype(np.int32)
    expected_distance = np.array([distance_map[idxs_yx_n2[0][0], idxs_yx_n2[0][1]],
                                  distance_map[idxs_yx_n2[1]
                                               [0], idxs_yx_n2[1][1]],
                                  distance_map[idxs_yx_n2[2][0], idxs_yx_n2[2][1]]],
                                 dtype=np.float32)
    cost_at_margin = 25. * p.goal_distance_objective.goal_margin**2
    expected_objective = 25. * expected_distance * expected_distance - cost_at_margin

    # Error in objectives
    # We have to allow a little bit of leeway in this test because the computation of FMM distance is not exact.
    objetive_error = abs(
        expected_objective - objective_values_13[0]) / (expected_objective + 1e-6)
    assert max(objetive_error) <= 0.1

    numerical_error = max(
        abs(objective_values_13[0] - [49.088074, 179.12201, 2071.5808]))
    assert numerical_error <= .01


def main_test():
    test_goal_distance()
    print("%sGoal-distance tests passed!%s" % (color_green, color_reset))


if __name__ == '__main__':
    main_test()
