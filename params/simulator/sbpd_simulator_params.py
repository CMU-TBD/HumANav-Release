from dotmap import DotMap
# from simulators.central_simulator import CentralSimulator
from params.obstacle_map.sbpd_obstacle_map_params import create_params as create_obstacle_map_params
from params.simulator.simulator_params import create_params as create_simulator_params


def create_params(render_3D=None):
    p = create_simulator_params()

    p.render_3D = render_3D

    # Load the dependencies
    p.obstacle_map_params = create_obstacle_map_params()

    # p.simulator = CentralSimulator

    # Custom goal reset parameters
    # 'random_v1 ': the goal position is initialized randomly on the
    # map but at least at a distance of the obstacle margin from the
    # obstacle and at most max_dist from the start. Additionally
    # the difference between fmm and l2 distance between goal and
    # start must be greater than some threshold (sampled based on
    # max_dist_diff)
    p.reset_params.goal_config = DotMap(position=DotMap(
        reset_type='random_v1',
        max_dist_diff=.5,
        max_fmm_dist=6.0
    ))
    return p
