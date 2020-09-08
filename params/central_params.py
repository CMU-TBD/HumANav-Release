import configparser
from dotmap import DotMap
import numpy as np
import os

# first thing to do is read params file
config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'params/params_example.ini'))
seed = config['base_params'].getint('seed')

# read params file for episodes configs
episodes_config = configparser.ConfigParser()
episodes_config.read(os.path.join(os.getcwd(), 'params/episode_params.ini'))


def get_path_to_socnav():
    # can use a literal string in params.ini as the path
    ### PATH_TO_HUMANAV = config['base_params']['base_directory']
    # or just use the relative path
    PATH_TO_HUMANAV = os.getcwd()
    if(not os.path.exists(PATH_TO_HUMANAV)):
        # the main directory should be the parent of params/central_params.py
        PATH_TO_HUMANAV = os.path.join(os.path.dirname(__file__), '..')
        if(os.path.exists(PATH_TO_HUMANAV)):
            return PATH_TO_HUMANAV
        print('\033[31m', "ERROR: Failed to find tbd_SocNavBench installation at",
              PATH_TO_HUMANAV, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_HUMANAV


def base_data_dir():
    PATH_TO_BASE_DIR = os.path.join(get_path_to_socnav(), 'LB_WayPtNav_Data')
    if(not os.path.exists(PATH_TO_BASE_DIR)):
        print('\033[31m', "ERROR: Failed to find the LB_WayPtNav_Data dir at",
              PATH_TO_BASE_DIR, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_BASE_DIR


def get_sbpd_data_dir():
    PATH_TO_SBPD = os.path.join(
        get_path_to_socnav(), 'sd3dis/stanford_building_parser_dataset')
    if(not os.path.exists(PATH_TO_SBPD)):
        print('\033[31m', "ERROR: Failed to find sd3dis installation at",
              PATH_TO_SBPD, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_SBPD


def get_traversible_dir():
    PATH_TO_TRAVERSIBLES = os.path.join(get_sbpd_data_dir(), 'traversibles')
    if(not os.path.exists(PATH_TO_TRAVERSIBLES)):
        print('\033[31m', "ERROR: Failed to find traversible directory at",
              PATH_TO_TRAVERSIBLES, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_TRAVERSIBLES


def get_surreal_mesh_dir():
    PATH_TO_SURREAL_MESH = os.path.join(
        get_path_to_socnav(), 'surreal/code/human_meshes')
    if(not os.path.exists(PATH_TO_SURREAL_MESH)):
        print('\033[31m', "ERROR: Failed to find SURREAL meshes at",
              PATH_TO_SURREAL_MESH, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_SURREAL_MESH


def get_surreal_texture_dir():
    PATH_TO_SURREAL_TEXTURES = os.path.join(
        get_path_to_socnav(), 'surreal/code/human_textures')
    if(not os.path.exists(PATH_TO_SURREAL_TEXTURES)):
        print('\033[31m', "ERROR: Failed to find SURREAL textures at",
              PATH_TO_SURREAL_TEXTURES, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_SURREAL_TEXTURES


def get_seed():
    return seed


def create_base_params():
    p = DotMap()
    base_p = config['base_params']
    p.dataset_name = base_p.get('dataset_name')
    p.building_name = base_p.get('building_name')
    p.flip = False
    p.load_meshes = base_p.getboolean('load_meshes')
    p.seed = seed
    p.load_traversible_from_pickle_file = base_p.getboolean('load_traversible')
    p.render_3D = base_p.getboolean('render_3D')
    cam_p = config['camera_params']
    p.camera_params = \
        DotMap(modalities=eval(cam_p.get('modalities')),
               width=cam_p.getint('width'),
               height=cam_p.getint('height'),
               z_near=cam_p.getfloat('z_near'),
               z_far=cam_p.getfloat('z_far'),
               fov_horizontal=cam_p.getfloat('fov_horizontal'),
               fov_vertical=cam_p.getfloat('fov_vertical'),
               img_channels=cam_p.getint('img_channels'),
               im_resize=cam_p.getfloat('im_resize'),
               max_depth_meters=cam_p.getfloat('max_depth_meters'))
    p.socnav_dir = get_path_to_socnav()
    p.traversible_dir = get_traversible_dir()
    if(p.render_3D):
        # SBPD Data Directory
        p.sbpd_data_dir = get_sbpd_data_dir()
        # Surreal Parameters
        surr_p = config['surreal_params']
        p.surreal = \
            DotMap(mode=surr_p['mode'],
                   data_dir=get_surreal_mesh_dir(),
                   texture_dir=get_surreal_texture_dir(),
                   body_shapes_train=eval(surr_p.get('body_shapes_train')),
                   body_shapes_test=eval(surr_p.get('body_shapes_test')),
                   compute_human_traversible=surr_p.getboolean(
                       'compute_human_traversible'),
                   render_humans_in_gray_only=surr_p.getboolean(
                       'render_humans_in_gray_only')
                   )
    return p


def create_robot_params():
    p = DotMap()
    # Load the dependencies
    rob_p = config['robot_params']
    p.port = rob_p.getint('port')
    p.repeat_freq: int = rob_p.getint('repeat_freq')
    p.physical_params = \
        DotMap(radius=rob_p.getfloat('radius'),
               base=rob_p.getfloat('base'),
               height=rob_p.getfloat('height'),
               sensor_height=rob_p.getfloat('sensor_height'),
               camera_elevation_degree=rob_p.getfloat(
                   'camera_elevation_degree'),
               delta_theta=rob_p.getfloat('delta_theta'))
    # joystick params
    p.sense_interval = max(1, rob_p.getint('sense_interval'))
    p.track_sim_states = rob_p.getboolean('track_sim_states')
    p.track_vel_accel = rob_p.getboolean('track_vel_accel')
    p.write_pandas_log = rob_p.getboolean('write_pandas_log')
    p.cmd_delay = rob_p.getfloat('cmd_delay')
    p.generate_movie = rob_p.getboolean('generate_movie')
    return p


def create_test_params(test: str):
    p = DotMap()
    test_p = episodes_config[test]
    p.name = test
    p.map_name = test_p.get('map_name')
    p.prerec_start_indx = test_p.getint('prerec_start_indx')
    p.agents_start = eval(test_p.get('agents_start'))
    p.agents_end = eval(test_p.get('agents_end'))
    p.robot_start_goal = eval(test_p.get('robot_start_goal'))
    p.max_time = test_p.getfloat('max_time')
    return p


def create_episodes_params():
    p = {}
    # NOTE: returns a dictionary of DotMaps to use string notation
    # Load the dependencies
    epi_p = episodes_config['episodes_params']
    tests = eval(epi_p.get('tests'))
    for t in tests:
        p[t] = create_test_params(test=t)
    return p


def create_planner_params():
    p = DotMap()

    # Load the dependencies
    p.control_pipeline_params = create_control_pipeline_params()

    from planners.sampling_planner import SamplingPlanner
    # Default of a planner
    p.planner = SamplingPlanner
    return p


def create_waypoint_params():
    p = DotMap()
    from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid
    p.grid = ProjectedImageSpaceGrid

    # Load the dependencies
    wayp_p = config['waypoint_params']

    p.num_waypoints = wayp_p.getint('num_waypoints')
    p.num_theta_bins = wayp_p.getint('num_theta_bins')

    p.bound_min = eval(wayp_p.get('bound_min'))
    p.bound_max = eval(wayp_p.get('bound_max'))

    camera_params = create_base_params().camera_params
    robot_params = create_robot_params().physical_params

    # Ensure square image and aspect ratio = 1
    # as ProjectedImageSpaceGrid assumes this
    assert(camera_params.width == camera_params.height)
    assert(camera_params.fov_horizontal == camera_params.fov_vertical)

    # Additional parameters for the projected grid from the image space to the world coordinates
    p.projected_grid_params = DotMap(
        # Focal length in meters
        # OpenGL default uses the near clipping plane
        f=camera_params.z_near,

        # Half-field of view
        fov=np.deg2rad(camera_params.fov_horizontal / 2.),

        # Height of the camera from the ground in meters
        h=robot_params.sensor_height / 100.,

        # Downwards tilt of the robot camera
        tilt=np.deg2rad(-robot_params.camera_elevation_degree),
    )

    return p


def create_system_dynamics_params():
    p = DotMap()
    from systems.dubins_v2 import DubinsV2
    p.system = DubinsV2

    # Load the dependencies
    dyn_p = config['dynamics_params']

    p.dt = dyn_p.getfloat('dt')

    p.v_bounds = eval(dyn_p.get('v_bounds'))
    p.w_bounds = eval(dyn_p.get('w_bounds'))

    p.linear_acc_max = dyn_p.getfloat('linear_acc_max')
    p.angular_acc_max = dyn_p.getfloat('angular_acc_max')

    p.simulation_params = \
        DotMap(simulation_mode=dyn_p.get('simulation_mode'),
               noise_params=DotMap(is_noisy=dyn_p.getboolean('is_noisy'),
                                   noise_type=dyn_p.get('noise_type'),
                                   noise_lb=eval(dyn_p.get('noise_lb')),
                                   noise_ub=eval(dyn_p.get('noise_ub')),
                                   noise_mean=eval(dyn_p.get('noise_mean')),
                                   noise_std=eval(dyn_p.get('noise_std'))))
    return p


def create_control_pipeline_params():
    p = DotMap()

    p.system_dynamics_params = create_system_dynamics_params()
    p.waypoint_params = create_waypoint_params()

    # Load the dependencies
    cp_p = config['control_pipeline_params']

    from control_pipelines.control_pipeline_v0 import ControlPipelineV0
    p.pipeline = ControlPipelineV0

    # The directory for saving the control pipeline files
    p.dir = os.path.join(base_data_dir(), 'control_pipelines')

    # The time interval between updates, global to system dynamics
    p.dt = create_system_dynamics_params().dt

    # Spline parameters
    from trajectory.spline.spline_3rd_order import Spline3rdOrder
    p.spline_params = DotMap(spline=Spline3rdOrder,
                             max_final_time=cp_p.getfloat('max_final_time'),
                             epsilon=1e-5)
    p.minimum_spline_horizon = cp_p.getfloat('minimum_spline_horizon')

    # LQR setting parameters
    from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
    p.lqr_params = DotMap(cost_fn=QuadraticRegulatorRef,
                          quad_coeffs=np.array(
                              eval(cp_p.get('quad_coeffs')), dtype=np.float32),
                          linear_coeffs=np.zeros((5), dtype=np.float32)
                          )

    # Velocity binning parameters
    p.binning_parameters = DotMap(num_bins=cp_p.getint('num_bins'),
                                  min_speed=p.system_dynamics_params.v_bounds[0],
                                  max_speed=p.system_dynamics_params.v_bounds[1])

    p.convert_K_to_world_coordinates = cp_p.getboolean(
        'convert_K_to_world_coordinates')
    p.discard_LQR_controller_data = cp_p.getboolean(
        'discard_LQR_controller_data')
    p.discard_precomputed_lqr_trajectories = cp_p.getboolean(
        'discard_precomputed_lqr_trajectories')
    p.track_trajectory_acceleration = cp_p.getboolean(
        'track_trajectory_acceleration')
    p.verbose = cp_p.getboolean('verbose')
    return p


def create_simulator_params(render_3D=False):
    p = DotMap()

    sim_p = config['simulator_params']

    # whether or not to wait for joystick inputs or set a repeat frame count
    p.block_joystick = sim_p.getboolean('block_joystick')
    p.delta_t_scale = sim_p.getfloat('delta_t_scale')
    p.socnav_params = create_base_params()
    p.img_scale = sim_p.getfloat('img_scale')
    p.max_frames = sim_p.getint('max_frames')
    # bound by 0 <= X <= 1
    p.fps_scale_down = max(0.0, min(1.0, sim_p.getfloat('fps_scale_down')))
    p.print_data = sim_p.getboolean('print_data')
    p.verbose = sim_p.getboolean('verbose')
    p.join_threads = sim_p.getboolean('join_threads')
    # sbpd simulator params:
    p.render_3D = render_3D
    # Load obstacle map params
    p.obstacle_map_params = create_obstacle_map_params()
    # much faster to only render the topview rather than use the 3D renderer
    if(not p.render_3D):
        print("Rendering topview only")
    else:
        print("Rendering depth and rgb images with 3D renderer")
    p.verbose_printing = sim_p.getboolean('verbose_printing')
    # simulation tick rate
    p.dt = create_system_dynamics_params().dt
    return p


def create_agent_params(with_planner=True, with_obstacle_map=False):
    p = DotMap()
    agent_p = config["agent_params"]

    p.radius = agent_p.getfloat('radius')

    p.record_video = agent_p.getboolean('record_video')
    p.save_trajectory_data = agent_p.getboolean('save_trajectory_data')

    # Load system dynamics params
    p.system_dynamics_params = create_system_dynamics_params()
    if with_planner:
        p.episode_horizon_s = agent_p.getfloat('episode_horizon_s')
        p.control_horizon_s = agent_p.getfloat('control_horizon_s')

        # Load the dependencies
        p.planner_params = create_planner_params()

        # Time discretization step
        dt = p.planner_params.control_pipeline_params.system_dynamics_params.dt

        # Whether or not to track acceleration
        p.track_accel = p.planner_params.control_pipeline_params.track_trajectory_acceleration

        # Updating horizons
        p.episode_horizon = int(np.ceil(p.episode_horizon_s / dt))
        p.control_horizon = int(np.ceil(p.control_horizon_s / dt))
        p.dt = dt

    if with_obstacle_map:
        p.obstacle_map_params = create_obstacle_map_params()

    # Define the Objectives

    # Obstacle Avoidance Objective
    p.avoid_obstacle_objective = \
        DotMap(obstacle_margin0=agent_p.getfloat('obstacle_margin0'),
               obstacle_margin1=agent_p.getfloat('obstacle_margin1'),
               power=agent_p.getfloat('power_obstacle'),
               obstacle_cost=agent_p.getfloat('obstacle_cost'))
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=agent_p.getfloat('power_angle'),
                                    angle_cost=agent_p.getfloat('angle_cost'))
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=agent_p.getfloat('power_goal'),
                                       goal_cost=agent_p.getfloat('goal_cost'),
                                       goal_margin=agent_p.getfloat('goal_margin'))

    # Personal Space cost parameters
    p.personal_space_objective = DotMap(psc_scale=1.0)

    p.objective_fn_params = DotMap(obj_type=agent_p.get('obj_type'))
    p.goal_margin = p.goal_distance_objective.goal_margin
    p.goal_dist_norm = p.goal_distance_objective.power  # Default is l2 norm
    p.episode_termination_reasons = ['Timeout', 'Collision', 'Success']
    p.episode_termination_colors = ['b', 'r', 'g']
    p.waypt_cmap = 'winter'
    p.num_validation_goals = agent_p.getint('num_validation_goals')
    return p


def create_obstacle_map_params():
    p = DotMap()

    # Load the dependencies
    obst_p = config['obstacle_map_params']
    # p.renderer_params = create_base_params()

    from obstacles.sbpd_map import SBPDMap
    p.obstacle_map = SBPDMap

    # Size of map
    # Same as for SocNav FMM Map of Area3
    p.map_size_2 = np.array(eval(obst_p.get('map_size_2')))

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    p.dx = obst_p.getfloat('dx')

    # Origin is always 0,0 for SBPD
    p.map_origin_2 = eval(obst_p.get('map_origin_2'))

    # Threshold distance from the obstacles to sample the start and the goal positions.
    p.sampling_thres = obst_p.getint('sampling_thres')

    # Number of grid steps around the start position to use for plotting
    p.plotting_grid_steps = obst_p.getint('plotting_grid_steps')
    return p


def create_map_params():
    p = DotMap()
    # NOTE: this is very much subject to change with diff maps

    # goal_pos_n2 = np.array([[9., 15.]])
    p.goal_pos_n2 = np.array([[13.0, 8.0]])

    # pos_nk2 = np.array([[[8., 16.], [8., 12.5], [18., 16.5]]], dtype=np.float32)
    p.pos_nk2 = np.array(
        [[[8., 9.], [8., 12.5], [18., 12.5]]], dtype=np.float32)

    # p.test_goal_ang_obj_ans = [19.634956, 29.616005, 74.31618]
    p.test_goal_ang_obj_ans = [5.0869893, 18.3243617, 60.214370]

    # p.test_goal_dist_ans = [49.088074, 179.12201, 2071.5808]
    p.test_goal_dist_ans = [644.88555, 1126.650778, 1126.65109]

    # p.test_obst_map_ans = [1.252921, 1.5730935, 1.7213388]
    p.test_obst_map_ans = [100.0, 0.5900573134422302, 100.0]

    # p.test_obst = [0., 0., 0.]
    p.test_obs_obj_ans = [0., 0., 10.345789710051179]

    return p
