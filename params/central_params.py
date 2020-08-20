from dotmap import DotMap
import numpy as np
import os

# seed for randomness generation
seed = 10


def create_base_params():
    p = DotMap()
    p.dataset_name = 'sbpd'
    p.building_name = 'area3'
    p.flip = False
    p.load_meshes = True
    p.seed = seed
    # False allows users to compute a new traversible when
    # using a new area dataset, True will look for the
    # precomputed traversible from the traversible folder
    p.load_traversible_from_pickle_file = False

    # Depending on computer, those equipped with an X graphical instance (or other display)
    # can set this to True to use the openGL renderer and render the 3D humans/scene
    p.render_3D = False
    # If unsure, a display exists if `echo $DISPLAY` yields some output (usually `:0`)

    p.camera_params = DotMap(modalities=['rgb'],  # rgb or disparity
                             width=64,
                             height=64,
                             z_near=.01,  # near plane clipping distance
                             z_far=20.0,  # far plane clipping distance
                             fov_horizontal=90.,
                             fov_vertical=90.,
                             img_channels=3,
                             im_resize=1.,
                             max_depth_meters=np.inf)

    # The robot is modeled as a solid cylinder
    # of height, 'height', with radius, 'radius',
    # base at height 'base' above the ground
    # The robot has a camera at height
    # 'sensor_height' pointing at
    # camera_elevation_degree degrees vertically
    # from the horizontal plane.
    p.robot_params = DotMap(radius=18,
                            base=5,
                            height=100,
                            sensor_height=80,
                            camera_elevation_degree=-45,  # camera tilt
                            delta_theta=1.0)

    # HumANav dir
    p.humanav_dir = get_path_to_humanav()

    # Traversible dir
    p.traversible_dir = get_traversible_dir()

    if(p.render_3D):
        # SBPD Data Directory
        p.sbpd_data_dir = get_sbpd_data_dir()

        # Surreal Parameters
        p.surreal = DotMap(mode='train',
                           data_dir=get_surreal_mesh_dir(),
                           texture_dir=get_surreal_texture_dir(),
                           body_shapes_train=[519, 1320,
                                              521, 523, 779, 365, 1198, 368],
                           body_shapes_test=[
                               337, 944, 1333, 502, 344, 538, 413],
                           compute_human_traversible=True,
                           render_humans_in_gray_only=False
                           )

    return p

# NOTE: these must be the ABSOLUTE path


def get_path_to_humanav():
    PATH_TO_HUMANAV = '/home/gustavo/Documents/tbd_SocNavBenchmark'
    if(not os.path.exists(PATH_TO_HUMANAV)):
        print('\033[31m', "ERROR: Failed to find HumANav installation at",
              PATH_TO_HUMANAV, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_HUMANAV


def base_data_dir():
    PATH_TO_BASE_DIR = os.path.join(get_path_to_humanav(), 'LB_WayPtNav_Data')
    if(not os.path.exists(PATH_TO_BASE_DIR)):
        print('\033[31m', "ERROR: Failed to find the LB_WayPtNav_Data dir at",
              PATH_TO_BASE_DIR, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_BASE_DIR


def get_sbpd_data_dir():
    PATH_TO_SBPD = os.path.join(
        get_path_to_humanav(), 'sd3dis/stanford_building_parser_dataset')
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
        get_path_to_humanav(), 'surreal/code/human_meshes')
    if(not os.path.exists(PATH_TO_SURREAL_MESH)):
        print('\033[31m', "ERROR: Failed to find SURREAL meshes at",
              PATH_TO_SURREAL_MESH, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_SURREAL_MESH


def get_surreal_texture_dir():
    PATH_TO_SURREAL_TEXTURES = os.path.join(
        get_path_to_humanav(), 'surreal/code/human_textures')
    if(not os.path.exists(PATH_TO_SURREAL_TEXTURES)):
        print('\033[31m', "ERROR: Failed to find SURREAL textures at",
              PATH_TO_SURREAL_TEXTURES, '\033[0m')
        os._exit(1)  # Failure condition
    return PATH_TO_SURREAL_TEXTURES


def get_seed():
    return seed


def create_robot_params():
    p = DotMap()

    # can be any valid port, this is an arbitrary choice
    p.port = 6000

    # in our case, the robot's length/width = 66.8 cm, radius is half of that
    # radius of robot, we are basing the drive of the robot off of a pr2 robot
    # more info here: https://robots.ieee.org/robots/pr2/
    p.radius: float = 0.668 / 2.0  # meters

    # number of times to repeat a command (if repeat is on)
    p.repeat_freq: int = 9  # number of frames to repeat last command
    return p


def create_planner_params():
    p = DotMap()

    # Load the dependencies
    p.control_pipeline_params = create_control_pipeline_params()

    from planners.sampling_planner import SamplingPlanner
    p.planner = SamplingPlanner
    return p


def create_waypoint_params():
    p = DotMap()
    from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid
    p.grid = ProjectedImageSpaceGrid

    # Parameters for the projected image space grid
    # Desired number of waypoints. Actual number may differ slightly
    # See ./waypoint_grids/uniform_sampling_grid.py for more info
    p.num_waypoints = 20000
    p.num_theta_bins = 21

    p.bound_min = [0., -2.5, -np.pi]
    p.bound_max = [2.5, 2.5, 0.]

    renderer_params = create_base_params()
    camera_params = renderer_params.camera_params
    robot_params = renderer_params.robot_params

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
    p.dt = 0.05
    p.v_bounds = [0.0, 0.6]
    p.w_bounds = [-1.1, 1.1]

    # Set the acceleration bounds such that
    # by default they are never hit
    p.linear_acc_max = 10e7
    p.angular_acc_max = 10e7

    p.simulation_params = DotMap(simulation_mode='ideal',
                                 noise_params=DotMap(is_noisy=False,
                                                     noise_type='uniform',
                                                     noise_lb=[-0.02, -
                                                               0.02, 0.],
                                                     noise_ub=[
                                                         0.02, 0.02, 0.],
                                                     noise_mean=[0., 0., 0.],
                                                     noise_std=[0.02, 0.02, 0.]))
    return p


def create_control_pipeline_params():
    p = DotMap()

    # Load the dependencies
    p.system_dynamics_params = create_system_dynamics_params()
    p.waypoint_params = create_waypoint_params()

    from control_pipelines.control_pipeline_v0 import ControlPipelineV0
    p.pipeline = ControlPipelineV0

    # The directory for saving the control pipeline files
    p.dir = os.path.join(base_data_dir(), 'control_pipelines')

    # Spline parameters
    from trajectory.spline.spline_3rd_order import Spline3rdOrder
    p.spline_params = DotMap(spline=Spline3rdOrder,
                             max_final_time=10.0,  # 60 crashes pc with 32Gb ram, 6 is default
                             epsilon=1e-5)
    p.minimum_spline_horizon = 1.5  # default 1.5

    # LQR setting parameters
    from costs.quad_cost_with_wrapping import QuadraticRegulatorRef
    p.lqr_params = DotMap(cost_fn=QuadraticRegulatorRef,
                          quad_coeffs=np.array(
                              [1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
                          linear_coeffs=np.zeros((5), dtype=np.float32))

    # Velocity binning parameters
    p.binning_parameters = DotMap(num_bins=20,  # 61 crashes pc with 32gb ram, 15 is slow
                                  min_speed=p.system_dynamics_params.v_bounds[0],
                                  max_speed=p.system_dynamics_params.v_bounds[1])

    # Converting K to world coordinates is slow
    # so only set this to true when LQR data is needed
    p.convert_K_to_world_coordinates = False

    # When not needed, LQR controllers can be discarded
    # to save memory
    p.discard_LQR_controller_data = True

    # Set this to True to ignore precomputed
    # LQR trajectories
    p.discard_precomputed_lqr_trajectories = False

    # Set this to true if you want trajectory objects to track
    # linear and angular acceleration. If not set to false to save memory
    p.track_trajectory_acceleration = True

    p.verbose = False
    return p


def create_simulator_params():
    p = DotMap()

    # Load the dependencies
    p.planner_params = create_planner_params()

    # Load HumANav dependencies
    p.humanav_params = create_base_params()

    # seed for the simulator (different than for numpy and tf) # default 10
    p.seed = 10

    # Horizons in seconds
    p.episode_horizon_s = 200  # more time to simulate a feasable path # default 200
    p.control_horizon_s = 0.5  # time used on every iteration of the controller

    # Whether to log videos taken during trajectories
    p.record_video = False

    # Whether or not to log all trajectory data to pickle
    # files when running this simulator
    p.save_trajectory_data = False

    # Define the Objectives

    # Obstacle Avoidance Objective
    p.avoid_obstacle_objective = DotMap(obstacle_margin0=0.3,
                                        obstacle_margin1=0.5,
                                        power=3,  # exponential cost constant
                                        obstacle_cost=1.0)  # scalar cost multiple
    # Angle Distance parameters
    p.goal_angle_objective = DotMap(power=1,
                                    angle_cost=.008)
    # Goal Distance parameters
    p.goal_distance_objective = DotMap(power=2,
                                       goal_cost=.08,
                                       goal_margin=0.3)  # cutoff distance for the goal

    p.objective_fn_params = DotMap(obj_type='valid_mean')
    p.reset_params = DotMap(
        obstacle_map=DotMap(reset_type='random',
                            params=DotMap(min_n=4, max_n=7,
                                          min_r=.3, max_r=.8)),
        start_config=DotMap(
            position=DotMap(
                # There could be different reset types
                # 'random': the position is initialized randomly on the
                # map but at least at a distance of the obstacle margin from the
                # obstacle.
                reset_type='random'
            ),
            heading=DotMap(
                # 'zero': the heading is initialized to zero.
                # 'random': the heading is initialized randomly within the given
                # bounds.
                reset_type='zero',
                bounds=[-np.pi,
                        np.pi - 1e-10]
            ),
            speed=DotMap(
                # For description of reset types see heading parameters above.
                reset_type='zero',
                bounds=[0., 0.6]
            ),
            ang_speed=DotMap(
                # For description of reset types see heading parameters above.
                reset_type='zero',
                bounds=[-0.5, 0.5],
                # [mean, variance]
                gaussian_params=[0.0, .5]
            )
        ),

        goal_config=DotMap(
            position=DotMap(
                # For description of reset types see position parameters in the
                # start_config above.
                reset_type='random'
            )
        )
    )

    p.goal_cutoff_dist = p.goal_distance_objective.goal_margin
    p.goal_dist_norm = p.goal_distance_objective.power  # Default is l2 norm
    p.episode_termination_reasons = ['Timeout', 'Collision', 'Success']
    p.episode_termination_colors = ['b', 'r', 'g']
    p.waypt_cmap = 'winter'

    p.num_validation_goals = 50
    return p


def create_obstacle_map_params():
    p = DotMap()

    # Load the dependencies
    p.renderer_params = create_base_params()

    from obstacles.sbpd_map import SBPDMap
    p.obstacle_map = SBPDMap

    # Size of map
    p.map_size_2 = np.array([521, 600])  # Same as for HumANav FMM Map of Area3

    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    p.dx = 0.05

    # Origin is always 0,0 for SBPD
    p.map_origin_2 = [0, 0]

    # Threshold distance from the obstacles to sample the start and the goal positions.
    p.sampling_thres = 2

    # Number of grid steps around the start position to use for plotting
    p.plotting_grid_steps = 100
    return p


def create_sbpd_simulator_params(render_3D=None):
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
