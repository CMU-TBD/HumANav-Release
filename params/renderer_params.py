from dotmap import DotMap
import numpy as np
import os


def create_params():
    p = DotMap()
    p.dataset_name = 'sbpd'
    p.building_name = 'area3'
    p.flip = False
    p.load_meshes = True
    # False allows users to compute a new traversible when
    # using a new area dataset, True will look for the  
    # precomputed traversible from the traversible folder
    p.load_traversible_from_pickle_file = True

    # Depending on pc, those equipped with an X graphical instance (or other display)
    # can set this to True to use the openGL renderer and render the 3D humans/scene
    p.render_with_display = False 
    # If unsure, a display exists if `echo $DISPLAY` yields some output (usually `:0`)

    p.camera_params = DotMap(modalities=['rgb'],  # rgb or disparity
                             width=64,
                             height=64,
                             z_near=.01, # near plane clipping distance
                             z_far=20.0, # far plane clipping distance
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

    # SBPD Data Directory
    p.sbpd_data_dir = get_sbpd_data_dir()

    # Surreal Parameters
    p.surreal = DotMap(mode='train',
                       data_dir=get_surreal_mesh_dir(),
                       texture_dir=get_surreal_texture_dir(),
                       body_shapes_train=[519, 1320, 521, 523, 779, 365, 1198, 368],
                       body_shapes_test=[337, 944, 1333, 502, 344, 538, 413],
                       compute_human_traversible=True,
                       render_humans_in_gray_only=False
                      )

    return p

# NOTE: this must be the ABSOLUTE path
def get_path_to_humanav():
    PATH_TO_HUMANAV = '/home/gsilvera/Documents/tbd_SocNavBenchmark'
    if(not os.path.exists(PATH_TO_HUMANAV)):
        print('\033[31m', "ERROR: Failed to find HumANav installation at", PATH_TO_HUMANAV, '\033[0m')
        os._exit(1) # Failure condition
    return PATH_TO_HUMANAV

def base_data_dir():
    PATH_TO_BASE_DIR = os.path.join(get_path_to_humanav(), 'LB_WayPtNav_Data')
    if(not os.path.exists(PATH_TO_BASE_DIR)):
        print('\033[31m', "ERROR: Failed to find the LB_WayPtNav_Data dir at", PATH_TO_BASE_DIR, '\033[0m')
        os._exit(1) # Failure condition
    return PATH_TO_BASE_DIR

def get_sbpd_data_dir():
    PATH_TO_SBPD = os.path.join(get_path_to_humanav(),'sd3dis/stanford_building_parser_dataset')
    if(not os.path.exists(PATH_TO_SBPD)):
        print('\033[31m', "ERROR: Failed to find SBPD installation at", PATH_TO_SBPD, '\033[0m')
        os._exit(1) # Failure condition
    return PATH_TO_SBPD

def get_traversible_dir():
    PATH_TO_TRAVERSIBLES = os.path.join(get_sbpd_data_dir(), 'traversibles')
    if(not os.path.exists(PATH_TO_TRAVERSIBLES)):
        print('\033[31m', "ERROR: Failed to find traversible directory at", PATH_TO_TRAVERSIBLES, '\033[0m')
        os._exit(1) # Failure condition
    return PATH_TO_TRAVERSIBLES

def get_surreal_mesh_dir():
    PATH_TO_SURREAL_MESH = os.path.join(get_path_to_humanav(), 'surreal/code/human_meshes')
    if(not os.path.exists(PATH_TO_SURREAL_MESH)):
        print('\033[31m', "ERROR: Failed to find SURREAL textures at", PATH_TO_SURREAL_MESH, '\033[0m')
        os._exit(1) # Failure condition
    return PATH_TO_SURREAL_MESH

def get_surreal_texture_dir():
    PATH_TO_SURREAL_TEXTURES = os.path.join(get_path_to_humanav(), 'surreal/code/human_textures')
    if(not os.path.exists(PATH_TO_SURREAL_TEXTURES)):
        print('\033[31m', "ERROR: Failed to find SURREAL meshes at", PATH_TO_SURREAL_TEXTURES, '\033[0m')
        os._exit(1) # Failure condition
    return PATH_TO_SURREAL_TEXTURES
