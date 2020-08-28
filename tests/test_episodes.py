import matplotlib as mpl
mpl.use('Agg')  # for rendering without a display
import matplotlib.pyplot as plt
import numpy as np
import os
from random import seed, random, randint
import pandas as pd
# Humanav
from humans.human import Human
from humans.recorded_human import PrerecordedHuman
from humans.human_configs import HumanConfigs
from simulators.robot_agent import RoboAgent
from humanav.humanav_renderer_multi import HumANavRendererMulti
# Planner + Simulator:
from simulators.central_simulator import CentralSimulator
from params.central_params import get_seed, create_base_params
from utils.utils import *
from utils.image_utils import *

# seed the random number generator
random.seed(get_seed())


def create_params():
    p = create_base_params()

    # The camera is assumed to be mounted on a robot at fixed height
    # and fixed pitch. See params/central_params.py for more information
    p.camera_params.width = 1024
    p.camera_params.height = 1024
    p.camera_params.fov_vertical = 75.0
    p.camera_params.fov_horizontal = 75.0

    # Introduce the robot params
    from params.central_params import create_robot_params
    p.robot_params = create_robot_params()

    # Introduce the episode params
    from params.central_params import create_episodes_params
    p.episode_params = create_episodes_params()

    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.physical_params.camera_elevation_degree = -10

    if p.render_3D:
        # Can only render rgb and depth if the host has an available display
        p.camera_params.modalities = ['rgb', 'disparity']
    else:
        p.camera_params.modalities = ['occupancy_grid']

    return p


def generate_robot(robot_start_goal, simulator):
    assert(len(robot_start_goal) == 2)
    rob_start = generate_config_from_pos_3(robot_start_goal[0])
    rob_goal = generate_config_from_pos_3(robot_start_goal[1])
    robot_configs = HumanConfigs(rob_start, rob_goal)
    robot_agent = RoboAgent.generate_robot(
        robot_configs
    )
    simulator.add_agent(robot_agent)


def generate_prerecorded_humans(start_ped, num_pedestrians, p, simulator, center_offset=np.array([0., 0.])):
    """"world_df" is a set of trajectories organized as a pandas dataframe.
    Each row is a pedestrian at a given frame (aka time point).
    The data was taken at 25 fps so between frames is 1/25th of a second. """
    if(num_pedestrians > 0 or num_pedestrians == -1):
        datafile = os.path.join(
            p.socnav_dir, "tests/world_coordinate_inter.csv")
        world_df = pd.read_csv(datafile, header=None).T
        world_df.columns = ['frame', 'ped', 'y', 'x']
        world_df[['frame', 'ped']] = world_df[['frame', 'ped']].astype('int')
        start_frame = world_df['frame'][0]  # default start (of data)
        max_peds = max(np.unique(world_df.ped))
        if(num_pedestrians == -1):
            num_pedestrians = max_peds - 1
        print("Gathering prerecorded agents from",
              start_ped, "to", start_ped + num_pedestrians)
        for i in range(num_pedestrians):
            ped_id = i + start_ped + 1
            if (ped_id >= max_peds):  # need data to be within the bounds
                print("%sRequested Prerec agent index out of bounds:" %
                      (color_red), ped_id, "%s" % (color_reset))
            if(ped_id not in np.unique(world_df.ped)):
                continue
            ped_i = world_df[world_df.ped == ped_id]
            times = []
            for j, f in enumerate(ped_i['frame']):
                if(i == 0 and j == 0):
                    start_frame = f  # update start frame to be representative of "first" pedestrian
                relative_time = (f - start_frame) * (1 / 25.)
                times.append(relative_time)
            record = []
            # generate a list of lists of positions (only x)
            for x in ped_i['x']:
                record.append([x + center_offset[0]])
            # append y to the list of positions
            for j, y in enumerate(ped_i['y']):
                record[j].append(y + center_offset[1])
            # append vector angles for all the agents
            for j, pos_2 in enumerate(record):
                if(j > 0):
                    last_pos_2 = record[j - 1]
                    theta = np.arctan2(
                        pos_2[1] - last_pos_2[1], pos_2[0] - last_pos_2[0])
                    record[j - 1].append(theta)
                    if(j == len(record) - 1):
                        record[j].append(theta)  # last element gets last angle
            # append linear speed to the list of variables
            for j, pos_2 in enumerate(record):
                if(j > 0):
                    last_pos_2 = record[j - 1]
                    # calculating euclidean dist / delta_t
                    delta_t = (times[j] - times[j - 1])
                    speed = euclidean_dist2(pos_2, last_pos_2) / delta_t
                    record[j].append(speed)  # last element gets last angle
                else:
                    record[0].append(0)  # initial speed is 0
            for j, t in enumerate(times):  # lastly, append t to the list
                record[j].append(t)
            simulator.add_agent(PrerecordedHuman(
                record, generate_appearance=p.render_3D))
            print("Generated Prerecorded Humans:", i + 1, "\r", end="")
        print("\n")


def generate_auto_humans(starts, goals, simulator, environment, p, r):
    """
    Generate and add num_humans number of randomly generated humans to the simulator
    """
    num_gen_humans = min(len(starts), len(goals))
    traversible = environment["traversibles"][0]
    for i in range(num_gen_humans):
        start_config = generate_config_from_pos_3(starts[i])
        goal_config = generate_config_from_pos_3(goals[i])
        start_goal_configs = HumanConfigs(start_config, goal_config)
        # Generates a random human from the environment
        new_human_i = Human.generate_human_with_configs(
            start_goal_configs,
            generate_appearance=p.render_3D
        )

        # update renderer
        if p.render_3D:
            r.add_human(new_human_i)
            environment["traversibles"] = \
                np.array([traversible, r.get_human_traversible()])
        else:
            environment["traversibles"] = np.array([traversible])

        # Input human fields into simulator
        simulator.add_agent(new_human_i)
        print("Generated Auto Humans:", i + 1, "\r", end="")
    if(num_gen_humans > 0):
        print("\n")


def test_episodes():
    """
    Code for loading a random human into the environment
    and rendering topview, rgb, and depth images.
    """
    p = create_params()  # used to instantiate the camera and its parameters
    # get the renderer from the camera p
    r = HumANavRendererMulti.get_renderer(p, deepcpy=False)
    # obtain "resolution and traversible of building"
    dx_cm, traversible = r.get_config()
    # Convert the grid spacing to units of meters. Should be 5cm for the S3DIS data
    dx_m = dx_cm / 100.0
    if p.render_3D:
        # Get the surreal dataset for human generation
        surreal_data = r.d
        # Update the Human's appearance classes to contain the dataset
        from humans.human_appearance import HumanAppearance
        HumanAppearance.dataset = surreal_data
        human_traversible = np.empty(traversible.shape)
        human_traversible.fill(True)  # initially all good

    # In order to print more readable arrays
    np.set_printoptions(precision=3)

    # TODO: make this a param element
    room_center = np.array([30., 9., 0.])
    # Create default environment which is a dictionary
    # containing ["map_scale", "traversibles"]
    # which is a constant and list of traversibles respectively

    environment = {}
    environment["map_scale"] = dx_m
    environment["room_center"] = room_center
    # obstacle traversible / human traversible
    if p.render_3D:
        environment["traversibles"] = np.array(
            [traversible, human_traversible])
    else:
        environment["traversibles"] = np.array([traversible])
    """
    Creating planner, simulator, and control pipelines for the framework
    of a human trajectory and pathfinding. 
    """

    for test in list(p.episode_params.keys()):
        episode = p.episode_params[test]
        print("%sStarting test:" % color_yellow, test, "%s" % color_reset)

        simulator = CentralSimulator(
            environment,
            renderer=r,
            render_3D=p.render_3D,
            episode_name=test
        )

        """
        Generate the robots for the simulator
        """
        generate_robot(episode.robot_start_goal, simulator)

        """
        Add the prerecorded humans to the simulator
        """
        generate_prerecorded_humans(episode.prerecs_start, episode.num_prerecs, p,
                                    simulator, center_offset=np.array([14.0, 2.0]))

        """
        Generate the autonomous human agents from the episode
        """
        generate_auto_humans(episode.agents_start, episode.agents_end,
                             simulator, environment, p, r)

        # run simulation
        simulator.simulate()

        # Remove all the humans from the renderer
        if p.render_3D:  # only when rendering with opengl
            r.remove_all_humans()


if __name__ == '__main__':
    # run basic room test with variable # of human
    test_episodes()
