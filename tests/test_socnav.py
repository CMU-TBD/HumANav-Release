import numpy as np
import os
from random import seed, random, randint
# Humanav
from humans.human import Human
from humans.recorded_human import PrerecordedHuman
from simulators.robot_agent import RobotAgent
from socnav.socnav_renderer import SocNavRenderer
# Planner + Simulator:
from simulators.central_simulator import CentralSimulator
from params.central_params import get_seed, create_base_params
from utils.utils import *

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
    # overwrite tests with custom basic test
    p.episode_params.tests = {}
    p.episode_params.tests['test_socnav'] = \
        DotMap(name='test_socnav',
               map_name='Double_Corridor_OG',
               prerec_start_indx=[15],
               prerec_data_filenames=['world_coordinate_inter.csv'],
               prerec_data_framerates=[25],
               prerec_posn_offsets=[[16.0, -0.5]],
               agents_start=[],
               agents_end=[],
               robot_start_goal=[],
               max_time=23,
               write_episode_log=False  # don't write episode log for test_socnav
               )

    # Tilt the camera 10 degree down from the horizontal axis
    p.robot_params.physical_params.camera_elevation_degree = -10

    if p.render_3D:
        # Can only render rgb and depth if the host has an available display
        p.camera_params.modalities = ['rgb', 'disparity']
    else:
        p.camera_params.modalities = ['occupancy_grid']

    return p


def establish_joystick_handshake(p):
    if(p.episode_params.without_robot):
        # lite-mode episode does not include a robot or joystick
        return
    import socket
    import json
    import time
    # sockets for communication
    RobotAgent.host = socket.gethostname()
    # port for recieving commands from the joystick
    RobotAgent.port_recv = p.robot_params.port
    # port for sending commands to the joystick (successor of port_recv)
    RobotAgent.port_send = RobotAgent.port_recv + 1
    RobotAgent.establish_joystick_receiver_connection()
    time.sleep(0.01)
    RobotAgent.establish_joystick_sender_connection()
    # send the preliminary episodes that the socnav is going to run
    json_dict = {}
    json_dict['episodes'] = list(p.episode_params.tests.keys())
    episodes = json.dumps(json_dict)
    # Create a TCP/IP socket
    send_episodes_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect the socket to the port where the server is listening
    server_address = ((RobotAgent.host, RobotAgent.port_send))
    send_episodes_socket.connect(server_address)
    send_episodes_socket.sendall(bytes(episodes, "utf-8"))
    send_episodes_socket.close()


def generate_auto_humans(num_generated_humans, simulator, environment, p, r):
    """
    Generate and add num_humans number of randomly generated humans to the simulator
    """
    traversible = environment["traversibles"][0]
    print("Generating {} autonomous human agents".format(num_generated_humans))
    for _ in range(num_generated_humans):
        # Generates a random human from the environment
        new_human_i = Human.generate_random_human_from_environment(
            environment,
            generate_appearance=p.render_3D
        )
        # Or specify a human's initial configs with a HumanConfig instance
        # Human.generate_human_with_configs(Human, fixed_start_goal)
        # Load a random human at a specified state and speed
        # update human traversible
        if p.render_3D:
            r.add_human(new_human_i)
            environment["traversibles"] = np.array(
                [traversible, r.get_human_traversible()])
        else:
            environment["traversibles"] = np.array([traversible])
        # Input human fields into simulator
        simulator.add_agent(new_human_i)


def test_socnav(num_generated_humans, num_prerecorded, starting_prerec=0):
    """
    Code for loading a random human into the environment
    and rendering topview, rgb, and depth images.
    """
    p = create_params()  # used to instantiate the camera and its parameters

    establish_joystick_handshake(p)

    for i, test in enumerate(list(p.episode_params.tests.keys())):
        episode = p.episode_params.tests[test]
        r = None  # free 'old' renderer
        if(i == 0 or (episode.map_name != p.building_name)):
            # update map to match the episode
            p.building_name = episode.map_name
            print("%s\n\nStarting episode \"%s\" in building \"%s\"%s\n\n" %
                  (color_yellow, test, p.building_name, color_reset))
            # get the renderer from the camera p
            r = SocNavRenderer.get_renderer(p)
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
            room_center = \
                np.array([traversible.shape[1] * 0.5,
                          traversible.shape[0] * 0.5,
                          0.0]
                         ) * dx_m
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
        simulator = CentralSimulator(
            environment,
            renderer=r,
            render_3D=p.render_3D,
            episode_params=episode
        )
        if(not p.episode_params.without_robot):
            """
            Generate the robots for the simulator
            """
            robot_agent = RobotAgent.generate_random_robot_from_environment(
                environment
            )
            simulator.add_agent(robot_agent)

        """
        Add the prerecorded humans to the simulator
        """
        for i in range(len(episode.prerec_data_filenames)):
            PrerecordedHuman.generate_prerecorded_humans(simulator, p,
                                                         max_agents=num_prerecorded,
                                                         start_idx=starting_prerec,
                                                         offset=episode.prerec_posn_offsets[i],
                                                         csv_file=episode.prerec_data_filenames[i],
                                                         fps=episode.prerec_data_framerates[i]
                                                         )
        # generate_prerecorded_humans(starting_prerec, num_prerecorded, p,
        #                             simulator, center_offset=np.array([14.0, 2.0]))
        """
        Generate the autonomous human agents from the episode
        """
        generate_auto_humans(num_generated_humans,
                             simulator, environment, p, r)
        # run simulation
        simulator.simulate()

        # Remove all the humans from the renderer
        if p.render_3D:  # only when rendering with opengl
            r.remove_all_humans()

    if(not p.episode_params.without_robot):
        RobotAgent.close_robot_sockets()


if __name__ == '__main__':
    # run basic room test with variable # of human
    test_socnav(num_generated_humans=5,
                num_prerecorded=15,  # use -1 to include ALL prerecorded agents
                starting_prerec=15)
