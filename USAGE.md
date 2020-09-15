# Usage
## Structure of the Simulator
The primary `tbd_SocNavBench` program runs through the various episodes provided (see `episode_params.ini`) and spawns a `CentralSimulator` for each test, the initial states of those simulators are based off the running test. However, in order to start an episode there must also be an external `Joystick` process that is used to send commands, requests, and signals to the robot through a socket communication protocol. The `Joystick` is what users will primarily be interacting with, as it provides the interface for any planning algorithm,

The joystick can:
- `sense()` by requesting a json-serialized `sim_state` which holds all the information about that state.
- `plan()` by using the updated information about the current state, such as the current simulator time, agent positions, and environment.
- `act()` by sending specific velocity commands to the robot to execute in the simulator, which the `CentralSimulator` blocks until the commands are sent.  

![Render Graphic](https://smlbansal.github.io/LB-WayPtNav-DH/resources/images/dataset.jpg)


## Using tbd_SocNavBench
To start the main process for `tbd_SocNavBenchmark` enter the main directory and run the first command below (1) to see `Waiting for joystick connection...` Now run the second command (2) as a separate executable (ie. in another shell instance) to start the `Joystick` process.
```
# The command to start the simulator (1)
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/test_episodes.py

# The command to start the joystick executable (2)
python3 joystick/test_example_joystick.py

# now the two executables will complete the connection handshake and run synchronously
```

## More about the Robot
The tbd_SocNavBenchmark RobotAgent
- Also note that we are making the assumption that both the system dynamics of the robot and the environment are the same.

## More about the Joystick
The main program relies on (and will block on) an inter-process communication channel such as the socket connection between the it and the external "Joystick process". To start a joystick executable you can simply run the `test_example_joystick.py` which will work independently of the type of `Joystick` class that is being used. 

As a head start, we've provided two sample `Joystick` classes in `joystick/example_joystick.py`:
- `JoystickRandom` uses a generic random planner that showcases one of the lightest uses of the Joystick interface.
- `JoystikWithPlanner` uses a basic sampling planner that showcases how a typical planner implementation might be integrated with the interface. 

### Further things to note
- The joystick can be made to run synchronously with the simulator or asynchronously by repeating the last command sent for a number of simulator frames. This can be toggled in [`params/params_example.ini`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/blob/master/params/params_example.ini) by editing the `block_joystick` param in `[simulator_params]`
- The communication port is defaulted to 6000, this can be changed by editing `port` in [`params/params_example.ini`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/blob/master/params/params_example.ini) under `[robot_params]`
  - Note that the program actually uses two sockets to ensure bidirectional communications for asynchronous data transmission. We have designited the successor of `port` to be set as the robot receiver port. Therefore in our default case, we are actually using ports 6000 and 6001.
- The joystick must be run in an external process (but within the same `conda env`)
    - Therefore, make sure before running `test_joystick.py` that the conda environment is `tbd_socnavbench` (same as for `test_socnav.py` and `test_episodes.py`)

### Visualization
Currently the mode is set to "topview only" which uses just matplotlib to render a top-down "bird's-eye-view" perspective without needing the intensive OpenGL renderer. However, to visualize the Depth/RGB modes change the `render_3D` parameter in [`params/params_example.ini`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/blob/master/params/params_example.ini) to `True`. Note that currently the program does not support parallel image rendering when using the 3D renderer, making it very time consuming.