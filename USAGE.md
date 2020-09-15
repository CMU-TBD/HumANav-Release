## Using tbd_SocNavBench
Within the `tbd_SocNavBenchmark` directory run the first command below (1) and it should print `Waiting for joystick connection...` upon success. At this point run the second command (2) as a separate executable (in another shell instance) to send joystick commands
```
# The command to start the simulator (1)
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/test_episodes.py

# The command to start the joystick executable (2)
python3 tests/test_joystick.py

# now the two executables will complete the connection handshake and run synchronously
```

### Visualization
Currently the mode is set to "topview only" which uses just matplotlib to render a top-down "bird's-eye-view" perspective without needing the intensive OpenGL renderer. However, to visualize the Depth/RGB modes change the `render_3D` parameter in [`params/params_exampmle.ini`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/blob/master/params/params_example.ini) to `True`.

## The external Joystick process
The simulation relies on (and will wait until a connection is established) a bidirectional connection to a local socket that communicates between the internal "robot" thread and an external "Joystick" process that can be run as a *separate executable* in `./tests/test_joystick.py`

### Things to note about this codebase
- The joystick must be run in an external process (but within the same `conda env`)
    - Therefore, make sure before running `test_joystick.py` that the conda environment is `tbd_socnavbench` (same as for `test_socnav.py` and `test_episodes.py`)
- The joystick can be made to run synchronously with the simulator or asynchronously by repeating the last command sent for a number of simulator frames. This can be toggled in [`params/params_example.ini`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/blob/master/params/params_example.ini) by editing the `block_joystick` param in `[simulator_params]`
- The communication port is defaulted to 6000, this can be changed by editing `port` in [`params/params_example.ini`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/blob/master/params/params_example.ini) under `[robot_params]`
- Also note that we are making the assumption that both the system dynamics of the robot and the environment are the same.
