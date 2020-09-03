# tbd_SocNavBenchmark
Welcome to the Social Navigation Benchmark utility (SocNavBenchmark), a codebase for benchmarking robot planning algorithms within an environment with several agents and indoor obstacles. We are a team of researches from Carnegie Mellon University's Robotics Institute at TBD Lab and HARP Lab. 

## HumANav
Welcome to the Human Active Navigation Dataset (HumANav), a codebase for photorealistic simulations of humans in indoor office environments! We are a team of researchers from UC Berkeley and Google Brain.

## Visual-Navigation
Welcome to LB-WayPtNav (Learning-Based Waypoint Navigation), a codebase for simulation of robot navigational scenarios in indoor-office environments! We are a team of researchers from UC Berkeley and UIUC.

This codebase is a part of our work in ["Visual Navigation Among Humans with Optimal Control as a Supervisor"](https://arxiv.org/pdf/2003.09354.pdf). In this work we show that HumANav enables zero-shot transfer of learning based navigation algorithms directly from simulation to reality. We hope that HumANav can be a useful tool for the broader visual navigation, computer vision, and robotics communities.

![HumANav Graphic](https://smlbansal.github.io/LB-WayPtNav-DH/resources/images/dataset.jpg)
For rendering purposes, we use the [Swiftshader](https://github.com/google/swiftshader) rendering engine, a CPU based rendering
engine for photorealistic visuals (rgb, disparity, surface normal, etc.) from textured meshes used in. We use mesh scans of office buildings from the [Stanford Large Scale 3d Indoor Spaces Dataset (SD3DIS)](http://buildingparser.stanford.edu/dataset.html) , however the rendering engine is independent of the meshes used. In principle, textured meshes from scans of any office buildings can be used. For human meshes we turn to the [SURREAL Dataset](https://www.di.ens.fr/willow/research/surreal/data/) which renders images of synthetic humans in a variety of poses, genders, body shapes, and lighting conditions. Though the meshes themselves are synthetic, the human poses in the SURREAL dataset come from real human motion capture data and contain a variety of actions including running, jumping, dancing, acrobatics, and walking. We focus on the subset of poses in which the human is walking.


More information & a live demo of the HumANav Dataset is available on the [project website](https://smlbansal.github.io/LB-WayPtNav-DH/).

## Download and Configure Data

### Download SMPL data & Render human meshes
Follow the instructions in [`surreal/README.md`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/blob/master/surreal/README.md) to correctly install the human meshes.

### Download SD3DIS data
Follow the instructions in [`sd3dis/README.md`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/blob/master/sd3dis/README.md) to correctly install the building/area meshes. 

### Configure HumANav to look for your data.
In `./params/renderer_params.py` change the following line to your installation of `tbd_SocNavBenchmark`
```
def get_path_to_humanav():
    return '/PATH/TO/tbd_SocNavBenchmark'
```

Note: HumANav is independent of the actual indoor office environment and human meshes used. In this work we use human meshes exported from the [SURREAL](https://www.di.ens.fr/willow/research/surreal/data/) dataset and scans of indoor office environments from the [S3DIS](http://buildingparser.stanford.edu/dataset.html) dataset. However, if you would like to use other meshes, please download and configure them yourself and update the parameters in renderer_params.py to point to your data installation.

## Setup
### Install Anaconda, gcc, g++
```
# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh

# Install gcc and g++ if you don't already have them
sudo apt-get install gcc
sudo apt-get install g++
```

#### Install Libassimp-dev
In the terminal run:
```
sudo apt-get install libassimp-dev
```

### Setup A Virtual Environment
```
conda env create -f environment.yml # this might end with "Pip failed" which is fine
conda activate tbd_socnavbench
```

#### Install pip/conda packages
In the terminal (and in the virtual environment from above [`tbd_lab`]) run:
```
chmod a+x get_packages.sh
./get_packages.sh # make sure the tbd_socnavbench conda environment is active!
```
The script will inform you of all packages being installed and their status, to install manually just look inside


#### Patch the OpenGL Installation
In the terminal run the following commands.
```
1. /PATH/TO/HumANav/humanav
2. bash patches/apply_patches_3.sh
# NOTE: after running get_packages.sh you should see:
# HUNK #3 succeeded at 401 (offset 1 line).
# Hunk #4 succeeded at 407 (offset 1 line).
```
If the script fails there are instructions in apply_patches_3.sh describing how to manually apply the patch. 

### Manually patch pyassimp bug
Additionally, this version of `pyassimp` has a bug which can be fixed by following [this commit](https://github.com/assimp/assimp/commit/b6d3cbcb61f4cc4c42678d5f183351f95c97c8d4) and simply changing `isinstance(obj,int)` to `isinstance(obj, (int, str, bytes))` on line 98 of `anaconda3/envs/tbd_humanav/lib/python3.6/site-packages/pyassimp/core.py`. Then try running the patches again, or manually (not recommended).


#### Install tbd_SocNavBenchmark as a pip package
Follow the steps below to install HumANav as a pip package, so it can be easily integrated with any other codebase.
```
cd /PATH/TO/tbd_SocNavBenchmark
pip install -e .
```

## Run the tbd_SocNavBenchmark installation
To get you started we've included `tests`, which contains the main code example for rendering the central simulaton. Currently the mode is set to "topview only" which focuses on the top-down "bird's-eye-view" perspective without using an OpenGL renderer. However, to run the 3D OpenGL renderer (which adds an RGB and Depth view from the robot's perspective) simply change the `p.render_3D` in `./params/renderer_params` to `True`.

## Run the external Joystick process
The simulation relies on (and will wait until a connection is established) a bidirectional connection to a local socket that communicates between the internal "robot" thread and an external "Joystick" process that can be run as a *separate executable* in `./tests/test_joystick.py`

### Things to note
- The joystick must be run in an external process (and within the same `conda` environment)
    - Therefore, make sure before running `test_joystick.py` that the conda environment is `tbd_socnavbench` (same as for `test_socnav.py`
- The joystick socket is defaulted to 6000, if this interferes with a port on your existing machine, edit the `p.port` in `./params/robot_params`

```
cd /PATH/TO/tbd_SocNavBenchmark/
# To run the main simulator that will wait for the joystick process to execute
PYOPENGL_PLATFORM=egl PYTHONPATH='.' python3 tests/test_socnav.py
# To run the joystick process (once the simulator says "Waiting for Joystick Connection"
python3 tests/test_joystick.py
```

![Expected Movie without OpenGL](https://github.com/GustavoSilvera/GustavoSilvera.github.io/blob/master/Images/proj/sim_without_humans.gif)
![Expected Movie with OpenGL](https://github.com/GustavoSilvera/GustavoSilvera.github.io/blob/master/Images/proj/sim_with_humans.gif)

## Citing This Work
If you find the HumANav dataset useful in your research please cite:
```
@article{tolani2020visual,
  title={Visual Navigation Among Humans with Optimal Control as a Supervisor},
  author={Tolani, Varun and Bansal, Somil and Faust, Aleksandra and Tomlin, Claire},
  journal={arXiv preprint arXiv:2003.09354},
  year={2020}
}
```
