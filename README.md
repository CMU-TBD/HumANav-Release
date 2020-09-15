# tbd_SocNavBenchmark
Welcome to the Social Navigation Benchmark utility (SocNavBenchmark), a codebase for benchmarking robot planning algorithms against various episodes of containing multi-agent environments. We are a team of researches from Carnegie Mellon University's Robotics Institute at TBD Lab and HARP Lab. 

![Expected Movie without OpenGL](https://raw.githubusercontent.com/GustavoSilvera/GustavoSilvera.github.io/master/Images/proj/sim_without_humans.gif)

![Expected Movie with OpenGL](https://raw.githubusercontent.com/GustavoSilvera/GustavoSilvera.github.io/master/Images/proj/sim_with_humans.gif)

## External README's
### Installation
Guide for installation at [`INSTALLATION.md`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/tree/master/INSTALLATION.md)
### Usage
Guide for usage at [`USAGE.md`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/tree/master/USAGE.md)

## Rendering Utility
For rendering purposes, we use the [Swiftshader](https://github.com/google/swiftshader) rendering engine, a CPU based rendering engine for photorealistic visuals (rgb, disparity, surface normal, etc.) from textured meshes used in. 
![Render Graphic](https://smlbansal.github.io/LB-WayPtNav-DH/resources/images/dataset.jpg)
### Building(map) Rendering
We use mesh scans of office buildings from the [Stanford Large Scale 3d Indoor Spaces Dataset (SD3DIS)](http://buildingparser.stanford.edu/dataset.html) , however the rendering engine is independent of the meshes used. In principle, textured meshes from scans of any 3D model (see [`sd3dis`](https://github.com/CMU-TBD/tbd_SocNavBenchmark/tree/master/sd3dis)). 
### Human Rendering
For human meshes we turn to the [SURREAL Dataset](https://www.di.ens.fr/willow/research/surreal/data/) which renders images of synthetic humans in a variety of poses, genders, body shapes, and lighting conditions. Though the meshes themselves are synthetic, the human poses in the SURREAL dataset come from real human motion capture data and contain a variety of actions including running, jumping, dancing, acrobatics, and walking. We focus on the subset of poses in which the human is in a walking state.

## Foundations
This project is built upon the *Human Active Navigation* ([HumANav](https://github.com/vtolani95/HumANav-Release)) and *Learning-Based Waypoint Navigation* ([Visual-Navigation](https://github.com/smlbansal/Visual-Navigation-Release)) codebases. Special thanks to [Varun Tolani](https://github.com/vtolani95) for helping us with his projects.
