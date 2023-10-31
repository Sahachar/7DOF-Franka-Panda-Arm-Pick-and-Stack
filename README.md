# 7DOF Franka Panda Arm Pick and Stack
A reliable system suitable for both stationary and moving block retrieval and stacking using the Franka Panda Arm. We rigorously tested our program in simulation and also conducted thorough tests on the actual robot, with promising outcomes in both settings. We explored various tactics and situations in both the simulated environment and on the actual robot during the testing process and optimized the pipeline to be time-efficient. The code integrates Forward and Inverse kinematics, as well as concepts like obstacle evasion and path planning techniques like RRT, in a finite states' approach as a modular solution to accurately pick and stack both stationary and dynamic blocks onto the desired platform in a time-efficient and secure way. The pipeline depends on the pose information of the blocks from any custom perception module. It is also capable of re-orienting the blocks to stack them with a particular face up (in our case-white face-up).

## Installation and Usage

The simulator must be run on Ubuntu 20.04 with ROS noetic installed. You can follow the standard installation instructions for [Ubuntu 20.04](https://phoenixnap.com/kb/install-ubuntu-20-04) and [ROS noetic](http://wiki.ros.org/noetic/Installation).

To get started using codebase, please follow the instructions to install [panda_simulator](https://github.com/justagist/panda_simulator), a Gazebo-based simulator for the Franka Emika Panda robot.

If all has been done correctly, you should be able to run

```
roslaunch panda_gazebo panda_world.launch
```

to launch the Gazebo simulation.

Now you can use this repository in your catkin_ws/src and build again. Run the labs/final.py python file. All the required components of the solution like forward and inver kinematics, planning, optimization, obstacle avoidance are in the libs folder.
