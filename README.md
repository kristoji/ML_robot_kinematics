# ML_robot_kinematics

Homework 1 for the Machine Learning course at Sapienza University of Rome, Master in Artificial Intelligence and Robotics. The goal of this homework is to implement a simple neural network to learn the forward kinematics of a robot arm. 

I also implemented two algorithms to solve the inverse kinematics problem, using the learned jacobian matrix: the Newton-Raphson method and the Levenberg-Marquardt algorithm.

At the end of the project, I also implemented a PID controller to control the robot arm, from an initial position to the target position, output of the inverse kinematics algorithm.

[Link to the assignment](https://github.com/iocchi/MLHW1_robot_kinematics)

[![3DOF_NN_InvKin_PID](Imgs/3DOF_NN_InvKin_PID.png)](Imgs/3DOF_NN_InvKin_PID.webm)

## Execution
To run the project, execute the following command:
```bash
cd src/
TF_CPP_MIN_LOG_LEVEL=2 python main.py
```

## Globals
◦ DIM and NJOINT: Define the workspace dimension and the number of joints. Values can be (2,2) for a 2DOF planar robot, (2,3) for a 3DOF planar robot, and (3,5) for a 5DOF robot.

◦ IN_SINCOS and OUT_ORIENTATION: Flags to use sine and cosine for input angles and orientation quaternions as output.

◦ NN: A tuple specifying the number of nodes per layer for the neural network.

◦ VALIDATION: When true, uses a separate validation dataset.
    
◦ NEWTON: When true, enables the Newton method; otherwise, the Levenberg-Marquardt algorithm is used as algorithm for the inverse kinematic.

◦ NUM_IT: Specifies the maximum number of iterations for the algorithm.

◦ STRESS_TEST: Repeats the inverse kinematics algorithm 100 times for statistical analysis of convergence and errors.

# TODO
Here are a few improvements worth considering to enhance the results:

• Implement kinematics and PID control for the 5DOF robot.

• Incorporate orientation into the inverse kinematics and subsequent PID control.

• Develop an inverse kinematics algorithm that accounts for joint boundaries.