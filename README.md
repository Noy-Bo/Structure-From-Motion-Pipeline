## Project Description

This project presents a tool that offers a virtual simulation environment, implemented in Python, for evaluating the accuracy of Structure from Motion (SfM) estimation methods. The tool synthesizes a moving camera viewing an arc object, and its customizable nature allows for testing and evaluation purposes.

The tool supports estimation methods for solving the camera position. Using simulated data, the tool solves the relative rotation and translation between frames, presenting the results through clear and informative graphs.

One of the key features of the tool is its ability to visualize the synthesized scene projection, which allows for the detection of singularities, and validation that the camera is correctly viewing the object. The results of the evaluations are compared to ground truth data from the simulation, and computation time is reported, providing valuable insight into the computational efficiency of each estimation method.

To ensure the validity and reliability of the results, the tool performs sanity tests, including compliance with Epipolar constraints and small reprojection error. The tool is user-friendly and intuitive, with easy configuration to allow for the addition of new estimation algorithms.

## Usage
1. Clone the repository
2. Install the required dependencies
3. Run the main script with desired settings (configure your setting above the main function)

## Vizualizations
![Alt Text](https://github.com/Noy-Bo/Structure-From-Motion-Pipeline/blob/main/images/simulator.gif)

### Projections
![Alt Text](https://github.com/Noy-Bo/Structure-From-Motion-Pipeline/blob/main/images/proj1.png)
![Alt Text](https://github.com/Noy-Bo/Structure-From-Motion-Pipeline/blob/main/images/proj2.png)


### Accuracy measurement - estimation vs ground truth (6-DoF)
![Alt Text](https://github.com/Noy-Bo/Structure-From-Motion-Pipeline/blob/main/images/est_vs_GT.png)
![Alt Text](https://github.com/Noy-Bo/Structure-From-Motion-Pipeline/blob/main/images/est_vs_GT2.png)

### Performance measurement - computation time
![Alt Text](https://github.com/Noy-Bo/Structure-From-Motion-Pipeline/blob/main/images/performance.png)
