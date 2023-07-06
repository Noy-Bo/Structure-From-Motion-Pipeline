from Estimator import Estimator
from Simulator import Simulator
import Utilities
## ------------------------ renderer playground control  ------------------- ##
camera_object_transformations = []
camera_euler_angles = [0, 0, 0]       # <<---------  camera playground here!
camera_position = [0, 0, 0]           # <<---------  camera playground here!
point_euler_angles = [0, -5, 0]        # <<---------  object playground here!
point_position = [0, 0, 0]            # <<---------  object playground here!
camera_object_transformations.append(camera_euler_angles)
camera_object_transformations.append(camera_position)
camera_object_transformations.append(point_euler_angles)
camera_object_transformations.append(point_position)
## ------------------------------------------------------------------------ ##

## input camera poses -  consists of 3DoF translation and 3DoF EA(euler angles in RADIANS) ##
input_camera_poses = []
# NOTE - for a fare inference, the translation vector must be a unit vector!
# pose 1 - should stay with 0 translation and 0 rotation.
t1 = [0, 0, 0]
EA1 = [0, 0, 0]
input_camera_poses.append((t1, EA1))
# pose 2
t2 = [0.5, 1, 0.1]
EA2 = [-0.4, 0.3, -0.1]
input_camera_poses.append((t2, EA2))
# # pose 3
# t3 = [0, 0, -1]
# EA3 = [-0.2, 0.2, 1]
# input_camera_poses.append((t3, EA3))
# # pose 4
# t4 = [1, 0, 0]
# EA4 = [0, -0.32, 0]
# input_camera_poses.append((t4, EA4))
# # pose 5
# t5 = [0, 1, 0]
# EA5 = [0.12, -0.32, 0]
# input_camera_poses.append((t5, EA5))
# # pose 6
# t6 = [0.7777, 0.7777, 0]
# EA6 = [0.4, -0.3, 0]
# input_camera_poses.append((t6, EA6))

## --------------------------------------------------------------------------- ##

if __name__ == "__main__":

    # simulation part - create data
    simulator = Simulator()

    """
    ## ------------------------ Simulator Modes ------------------------ ##
    # "rotation demo"  -  This mode is for testing the renderer
    #                     This demo mode demonstrates a camera
    #                     observing a rotating 3D object. The goal of this
    #                     mode check that the camera & model are
    #                     transformed and projected correctly
    # "playground"     -  This mode is for testing the transformations
    #                     Playground mode enable to synthesize scene
    #                     with full control over Camera & Object transformation.
    #                     The control is via  the global lists above
    #                     (poses/point R,t), see playground section above
    # "simulate poses" -  This mode is for inference and estimating.
    #                     This mode will synthesize the scenes of the input
    #                     poses consecutively. to get the next scene press
    #                     DOWN_ARROW_KEY
    ## ------------------------------------------------------------------ ##
    """
    mode = "simulate poses"
    logger = simulator.run_simulator(camera_poses=input_camera_poses, camera_object_transformations=camera_object_transformations, mode=mode, hold_poses=False)

    if logger is not None:
        # estimation part- retrieve R,t
        estimator = Estimator(logger=logger)

        # predicting t0-,..,tend; t0->t1, t0->t2, ... t0->tend.
        t0 = 0
        tend = len(input_camera_poses)-1
        GT_poses, estimations = estimator.estimate_relative_pose_and_landmarks(logger=logger, t0=t0, t1=tend)

        # plotting statistics.
        Utilities.plot_statistics(estimations=estimations, GT=GT_poses)