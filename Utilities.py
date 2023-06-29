import math
import matplotlib.pyplot as plt
import numpy as np
import Parameters
import cv2



# euler angles to rotation matrix
def construct_rotation_from_euler(euler_angles):
    yaw_x = euler_angles[0]
    pitch_y = euler_angles[1]
    roll_z = euler_angles[2]

    x_rotation = np.matrix([[1, 0, 0],
                            [0, np.cos(yaw_x), -np.sin(yaw_x)],
                            [0, np.sin(yaw_x), np.cos(yaw_x)]])
    z_rotation = np.matrix([[np.cos(roll_z), -np.sin(roll_z), 0],
                            [np.sin(roll_z), np.cos(roll_z), 0],
                            [0, 0, 1]])
    y_rotation = np.matrix([[np.cos(pitch_y), 0, np.sin(pitch_y)],
                            [0, 1, 0],
                            [-np.sin(pitch_y), 0, np.cos(pitch_y)]])

    #return y_rotation @ x_rotation @ z_rotation
    return z_rotation @ y_rotation @ x_rotation


# Checks if a matrix is a valid rotation matrix.
def is_rotation_matrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotation_matrix_2_euler_angles(R):
    assert (is_rotation_matrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])

# consturcts the translation vector
def construct_translation_vector(positions):
    return np.matrix([positions[0], positions[1], positions[2]])

# constructs T matrix [R | t] (pose matrix)
def construct_pose_matrix(R,t):
    return np.matrix([[R.A[0][0], R.A[0][1], R.A[0][2], t.A[0][0]],
                      [R.A[1][0], R.A[1][1], R.A[1][2], t.A[0][1]],
                      [R.A[2][0], R.A[2][1], R.A[2][2], t.A[0][2]]])

# Given a rotation matrix and 3D point, rotates the point accordingly
def rotate_point(point, euler_angles):
    R = construct_rotation_from_euler(euler_angles)
    return R @ point.transpose()

# Given a pose in the form [R | t] computes the inverse of this pose.
def compute_pose_inverse(T):

    t = T[0:3,3]
    R = T[0:3,0:3]
    R_inv = np.linalg.inv(R)
    t = -t
    T_inv = np.zeros((3,4))
    T_inv[0:3,0:3] = R_inv
    T_inv[0:3,3] = t.transpose()
    #T_inv = np.concatenate((R_inv,t),axis=1)
    return T_inv

# The compute_pose_composition function takes two input arguments, T1 and T2,
# which are both 3x4 matrices representing poses in a 3D coordinate space.
# the function computes the composition of the two poses by performing a matrix multiplication between T2_aug and T1_aug,
# and returns the result variable T_comp.
# Note the returned pose is now in T2 frame
def compute_pose_composition(T1, T2):

   R1 = T1[0:3, 0:3]
   t1 = T1[0:3, 3]
   R2 = T2[0:3, 0:3]
   t2 = T2[0:3, 3]

   #R = R2 @ R1.transpose()
   R = R1.transpose() @ R2
   t = R1.transpose() @ (t2-t1)

   T_rel = np.zeros((3,4))
   T_rel[0:3,0:3] = R
   T_rel[0:3,3] = t.transpose()
   return T_rel

def normalize_vec(v):
    if np.linalg.norm(v) != 0:
        return (v / np.linalg.norm(v))
    return v

def angle_between(v1, v2):
    v1_u = normalize_vec(v1)
    v2_u = normalize_vec(v2)
    return np.arccos(np.clip(np.dot(v1_u.transpose(), v2_u), -1.0, 1.0))


def calc_essential_from_R_t(R,t):
    t_cross = np.zeros((3,3))
    t_cross[0][1] = -t[2]
    t_cross[1][0] = t[2]
    t_cross[0][2] = -t[1]
    t_cross[2][0] = t[1]
    t_cross[1][2] = -t[0]
    t_cross[2][1] = t[0]

    return t_cross @ R

#input (R1,R2,t) estimation and (R,t) ground truth
# chooses the closest R,t to GT's R,t
# 4 possible soltions (R1,t) (R1,-t) (R2,t) (R2,-t)
def choose_closest_solution(est, GT):
    # choose translation
    est[2] = normalize_vec(est[2])
    GT[1] = normalize_vec(GT[1])
    t = est[2]
    if np.mean(np.abs(est[2].transpose() - GT[1])) > np.mean(np.abs(-est[2].transpose() - GT[1])):
        t = -est[2]

    err1 = np.linalg.norm(est[0] - GT[0])
    err2 = np.linalg.norm(est[1] - GT[0])
    if err2 > err1:
        return (est[0],t)
    return (est[1],t)

#plot all statistics
def plot_statistics(estimations, GT):

    plot_poses_data(estimations, GT)
    plot_runtime_data(estimations)


def plot_poses_data(estimations, GT):
    num_timestamps = len(estimations[0].keys()) - 1

    for solver in estimations:

        f = plt.figure()

        for i in range(0,num_timestamps):

            (R_GT, t_GT) = GT[i][:2]
            (R,t) = solver[i][:2]

            t = np.array(t)

            # convert rotation matrix to euler angles
            (x_roll_est,y_pitch_est,z_yaw_est) = rotation_matrix_2_euler_angles(R) * 180/np.pi # to degrees
            (x_roll_GT,y_pitch_GT,z_yaw_GT) = rotation_matrix_2_euler_angles(R_GT) * 180/np.pi

            # plot
            plt.suptitle('Pose estimation results for {} solver vs Ground Truth'.format(solver['name']),fontsize=20)

            # rotation
            ax = plt.subplot(2,3,1)
            plt.title('x angle [deg]')
            plt.scatter(i, x_roll_est, marker='x',color='r')
            plt.scatter(i, x_roll_GT, facecolors='none', edgecolors='b')
            ax.set_xlabel('timestamp')
            ax.set_ylabel('angle [deg]')
            ax.set_ylim(-180, 180)
            plt.legend(["Estimation", "Ground Truth"])

            ax = plt.subplot(2,3,2)
            plt.title('y angle [deg]')
            plt.scatter(i, y_pitch_est, marker='x',color='r')
            plt.scatter(i, y_pitch_GT, facecolors='none', edgecolors='b')
            ax.set_xlabel('timestamp')
            ax.set_ylabel('angle [deg]')
            ax.set_ylim(-180, 180)
            plt.legend(["Estimation", "Ground Truth"])

            ax = plt.subplot(2,3,3)
            plt.title('z angle [deg]')
            plt.scatter(i, z_yaw_est, marker='x',color='r')
            plt.scatter(i, z_yaw_GT, facecolors='none', edgecolors='b')
            ax.set_xlabel('timestamp')
            ax.set_ylabel('angle [deg]')
            ax.set_ylim(-180, 180)
            plt.legend(["Estimation", "Ground Truth"])

            # translation
            ax = plt.subplot(2,3,4)
            plt.title('x translation')
            plt.scatter(i, t[0], marker='x',color='r')
            plt.scatter(i, t_GT[0],facecolors='none', edgecolors='b')
            ax.set_xlabel('timestamp')
            ax.set_ylabel('position')
            ax.set_ylim(-10, 10)
            plt.legend(["Estimation", "Ground Truth"])

            ax = plt.subplot(2,3,5)
            plt.title('y translation')
            plt.scatter(i, t[1],  marker='x',color='r')
            plt.scatter(i, t_GT[1], facecolors='none', edgecolors='b')
            ax.set_xlabel('timestamp')
            ax.set_ylabel('position')
            ax.set_ylim(-10, 10)
            plt.legend(["Estimation", "Ground Truth"])

            ax = plt.subplot(2,3,6)
            plt.title('z translation')
            plt.scatter(i, t[2], marker='x',color='r')
            plt.scatter(i, t_GT[2], facecolors='none', edgecolors='b')
            ax.set_xlabel('timestamp')
            ax.set_ylabel('position')
            ax.set_ylim(-10, 10)
            plt.legend(["Estimation", "Ground Truth"])

        plt.show()
def plot_runtime_data(estimations):
    num_timestamps = len(estimations[0].keys()) - 1

    f = plt.figure()
    names = []
    for solver in estimations:
        names.append(solver['name'])
        times = []
        x = []
        for i in range(0, num_timestamps):
            time = solver[i][3]
            times.append(time)
            x.append(i)

        plt.plot(x, times, 'x--')
    plt.legend(names)
    plt.title("Algorithms computation time", fontsize=20)
    plt.ylabel("time [seconds]")
    plt.xlabel("timestamp")
    plt.show()

def vec_2_skewsym(v):
    m = np.zeros((3,3))
    m[1,2] = -v[0,0]
    m[2,1] = v[0,0]
    m[0,2] = v[1,0]
    m[2,0] = -v[1,0]
    m[0,1] = -v[2,0]
    m[1,0] = v[2,0]
    return m

def calc_reprojection_error(name, pts_i1, pts_i2, R_rel, t_rel, K, scale):
    err_list = []
    K_inv = np.linalg.pinv(K)

    passed_epipolar_test = True
    passed_reprojection_test = True

    for i in range(len(pts_i1)):
        X1 = pts_i1[i]  # 2D point
        X2 = pts_i2[i]  # 2D point # CONTINUE FROM HERE CONSTRUCT X_HAT
        X1_hat = K_inv @ np.matrix([X1[0].item(), X1[1].item(), 1], dtype='float').transpose()  # move to camera frame
        X2_hat = K_inv @ np.matrix([X2[0].item(), X2[1].item(), 1], dtype='float').transpose()  # move to camera frame

        # epipolar constraint sanity check
        t_skewsym = vec_2_skewsym(-t_rel)
        epipolar_error = X2_hat.transpose() @ t_skewsym @ R_rel.transpose() @ X1_hat # [pEp' = 0]
        # test value vs tolerance
        if (epipolar_error > Parameters.TOLERANCE):
            passed_epipolar_test = False

        # rotate point in I2 to I1 and check reprojection error
        X2_est = (R_rel.transpose() @ X1_hat) - t_rel/(scale[i]) # note that the scale must be used here to get the results here. (point_scale = camera_scale + landmark_scale)
        X2_pix_est = K @ X2_est
        X2_pix_est = X2_pix_est / X2_pix_est[-1]
        err = np.square(np.subtract(np.array(X2_pix_est[0:2]), np.matrix(np.array(X2)).transpose()))
        # test value vs tolerance
        if (err.max() > Parameters.TOLERANCE):
            passed_reprojection_test = False


        err_list.append(err)

    # report if test passed.
    if passed_epipolar_test is False:
        print("[Epipolar Test][{}] result is above tolerance value.".format(name))
    if passed_reprojection_test is False:
        print("[Reprojection Test][{}] result is above tolerance value.".format(name))

    mean_err = np.array(err_list).mean()
    return mean_err

def zero_small_values(a):
    close_to_zero_boolean = np.isclose(a,0)
    a[close_to_zero_boolean] = 0
    return a
def plot_cam(ax, rvec, tvec, color):
    # Convert the rotation vector to a rotation matrix
    R, _ = cv2.Rodrigues(rvec)

    # Create a 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.T

    # Create the basis vectors for the camera pose
    x = T @ np.array([1, 0, 0, 1])
    y = T @ np.array([0, 1, 0, 1])
    z = T @ np.array([0, 0, 1, 1])
    o = T @ np.array([0, 0, 0, 1])

    # Normalize the vectors
    x = x / np.linalg.norm(x - o)
    y = y / np.linalg.norm(y - o)
    z = z / np.linalg.norm(z - o)

    # Plot the basis vectors
    ax.quiver(o[0], o[1], o[2], x[0] - o[0], x[1] - o[1], x[2] - o[2], color=color)
    ax.quiver(o[0], o[1], o[2], y[0] - o[0], y[1] - o[1], y[2] - o[2], color=color)
    ax.quiver(o[0], o[1], o[2], z[0] - o[0], z[1] - o[1], z[2] - o[2], color=color)

    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
def cam_visualization(cam_params_solution, cam_params_init, cam_params_GT):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cam_idx in range(cam_params_solution.shape[0]):
        rvec_solution = cam_params_solution[cam_idx, :3]
        tvec_solution = cam_params_solution[cam_idx, 3:6]
        plot_cam(ax,rvec_solution, tvec_solution, 'b')

        rvec_init = cam_params_init[cam_idx, :3]
        tvec_init = cam_params_init[cam_idx, 3:6]
        plot_cam(ax, rvec_init, tvec_init, 'r')

        rvec_GT = cam_params_GT[cam_idx, :3]
        tvec_GT = cam_params_GT[cam_idx, 3:6]
        plot_cam(ax, rvec_GT, tvec_GT, 'g')

    plt.show()

def arc_visualization(points3d, model_lines):
    # vizualization of triangulation output
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for i in range(0, points3d.shape[0]):
        xs = points3d[i][0]
        ys = points3d[i][1]
        zs = points3d[i][2]
        ax.scatter(xs, ys, zs)

    for line in model_lines:
        point1 = (points3d[line[0]][0], points3d[line[0]][1], points3d[line[0]][2])
        point2 = (points3d[line[1]][0], points3d[line[1]][1], points3d[line[1]][2])

        x = np.array([point1[0], point2[0]])
        y = np.array([point1[1], point2[1]])
        z = np.array([point1[2], point2[2]])

        # Plot the line
        ax.plot(x, y, z)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

def cam_and_pts_visualization(points_3d, points_3d_solution, camera_params, camera_params_solution, camera_params_GT, points_3d_GT):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # plot pts init
    for i in range(0, points_3d.shape[0]):
        xs = points_3d[i][0]
        ys = points_3d[i][1]
        zs = points_3d[i][2]
        ax.scatter(xs, ys, zs, color='red')

    # plot pts solution
    for i in range(0, points_3d_solution.shape[0]):
        xs = points_3d_solution[i][0]
        ys = points_3d_solution[i][1]
        zs = points_3d_solution[i][2]
        ax.scatter(xs, ys, zs, color= 'blue')

    # plot pts GT
    for i in range(0, points_3d_GT.shape[0]):
        xs = points_3d_GT[i][0]
        ys = points_3d_GT[i][1]
        zs = points_3d_GT[i][2]
        ax.scatter(xs, ys, zs, color='green')

    #plot cameras
    for cam_idx in range(camera_params_solution.shape[0]):
        #solution
        rvec_solution = camera_params_solution[cam_idx, :3]
        tvec_solution = camera_params_solution[cam_idx, 3:6]
        plot_cam(ax,rvec_solution, tvec_solution, 'b')

        #init
        rvec_init = camera_params[cam_idx, :3]
        tvec_init = camera_params[cam_idx, 3:6]
        plot_cam(ax, rvec_init, tvec_init, 'r')

        #GT:
        rvec_init = camera_params_GT[cam_idx, :3]
        tvec_init = camera_params_GT[cam_idx, 3:6]
        plot_cam(ax, rvec_init, tvec_init, 'g')

    plt.axis('equal')
    plt.show()