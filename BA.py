from __future__ import print_function
import matplotlib.pyplot as plt
import time
from scipy.optimize import least_squares
import numpy as np
from scipy.sparse import lil_matrix
from Parameters import FOCAL_LENGTH, WIDTH, HEIGHT


def run_BA(camera_params, points_3d, camera_indices, point_indices, points_2d):
    """
    Performs Bundle Adjustment on camera parameters and 3D points.

    Params:
        camera_params: A ndarray of shape (n_cameras, 9) containing initial estimates for camera parameters.
        points_3d: A ndarray of shape (n_points, 3) containing initial estimates of 3D point coordinates in the world frame.
        camera_indices: A ndarray of shape (n_observations,) containing indices for cameras corresponding to each observation.
        point_indices: A ndarray of shape (n_observations,) containing indices for points corresponding to each observation.
        points_2d: A ndarray of shape (n_observations, 2) containing 2D coordinates for points projected on images during each observation.

    Returns:
        camera_params, points_3d: Adjusted camera parameters and 3D points.

    """
    n_cameras = camera_params.shape[0]
    n_points = points_3d.shape[0]

    shape_camera_params = camera_params.shape
    shape_points_3d = points_3d.shape

    n = 9 * n_cameras + 3 * n_points
    m = 2 * points_2d.shape[0]
    print("Running Bundle Adjustment Optimization...")
    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    # visualizing the residuals before
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
    f0 = compute_residuals(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    plt.close("all")

    A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    t0 = time.time()
    res = least_squares(compute_residuals, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))

    plt.plot(f0, 'r', label="Before Optimization")
    plt.plot(res.fun, 'b', label="After Optimization")
    plt.legend()
    plt.title("Reprojection Error:")
    plt.xlabel("Projection")
    plt.ylabel("Error")
    plt.show()

    # extract parameters from the least square solution
    # split x0 back into two
    len_camera_params = np.prod(shape_camera_params)  # total number of elements in camera_params
    camera_params_solution = res.x[:len_camera_params]
    points_3d_solution = res.x[len_camera_params:]

    # reshape each array back to its original shape
    camera_params_solution = camera_params_solution.reshape(shape_camera_params)
    points_3d_solution = points_3d_solution.reshape(shape_points_3d)

    return camera_params_solution, points_3d_solution

def rotate(points, rot_vecs):
    """
    Rotates points by given rotation vectors using Rodrigues' rotation formula.

    Parameters:
        points: A ndarray of shape (n_points, 3) containing 3D point coordinates.
        rot_vecs: A ndarray of equivalent shape containing rotation vectors.

    Returns:
        ret: The rotated 3D points.

    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

def project_with_distortion(points, camera_params):
    """
    Converts 3D points to 2D by projecting onto images.

    The function rotates the 3D points then translates them using camera parameters.

    Params:
        points: A ndarray of shape (n_points, 3) containing 3D point coordinates in the world frame.
        camera_params: A ndarray of shape (n_cameras, 9) containing camera parameters.

    Returns:
        points_proj: The projected 2D points.
    """
    points_proj = rotate(points, camera_params[:, :3])
    points_proj += camera_params[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params[:, 6]
    k1 = camera_params[:, 7]
    k2 = camera_params[:, 8]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj

def project(points, camera_params):
    """
    Converts 3D points to 2D by projecting onto images.

    The function rotates the 3D points then translates them using camera parameters.

    Params:
        points: A ndarray of shape (n_points, 3) containing 3D point coordinates in the world frame.
        camera_params: A ndarray of shape (n_cameras, 9) containing camera parameters.

    Returns:
        points_proj: The projected 2D points.
    """
    points_proj = rotate(points, -camera_params[:, :3])
    points_proj += -camera_params[:, 3:6]
    points_proj[:,:2] *= FOCAL_LENGTH
    points_proj[:,0] += ((WIDTH/2) * points_proj[:,2])
    points_proj[:,1] += ((HEIGHT/2) * points_proj[:,2])
    points_proj = points_proj[:, :2] / points_proj[:, 2, np.newaxis]

    points_proj = points_proj[:, :]
    return points_proj

def compute_residuals(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """
    Compute the residuals between the projected 2D points and given 2D points.

    The function project() is referred to perform projection from 3D to 2D.

    Parameters:
        params: A 1-D array containing the flattened camera parameters and 3-D point coordinates.
        n_cameras: The number of cameras involved.
        n_points: The number of 3D points involved.
        camera_indices: A ndarray of shape (n_observations,) containing indices for cameras corresponding to each observation.
        point_indices: A ndarray of shape (n_observations,) containing indices for points corresponding to each observation.
        points_2d: A ndarray of shape (n_observations, 2) containing 2D coordinates for points projected on images during each observation.

    Returns:
        Residuals between the projected 2D points and the given 2D points.
    """
    camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
    points_3d = params[n_cameras * 6:].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params[camera_indices])
    return ((points_proj - points_2d)).ravel()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    """
    Computes the sparsity pattern for the Jacobian matrix of a bundle adjustment problem.

    Params:
        n_cameras: The number of cameras involved.
        n_points: The number of 3D points involved.
        camera_indices: A ndarray of shape (n_observations,) containing indices for cameras corresponding to each observation.
        point_indices: A ndarray of shape (n_observations,) containing indices for points corresponding to each observation.

    Returns:
        A: A sparse matrix containing the sparsity pattern of the Jacobian matrix.

    """
    m = camera_indices.size * 2
    n = n_cameras * 6 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(6):
        A[2 * i, camera_indices * 6 + s] = 1
        A[2 * i + 1, camera_indices * 6 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

    return A



# def compute_residuals_9params(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
#     """
#     Compute the residuals between the projected 2D points and given 2D points.
#
#     The function project() is referred to perform projection from 3D to 2D.
#
#     Parameters:
#         params: A 1-D array containing the flattened camera parameters and 3-D point coordinates.
#         n_cameras: The number of cameras involved.
#         n_points: The number of 3D points involved.
#         camera_indices: A ndarray of shape (n_observations,) containing indices for cameras corresponding to each observation.
#         point_indices: A ndarray of shape (n_observations,) containing indices for points corresponding to each observation.
#         points_2d: A ndarray of shape (n_observations, 2) containing 2D coordinates for points projected on images during each observation.
#
#     Returns:
#         Residuals between the projected 2D points and the given 2D points.
#     """
#     camera_params = params[:n_cameras * 9].reshape((n_cameras, 9))
#     points_3d = params[n_cameras * 9:].reshape((n_points, 3))
#     points_proj = project(points_3d[point_indices], camera_params[camera_indices])
#     return ((points_proj - points_2d)).ravel()

# def bundle_adjustment_sparsity_9params(n_cameras, n_points, camera_indices, point_indices):
#     """
#     Computes the sparsity pattern for the Jacobian matrix of a bundle adjustment problem.
#
#     Params:
#         n_cameras: The number of cameras involved.
#         n_points: The number of 3D points involved.
#         camera_indices: A ndarray of shape (n_observations,) containing indices for cameras corresponding to each observation.
#         point_indices: A ndarray of shape (n_observations,) containing indices for points corresponding to each observation.
#
#     Returns:
#         A: A sparse matrix containing the sparsity pattern of the Jacobian matrix.
#
#     """
#     m = camera_indices.size * 2
#     n = n_cameras * 9 + n_points * 3
#     A = lil_matrix((m, n), dtype=int)
#
#     i = np.arange(camera_indices.size)
#     for s in range(9):
#         A[2 * i, camera_indices * 9 + s] = 1
#         A[2 * i + 1, camera_indices * 9 + s] = 1
#
#     for s in range(3):
#         A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
#         A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1
#
#     return A
# def read_bal_data(file_name):
#     with bz2.open(file_name, "rt") as file:
#         n_cameras, n_points, n_observations = map(
#             int, file.readline().split())
#
#         camera_indices = np.empty(n_observations, dtype=int)
#         point_indices = np.empty(n_observations, dtype=int)
#         points_2d = np.empty((n_observations, 2))
#
#         for i in range(n_observations):
#             camera_index, point_index, x, y = file.readline().split()
#             camera_indices[i] = int(camera_index)
#             point_indices[i] = int(point_index)
#             points_2d[i] = [float(x), float(y)]
#
#         camera_params = np.empty(n_cameras * 9)
#         for i in range(n_cameras * 9):
#             camera_params[i] = float(file.readline())
#         camera_params = camera_params.reshape((n_cameras, -1))
#
#         points_3d = np.empty(n_points * 3)
#         for i in range(n_points * 3):
#             points_3d[i] = float(file.readline())
#         points_3d = points_3d.reshape((n_points, -1))
#
#     return camera_params, points_3d, camera_indices, point_indices, points_2d



# def BA_from_local_dataset():
#
#     FILE_NAME = "G:\My Drive\GitRepos\SFM\problem-49-7776-pre.txt.bz2"
#     # if not os.path.isfile(FILE_NAME):
#     #     urllib.request.urlretrieve(URL, FILE_NAME)
#
#     camera_params, points_3d, camera_indices, point_indices, points_2d = read_bal_data(FILE_NAME)
#     n_cameras = camera_params.shape[0]
#     n_points = points_3d.shape[0]
#
#     n = 9 * n_cameras + 3 * n_points
#     m = 2 * points_2d.shape[0]
#
#     print("n_cameras: {}".format(n_cameras))
#     print("n_points: {}".format(n_points))
#     print("Total number of parameters: {}".format(n))
#     print("Total number of residuals: {}".format(m))
#
#     # visualizing the residuals before
#     x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))
#     f0 = compute_residuals(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
#     plt.close("all")
#     plt.plot(f0)
#
#     A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
#     t0 = time.time()
#     res = least_squares(compute_residuals, x0, jac_sparsity=A, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
#                         args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
#     t1 = time.time()
#     print("Optimization took {0:.0f} seconds".format(t1 - t0))
#     plt.plot(res.fun)

