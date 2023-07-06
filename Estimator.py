import copy

import numpy as np
import cv2
import matplotlib.pyplot as plt
import Model
import Utilities
import time
from scipy.spatial.transform import Rotation as Rotation
from BA import run_BA
from Parameters import TOLERANCE
from Utilities import arc_visualization, cam_visualization, cam_and_pts_visualization

class Estimator:
    def __init__(self, logger):
        self._logger = logger

    # pose estimation
    def estimate_relative_pose_and_landmarks(self, logger, t0, t1):

        # define data structure for each solver & GT
        estimations = []
        point5_solutions = {}
        triangulation_DLT_solutions = {}
        triangulation_OpenCV_solutions = {}
        point8_solutions = {}
        naive_solutions = {}
        point5_solutions['name'] = '5 point'
        triangulation_DLT_solutions['name'] = 'Triangulation DLT'
        triangulation_OpenCV_solutions['name'] = 'Triangulation OpenCV'
        point8_solutions['name'] = '8 point'
        naive_solutions['name'] = 'naive least squares'
        estimations.append(point5_solutions)
        estimations.append(point8_solutions)
        estimations.append(naive_solutions)
        # estimations.append(triangulation_DLT_solutions)
        # estimations.append(triangulation_OpenCV_solutions)


        GT_poses = {}
        GT_poses['name'] = 'GT'

        for i in range(t0, t1):
            prev = 0 # estimate R,t with relation to first pose which should be 0 translation and 0 rotation.
            curr = i+1

            # compute relative pose (returned pose is in T1 frame, T1->T0)
            GT_T_rel = Utilities.compute_pose_composition(logger[prev]['viewing_camera']._T, logger[curr]['viewing_camera']._T)

            GT_R_rel = GT_T_rel[0:3, 0:3]
            GT_t_rel = GT_T_rel[0:3, 3].transpose()
            err_GT = Utilities.calc_reprojection_error(GT_poses['name'], pts_i1=logger[prev]['projections'], pts_i2=logger[curr]['projections'], R_rel=GT_R_rel, t_rel=np.matrix(GT_t_rel).transpose(), K=logger[prev]['viewing_camera']._K ,scale=logger[prev]['points_scale'])
            GT_poses[i] = ((GT_R_rel, GT_t_rel, err_GT))

            # estimating with 5 point algorithm (opencv)
            R1_5p, R2_5p, t_5p, time_5p = self.estimate_R_t_opencv(t0=prev, t1=curr, method=cv2.FM_RANSAC)
            R1_5p = Utilities.zero_small_values(R1_5p)
            R2_5p = Utilities.zero_small_values(R2_5p)
            R_5p, t_5p = Utilities.choose_closest_solution(est=[R1_5p, R2_5p, t_5p], GT=[GT_R_rel, GT_t_rel])
            err_5p = Utilities.calc_reprojection_error(point5_solutions['name'], pts_i1=logger[prev]['projections'], pts_i2=logger[curr]['projections'], R_rel=R_5p, t_rel=t_5p, K=logger[prev]['viewing_camera']._K , scale=logger[prev]['points_scale'])
            point5_solutions[i] = ((R_5p,t_5p, err_5p, time_5p))

            # eestimating with 8 point algorithm (opencv)
            R1_8p, R2_8p, t_8p, time_8p = self.estimate_R_t_opencv(t0=prev, t1=curr, method=cv2.FM_8POINT)
            R1_8p = Utilities.zero_small_values(R1_8p)
            R2_8p = Utilities.zero_small_values(R2_8p)

            R_8p, t_8p = Utilities.choose_closest_solution(est=[R1_8p, R2_8p, t_8p], GT=[GT_R_rel, GT_t_rel])
            err_8p = Utilities.calc_reprojection_error(point8_solutions['name'], pts_i1=logger[prev]['projections'],
                                                       pts_i2=logger[curr]['projections'], R_rel=R_8p, t_rel=t_8p,
                                                       K=logger[prev]['viewing_camera']._K,
                                                       scale=logger[prev]['points_scale'])
            point8_solutions[i] = ((R_8p, t_8p, err_8p, time_8p))


            # estimating with my code - classic least square without ransac
            R1_MY, R2_MY, t_MY, time_naive = self.estimate_R_t_naive(t0=prev, t1=curr)
            R1_MY = Utilities.zero_small_values(R1_MY)
            R2_MY = Utilities.zero_small_values(R2_MY)
            R_MY, t_MY = Utilities.choose_closest_solution(est=[R1_MY, R2_MY, t_MY], GT=[GT_R_rel, GT_t_rel])
            err_MY = Utilities.calc_reprojection_error(naive_solutions['name'], pts_i1=logger[prev]['projections'], pts_i2=logger[curr]['projections'], R_rel=R_MY, t_rel=t_MY, K=logger[prev]['viewing_camera']._K , scale=logger[prev]['points_scale'])
            naive_solutions['reprojection error'] = err_MY
            naive_solutions[i] = ((R_MY, t_MY, err_MY, time_naive))


            # Triangulation using 5point algorithm results
            points3D_opencv = self.triangulation_opencv(t0=prev, t1=curr, R_r=R_5p, t_r=t_5p)
            triangulation_OpenCV_solutions[i] = points3D_opencv
            points3D_DLT =  self.dlt_triangulation(t0=prev, t1=curr, R_r=R_5p, t_r=t_5p)
            triangulation_DLT_solutions[i] = points3D_DLT

            # Bundle Adjustment
            camera_params_estimation, points_3d_estimation = self.solveBA(t0=prev, t1=curr, estimationPts=triangulation_OpenCV_solutions, estimationCams=point8_solutions, addNoise=True)

        return GT_poses, estimations
    def estimate_R_t_naive(self, t0, t1):
        """
        Estimate rotation (R) and translation (t) matrices for two cameras given time steps t0 and t1 using a naive approach.

        This function calculates the Essential matrix (E) from corresponding 2D points in two images at time steps t0 and t1.
        It constructs the A matrix using the projections and camera intrinsic matrix, solves for the Essential matrix using
        Singular Value Decomposition (SVD) and reshaping, extracts R and t from the Essential matrix, and returns the R and t matrices.

        Args:
            t0 (numpy.ndarray): Timestamp of the first frame.
            t1 (numpy.ndarray): Timestamp of the second frame.

        Returns:
            tuple: Tuple containing R1, R2, t, and the computation time.
                R1 (numpy.ndarray): First rotation matrix candidate.
                R2 (numpy.ndarray): Second rotation matrix candidate.
                t (numpy.ndarray): Translation vector.
                computation_time (float): Computation time in seconds.

        Notes:
            - The function assumes that the number of projections for both time steps is the same.
            - The function returns the inverse pose t1->t0.

        References:
            - Hartley, R., & Zisserman, A. (2004). Multiple View Geometry in Computer Vision (2nd Edition). Cambridge University Press.
              Section 9.6: The essential matrix.
        """
        assert len(self._logger[t0]['projections'])==len(self._logger[t1]['projections']) # must be same size

        tic = time.perf_counter()

        num_of_pts = len(self._logger[t0]['projections'])
        k_inv = np.linalg.pinv(self._logger[0]['viewing_camera']._K)
        # build A matrix of size Nx9, where N is number of points
        A = None
        for i in range(num_of_pts):
            X1 = self._logger[t0]['projections'][i] # 2D point
            X2 = self._logger[t1]['projections'][i] # 2D point # CONTINUE FROM HERE CONSTRUCT X_HAT
            X1_hat = k_inv @ np.matrix([X1[0].item(),X1[1].item(),1], dtype='float').transpose() # move to camera frame (lines from pinhole)
            X2_hat = k_inv @ np.matrix([X2[0].item(),X2[1].item(),1], dtype='float').transpose() # move to camera frame (lines from pinhole)
            Ai = self.construct_E_matrix_row_constraint(xy_projection_t0=X1_hat, xy_projection_t1=X2_hat)
            if A is None:
                A = Ai
            else:
                A = np.concatenate((A,Ai),axis=0)

        # solve Ax = 0 with SVD(A) and take smallest eigen vector of Vt
        (U, D, Vt) = np.linalg.svd(A)
        smallest_vec_Vt = Vt[:][8] # solution assuming SVD ordering from largest to smallest
        E = np.reshape(smallest_vec_Vt, (3,3))
        #E = E.transpose()
        E = Utilities.zero_small_values(E)

        # extract R,t from E
        (U, D, Vt) = np.linalg.svd(E)

        # # force last eigen value be 0
        new_D = np.diag((1,1,0))
        new_E = U @ new_D @ Vt
        new_E = Utilities.zero_small_values(new_E)
        (U, D, Vt) = np.linalg.svd(new_E)


        # define W utility matrix
        W = np.matrix([[0, -1, 0],
                      [1, 0, 0],
                      [0, 0, 1]])

        # calculte R options
        R1 = U @ W @ Vt
        R2 = U @ W.transpose() @ Vt

        # calculate t options
        t = U[:,2]

        toc = time.perf_counter()

        # check if Rotation matrix is close to GT
        R2_GT = self._logger[1]['viewing_camera']._R
        t2_GT = -self._logger[1]['viewing_camera']._t

        if not (np.allclose(np.asarray(R2), np.asarray(R2_GT)) or np.allclose(np.asarray(R1), np.asarray(R2_GT),
                                                                                atol=TOLERANCE)):
            print("[Rotation matrix est][LeastSquares] estimation is above tolerance value")

        # return inverse pose t1->t0
        return R1, R2, t, (toc-tic)
    def construct_E_matrix_row_constraint(self, xy_projection_t0, xy_projection_t1):
        """
        Construct a row constraint Ai for building the Essential matrix (E) from x, x' projections.

        This function constructs a row constraint Ai that is used to build the Essential matrix by solving the equation x'Ex = 0.
        The row constraint Ai is a 1x9 matrix that contains elements computed from the x, x' projections.

        Args:
            xy_projection_t0 (numpy.matrix): 2D projection in the first image.
            xy_projection_t1 (numpy.matrix): 2D projection in the second image.

        Returns:
            numpy.matrix: Row constraint Ai.

        References:
            - Hartley, R., & Zisserman, A. (2004). Multiple View Geometry in Computer Vision (2nd Edition). Cambridge University Press.
              Section 9.6: The essential matrix.
        """
        x1 = xy_projection_t0[0].A[0]
        y1 = xy_projection_t0[1].A[0]
        x2 = xy_projection_t1[0].A[0]
        y2 = xy_projection_t1[1].A[0]
        return np.matrix([(x2.item() * x1.item()), (y2.item() * x1.item()), (x1.item()), (x2.item() * y1.item()),
                          (y2.item() * y1.item()), (y1.item()), (x2.item()), (y2.item()), (1)], dtype='float')
    def estimate_R_t_opencv(self, t0, t1, method):

        tic = time.perf_counter()
        K = self._logger[0]['viewing_camera']._K
        X1 = np.array(self._logger[t0]['projections']).squeeze()
        X2 = np.array(self._logger[t1]['projections']).squeeze()
        E, _ = cv2.findEssentialMat(X1, X2, K, method=method, threshold=0.1)
        R1, R2, t = cv2.decomposeEssentialMat(E)

        R1 = Utilities.zero_small_values(R1)
        R2 = Utilities.zero_small_values(R2)
        t = Utilities.zero_small_values(t)

        R_identity = np.matrix(np.eye(3))
        t_zero = np.matrix(np.zeros((3)))
        T_zero = Utilities.construct_pose_matrix(R_identity, t_zero)
        # # testing triangulate 3d points
        T_rel = np.zeros((3,4))
        T_rel[0:3,0:3] = R2.transpose()
        T_rel[0:3,3] = t.transpose()
        T_rel = Utilities.zero_small_values(T_rel)

        toc = time.perf_counter()

        # check if Rotation matrix is close to GT
        R2_GT = self._logger[1]['viewing_camera']._R
        t2_GT = -self._logger[1]['viewing_camera']._t


        if not(np.allclose(np.asarray(R2).T, np.asarray(R2_GT)) or np.allclose(np.asarray(R1).T, np.asarray(R2_GT), atol=TOLERANCE)):
            if method == 2: #8point
                print("[Rotation matrix est][8point] estimation is above tolerance value")
            elif method == 8: #5point
                print("[Rotation matrix est][5point] estimation is above tolerance value")

        # return inverse pose (converting the relative pose from t0->t1 to t1->t0)
        return R1.transpose(),R2.transpose(),-t,(toc-tic)

    # 3D points estimation
    # def estimate_3D_points(self, logger, t0, t1):
    #
    #     # define data structure for each solver & GT
    #     estimations = []
    #     point5_solutions = {}
    #     #point7_solutions = {}
    #     point8_solutions = {}
    #     naive_solutions = {}
    #     point5_solutions['name'] = '5 point'
    #     #point7_solutions['name'] = '7 point'
    #     point8_solutions['name'] = '8 point'
    #     naive_solutions['name'] = 'naive least squares'
    #     estimations.append(point5_solutions)
    #     #estimations.append(point7_solutions)
    #     estimations.append(point8_solutions)
    #     estimations.append(naive_solutions)
    #
    #     GT_poses = {}
    #     GT_poses['name'] = 'GT'
    #
    #     for i in range(t0, t1):
    #         prev = 0 # estimate R,t with relation to first pose which should be 0 translation and 0 rotation.
    #         curr = i+1
    #
    #         # Triangulation using 5point algorithm results
    #         points3D_opencv = self.triangulation_opencv(t0, t1, R_5p, t_5p)
    #         points3D_DLT =  self.dlt_triangulation(t0, t1, R_5p, t_5p)

        return GT_poses, estimations
    def dlt_triangulation(self, t0, t1, R_r, t_r):
        """
        Perform linear triangulation (DLT) to estimate 3D points from corresponding 2D points in two images.

        Args:
            t0 (numpy.ndarray): timestamp of the first(left) frame - it is used to extract data from the logger (E.G. projection points)
            t1 (numpy.ndarray): timestamp of the second(left) frame - it is used to extract data from the logger (E.G. projection points)
            R_r (numpy.ndarray): 3x3 rotation matrix for second(right) camera.
            t_r (numpy.ndarray): 3x1 translation vector  for the second(right) camera.

        Returns:
            numpy.ndarray: list of shape (N, 3) containing estimated 3D points.

        References:
            - Hartley, R., & Zisserman, A. (2004). Multiple View Geometry in Computer Vision (2nd Edition). Cambridge University Press.
              Section 12.2.2: Linear triangulation.
        """
        K = self._logger[0]['viewing_camera']._K
        points1 = np.array(self._logger[t0]['projections']).squeeze()
        points2 = np.array(self._logger[t1]['projections']).squeeze()

        R_l = np.asarray(np.eye(3))
        t_l = np.asarray([[0,0,0]])

        P1 = K @ np.hstack((R_l, t_l.T))
        P2 = K @ np.hstack((R_r.T, -t_r))

        # Number of points

        points3D = []
        for p_idx in range(len(points1)):
            A = np.zeros((4,4))
            x1, y1 = points1[p_idx]
            x2, y2 = points2[p_idx]
            A[0] = x1 * P1[2] - P1[0]
            A[1] = y1 * P1[2] - P1[1]
            A[2] = x2 * P2[2] - P2[0]
            A[3] = y2 * P2[2] - P2[1]

            # Perform Singular Value Decomposition (SVD) on A
            _, _, V = np.linalg.svd(A)

            # Extract the solution (last column of V)
            X = V[-1, :4]

            # Convert to Cartesian coordinates
            x, y, z, w = X
            X3D = np.array([x / w, y / w, z / w])

            points3D.append(X3D.tolist())

        # visualization
        # vizualization of triangulation output
        GT_points3D = []
        for i in range(0, len(Model.model_points)):
            p = []
            p.append(Model.model_points[i].flatten()[0][0, 0])
            p.append(Model.model_points[i].flatten()[0][0, 1])
            p.append(Model.model_points[i].flatten()[0][0, 2])
            GT_points3D.append(p)
        GT_points3D = np.asarray(GT_points3D)
        arc_visualization(np.asarray(points3D), Model.model_lines, "DLT Triangulated Model")
        arc_visualization(GT_points3D, Model.model_lines, "GT Model")

        return points3D
    def triangulation_opencv(self, t0, t1, R_r, t_r):
        X1 = np.array(self._logger[t0]['projections']).squeeze()
        X2 = np.array(self._logger[t1]['projections']).squeeze()
        K = self._logger[0]['viewing_camera']._K

        R_l = np.asarray(np.eye(3))
        t_l = np.asarray([[0,0,0]])

        P1 = K @ np.hstack((R_l, t_l.T))
        P2 = K @ np.hstack((R_r.T, -t_r))  # note that we have Ri->i+1 from 8point, but we want to take the inverse of that Ri+1->i. same for translation vec.

        points1u = cv2.undistortPoints(X1, K, 0, None, K)
        points2u = cv2.undistortPoints(X2, K, 0, None, K)

        points4d = cv2.triangulatePoints(P1, P2, points1u, points2u)
        points3d = (points4d[:3, :] / points4d[3, :]).T

        # vizualization of triangulation output
        GT_points3D = []
        for i in range(0, len(Model.model_points)):
            p = []
            p.append(Model.model_points[i].flatten()[0][0, 0])
            p.append(Model.model_points[i].flatten()[0][0, 1])
            p.append(Model.model_points[i].flatten()[0][0, 2])
            GT_points3D.append(p)
        GT_points3D = np.asarray(GT_points3D)
        arc_visualization(points3d, Model.model_lines, "OpenCV Triangulated Model")
        arc_visualization(GT_points3D, Model.model_lines, "GT Model")

        return points3d

    def solveBA(self, t0, t1, estimationPts, estimationCams, addNoise=False, pts_mean=0, pts_var=0.15, cams_t_mean=0, cams_t_var=0.01,  cams_r_mean=0, cams_r_var=0.01):

         camera_params, points_3D, camera_ind, point_ind, points_2D, camera_params_GT, points_3D_GT  = self.my_prepare_data_BA(t0, t1, estimationPts, estimationCams)

         if addNoise is True:
            camera_params, points_3D = self.add_gaussian_noise_to_params(camera_params, points_3D, pts_mean, pts_var, cams_t_mean, cams_t_var,  cams_r_mean, cams_r_var)

         camera_params_solution, points_3d_solution = run_BA(camera_params, points_3D, camera_ind, point_ind, points_2D)

         # mse_cams_input_GT = np.mean((camera_params_GT - camera_params[:,:6])**2)
         # mse_pts_input_GT = np.mean((points_3D_GT - points_3D)**2)
         #
         # mse_cams_opt_GT = np.mean((camera_params_GT - camera_params_solution[:,:6])**2)
         # mse_pts_opt_GT = np.mean((points_3D_GT - points_3d_solution)**2)
         #
         # print("MSE input vs GT:")
         # print("cams: {}".format(str(mse_cams_input_GT)))
         # print("pts: {}".format(str(mse_pts_input_GT)))
         #
         # print("MSE optimized vs GT:")
         # print("cams: {}".format(str(mse_cams_opt_GT)))
         # print("pts: {}".format(str(mse_pts_opt_GT)))

         #vizualization
         arc_visualization(points_3D, Model.model_lines, "Input Model")
         arc_visualization(points_3d_solution, Model.model_lines, "Optimized Model")

         cam_visualization(camera_params_solution, camera_params, camera_params_GT)

         cam_and_pts_visualization(points_3D, points_3d_solution, camera_params, camera_params_solution, camera_params_GT, points_3D_GT)

         return camera_params_solution, points_3d_solution

    def my_prepare_data_BA(self, t0, t1, estimationPts, estimationCams):
        ############################################################
        ############################################################
        ############################################################
        ############################################################
        # camera_params with shape (n_cameras, 9) contains initial estimates of parameters for all cameras.
        # First 3 components in each row form a rotation vector ( https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula),
        # next 3 components form a translation vector, then a focal distance and two distortion parameters.
        # Format - looks like camera poses on global frame pose from world coords to frame_i coords)
        camera_params = []

        # adding the first cam (from GT)
        cam_param = []
        R = self._logger[t0]['viewing_camera']._R
        t = self._logger[t0]['viewing_camera']._t
        K = self._logger[t0]['viewing_camera']._K

        rotation_object = Rotation.from_matrix(R)
        rot_vec = rotation_object.as_rotvec()

        cam_param.append(rot_vec[0])
        cam_param.append(rot_vec[1])
        cam_param.append(rot_vec[2])

        cam_param.append(t[0,0])
        cam_param.append(t[0,1])
        cam_param.append(t[0,2])

        # cam_param.append((K[0,0] + K[1,1]) / 2)
        # cam_param.append(0) # assuming 0 distortion
        # cam_param.append(0) # assuming 0 distortion

        camera_params.append(cam_param)
        # adding other cams from estimations
        for ti in range(t0,t1):
            cam_param = []

            R = estimationCams[ti][0]
            t = estimationCams[ti][1]

            rotation_object = Rotation.from_matrix(R)
            rot_vec = rotation_object.as_rotvec()

            cam_param.append(rot_vec[0])
            cam_param.append(rot_vec[1])
            cam_param.append(rot_vec[2])

            cam_param.append(t[0].item())
            cam_param.append(t[1].item())
            cam_param.append(t[2].item())

            # cam_param.append((K[0, 0] + K[1, 1]) / 2)
            # cam_param.append(0)  # assuming 0 distortion
            # cam_param.append(0)  # assuming 0 distortion

            camera_params.append(cam_param)

        camera_params_np = np.asarray(camera_params)

        ############################################################
        ############################################################
        ############################################################
        ############################################################
        # points_3d with shape (n_points, 3) contains initial estimates of point coordinates in the world frame.
        points_3D_np = np.copy(estimationPts[t0])


        ############################################################
        ############################################################
        ############################################################
        ############################################################
        # camera_ind with shape (n_observations,) contains indices of cameras (from 0 to n_cameras - 1) involved in each observation.
                # for example camera_ind[5] = 2 means that for observation no. 5 (points_2d[5]) the viewing camera is camera no. 2 (camera_params [2])
        # point_ind with shape (n_observations,) contatins indices of points (from 0 to n_points - 1) involved in each observation.
                # for example point_ind[8] = 1 means that for observation no. 8 (points_2d[8]) the correspoinding 3D point is point no. 1 (points_3d[1])
        # points_2d with shape (n_observations, 2) contains measured 2-D coordinates of points projected on images in each observations.
        camera_ind = []
        point_ind = []
        points_2D = []
        for ti in range(t0, t1+1):
            for point_idx, proj in enumerate(self._logger[ti]['projections']):
                points_2D.append([proj[0].item(), proj[1].item()])
                camera_ind.append(ti)
                point_ind.append(point_idx)

        camera_ind_np = np.asarray(camera_ind)
        point_ind_np = np.asarray(point_ind)
        points_2D_np = np.asarray(points_2D)

        # prepare GT data:
        GT_points_3D = copy.deepcopy(np.asarray(self._logger[t0]['GT_points_3D']))
        GT_camera_params = []

        for ti in range(t0, t1+1):
            cam = []
            rotation_object = Rotation.from_matrix(self._logger[ti]['viewing_camera']._R)
            rot_vec = rotation_object.as_rotvec()

            cam.append(rot_vec[0])
            cam.append(rot_vec[1])
            cam.append(rot_vec[2])

            cam.append(self._logger[ti]['viewing_camera']._t[0,0])
            cam.append(self._logger[ti]['viewing_camera']._t[0,1])
            cam.append(self._logger[ti]['viewing_camera']._t[0,2])
            # GT_camera_params['Rvec'].append(rot_vec)
            # GT_camera_params['tvec'].append(self._logger[ti]['viewing_camera']._t)
            GT_camera_params.append(copy.deepcopy(cam))
        return camera_params_np, points_3D_np, camera_ind_np, point_ind_np, points_2D_np, np.asarray(GT_camera_params), GT_points_3D.squeeze()

    def add_gaussian_noise_to_params(self, camera_params, points_3D, pts_mean, pts_var, cams_t_mean, cams_t_var,  cams_r_mean, cams_r_var):
        # pts
        samples_pts = np.random.normal(loc=pts_mean, scale=pts_var, size=points_3D.size)
        points_3D += samples_pts.reshape(points_3D.shape)

        # cam translation
        samples_cams = np.random.normal(loc=cams_t_mean, scale=cams_t_var, size=(camera_params.shape[0]*3))
        camera_params[:,3:6] += samples_cams.reshape(camera_params.shape[0],3)

        # cam rotation
        samples_cams = np.random.normal(loc=cams_r_mean, scale=cams_r_var, size=(camera_params.shape[0]*3))
        camera_params[:,:3] += samples_cams.reshape(camera_params.shape[0],3)

        #camera_params[:,0:3] += np.random.normal(loc=pts_mean, scale=0.05, size=6).reshape((2,3))
        return camera_params, points_3D


# helper functions
# utility function that chooses samples from data without duplicates. returns sampled data
def choose_random_samples(X1,X2,samples):
    num_of_pts = len(X1)
    random_indexes = []
    while(len(random_indexes) < samples):
        i = np.random.randint(0,num_of_pts)
        if i not in random_indexes:
            random_indexes.append(i)

    X1_samples = np.matrix(X1[random_indexes[0]])
    X2_samples = np.matrix(X2[random_indexes[0]])

    for i in range(1,samples):
        X1_samples = np.concatenate((X1_samples, np.matrix(X1[random_indexes[i]])),axis=0)
        X2_samples = np.concatenate((X2_samples, np.matrix(X2[random_indexes[i]])),axis=0)


    return X1_samples, X2_samples
def find_similarty_transofrm(a, b):
    b = [np.squeeze(np.asarray(matrix)).tolist() for matrix in b]

    # Find the centroids of the point sets
    centroid1 = np.mean(a, axis=0)
    centroid2 = np.mean(b, axis=0).flatten()

    # Center the point sets around the centroids
    centered_points1 = a - centroid1
    centered_points2 = b - centroid2

    # Compute the covariance matrix
    covariance_matrix = centered_points1.T @ centered_points2

    # Perform SVD on the covariance matrix
    U, _, Vt = np.linalg.svd(covariance_matrix)

    # Calculate the rotation matrix
    rotation_matrix = U @ Vt

    # Calculate the scaling factor
    scaling_factor = np.trace(covariance_matrix) / np.trace(centered_points1.T @ centered_points1)

    # Calculate the translation vector
    translation_vector = centroid2 - scaling_factor * rotation_matrix @ centroid1

    # Apply similarity transformation to points1
    transformed_points1 = scaling_factor * a @ rotation_matrix.T + translation_vector

    # Calculate the error between transformed points1 and points2
    error = np.linalg.norm(transformed_points1 - b)

    # Print the similarity transformation components
    print("Rotation matrix:")
    print(rotation_matrix)
    print("Scaling factor:")
    print(scaling_factor)
    print("Translation vector:")
    print(translation_vector)
    print("Transformed points1:")
    print(transformed_points1)
    print("Error:")
    print(error)
