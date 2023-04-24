import numpy as np
import cv2
from matplotlib import pyplot as plt

import Utilities
import time

class Estimator:
    def __init__(self, logger):
        self._logger = logger

    # My implementation for estimating rotation (R) and translation (t) matrices
    # for two cameras given time steps t0 and t1. It first asserts that
    # the number of projections for both time steps are the same.
    # It then constructs the A matrix using the projections and camera intrinsic matrix,
    # solves for the Essential matrix using SVD and reshaping, extracts
    # R and t from the Essential matrix, and returns the R and t matrices.
    # steps:
    # calculate E matrix from x'Ex = 0 over all projections
    # construct Ax=0 where each row corresponds to some x,x' constraint
    # solve Ax=0 with SVD while forcing last eigenvalue to be 0
    # NOTE that there are 4 candidates for solution {R1, R2} X {t, -t}
    # returns inverse pose  t1->t0
    def estimate_R_t_naive(self, t0, t1):

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
            Ai = construct_E_matrix_row_constraint(xy_projection_t0=X1_hat, xy_projection_t1=X2_hat)
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
        # return inverse pose t1->t0
        return R1, R2, t, (toc-tic)

    # This function estimates the rotation (R) and translation (t) matrices for two cameras
    # given time steps t0 and t1 using RANSAC algorithm. It first gets the camera
    # intrinsic matrix K, the corresponding projections for time steps t0 and t1,
    # and then finds the essential matrix(E) using the OpenCV function cv2.findEssentialMat().
    # It then uses OpenCV's cv2.recoverPose() function to recover the relative camera pose,
    # including the number of inliers, R, t, and the inlier mask. It returns the R and t matrices.
    # def estimate_R_t_RANSAC(self, t0, t1, RANSAC_iteration=100):
    #     K = self._logger[0]['viewing_camera']._K
    #     X1 = np.array(self._logger[t0]['projections']).squeeze()
    #     X2 = np.array(self._logger[t1]['projections']).squeeze()
    #
    #     best_inliers_count = 0
    #     best_R = None
    #     best_t = None
    #
    #     for i in range(0,RANSAC_iteration):
    #         X1_samples, X2_samples = choose_random_samples(X1,X2, samples=6)
    #         E, _ = cv2.findEssentialMat(X1_samples, X2_samples, K)
    #         num_inliers, R, t, _ = cv2.recoverPose(E, X1, X2, K) # recover pose performs cheirality constraint (positive depth)
    #         if num_inliers > best_inliers_count:
    #             best_R = R
    #             best_t = t
    #     # return inverse pose (from points perspective to camera perspective)
    #     return best_R.transpose(), -best_t

    #This function estimates the rotation (R1,R2) and translation (t) matrices
    # for two cameras given time steps t0 and t1 using the SVD algorithm.
    # It first gets the camera intrinsic matrix K, the corresponding projections
    # for time steps t0 and t1, and then finds the essential matrix(E) using the
    # OpenCV function cv2.findEssentialMat(). It then uses OpenCV's
    # cv2.decomposeEssentialMat() function to decompose the essential matrix into its
    # relative rotation (R1, R2) and translation (t) matrices. It returns the R1, R2 and t matrices
    # NOTE that there are 4 candidates for solution {R1, R2} X {t, -t}
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
        T_rel[0:3,3] = -t.transpose()
        T_rel = Utilities.zero_small_values(T_rel)
        X1_undist = cv2.undistortPoints(X1,K,None)
        X2_undist = cv2.undistortPoints(X2,K,None)
        pts3D = cv2.triangulatePoints(K@T_zero, K@T_rel,X1_undist, X2_undist)

        # normalizing homoegenous coordinates
        #normalized_pts3D = pts3D[0:3] / pts3D[3]
        normalized_pts3D = cv2.convertPointsFromHomogeneous(pts3D.T)
        #checking if points lie on plane with svd
        u, v, d = np.linalg.svd(normalized_pts3D.T)
        print(v)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')


        for i in range(0,normalized_pts3D.shape[0]):
            xs = normalized_pts3D[i][0][0]
            ys = normalized_pts3D[i][0][1]
            zs = normalized_pts3D[i][0][2]
            ax.scatter(xs, ys, zs)

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # project back
        # Reproject back into the two cameras
        rvec1, _ = cv2.Rodrigues(R_identity)  # Change
        rvec2, _ = cv2.Rodrigues(R1)  # Change
        rvec3, _ = cv2.Rodrigues(R2)  # Change
        rvec1t, _ = cv2.Rodrigues(R_identity.T)  # Change
        rvec2t, _ = cv2.Rodrigues(R1.T)  # Change
        rvec3t, _ = cv2.Rodrigues(R2.T)  # Change

        p1, _ = cv2.projectPoints(normalized_pts3D, rvec1, t_zero, K, distCoeffs=None)  # Change
        p2, _ = cv2.projectPoints(normalized_pts3D, rvec2, t, K, distCoeffs=None)  # Change
        p3, _ = cv2.projectPoints(normalized_pts3D, rvec3, t, K, distCoeffs=None)  # Change
        p4, _ = cv2.projectPoints(normalized_pts3D, rvec1, -t_zero, K, distCoeffs=None)  # Change
        p5, _ = cv2.projectPoints(normalized_pts3D, rvec2, -t, K, distCoeffs=None)  # Change
        p6, _ = cv2.projectPoints(normalized_pts3D, rvec3, -t, K, distCoeffs=None)
        p7, _ = cv2.projectPoints(normalized_pts3D, rvec1t, -t_zero, K, distCoeffs=None)  # Change
        p8, _ = cv2.projectPoints(normalized_pts3D, rvec2t, -t, K, distCoeffs=None)  # Change
        p9, _ = cv2.projectPoints(normalized_pts3D, rvec3t, -t, K, distCoeffs=None)
        p10, _ = cv2.projectPoints(normalized_pts3D, rvec1t, t_zero, K, distCoeffs=None)  # Change
        p11, _ = cv2.projectPoints(normalized_pts3D, rvec2t, t, K, distCoeffs=None)  # Change
        p12, _ = cv2.projectPoints(normalized_pts3D, rvec3t, t, K, distCoeffs=None)
        p1not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec1, t_zero, K, distCoeffs=None)  # Change
        p2not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec2, t, K, distCoeffs=None)  # Change
        p3not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec3, t, K, distCoeffs=None)  # Change
        p4not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec1, -t_zero, K, distCoeffs=None)  # Change
        p5not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec2, -t, K, distCoeffs=None)  # Change
        p6not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec3, -t, K, distCoeffs=None)
        p7not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec1t, -t_zero, K, distCoeffs=None)  # Change
        p8not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec2t, -t, K, distCoeffs=None)  # Change
        p9not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec3t, -t, K, distCoeffs=None)
        p10not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec1t, t_zero, K, distCoeffs=None)  # Change
        p11not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec2t, t, K, distCoeffs=None)  # Change
        p12not_norm, _ = cv2.projectPoints(pts3D[0:3], rvec3t, t, K, distCoeffs=None)

        toc = time.perf_counter()

        # return inverse pose (converting the relative pose from t0->t1 to t1->t0)
        return R1.transpose(),R2.transpose(),-t,(toc-tic)


    # predicting relative pose t1->t0 [R,t]
    def estimate_relative_pose(self, logger, t0, t1):

        # define data structure for each solver & GT
        estimations = []
        point5_solutions = {}
        #point7_solutions = {}
        point8_solutions = {}
        naive_solutions = {}
        point5_solutions['name'] = '5 point'
        #point7_solutions['name'] = '7 point'
        point8_solutions['name'] = '8 point'
        naive_solutions['name'] = 'naive least squares'
        estimations.append(point5_solutions)
        #estimations.append(point7_solutions)
        estimations.append(point8_solutions)
        estimations.append(naive_solutions)

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

            # # estimating with 7 point algorithm (opencv)
            # R1_7p, R2_7p, t_7p, time_7p = self.estimate_R_t_opencv(t0=prev, t1=curr, method=cv2.FM_7POINT)
            # R1_7p = Utilities.zero_small_values(R1_7p)
            # R2_7p = Utilities.zero_small_values(R2_7p)
            # R_7p, t_7p = Utilities.choose_closest_solution(est=[R1_7p, R2_7p, t_7p], GT=[GT_R_rel, GT_t_rel])
            # err_7p = Utilities.calc_reprojection_error(point5_solutions['name'], pts_i1=logger[prev]['projections'], pts_i2=logger[curr]['projections'], R_rel=R_7p, t_rel=t_7p, K=logger[prev]['viewing_camera']._K , scale=logger[prev]['points_scale'])
            # point7_solutions[i] = ((R_7p,t_7p, err_7p, time_7p))

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

        return GT_poses, estimations

# construct Ai row constraint from x,x' projections.
# [x2x1 , y2x1, x1, x2y1, y2y1, y1, x2, y2, 1] * E = 0
def construct_E_matrix_row_constraint(xy_projection_t0, xy_projection_t1):
    x1 = xy_projection_t0[0].A[0]
    y1 = xy_projection_t0[1].A[0]
    x2 = xy_projection_t1[0].A[0]
    y2 = xy_projection_t1[1].A[0]
    return np.matrix([(x2.item()*x1.item()), (y2.item()*x1.item()), (x1.item()), (x2.item()*y1.item()), (y2.item()*y1.item()), (y1.item()), (x2.item()), (y2.item()), (1)], dtype='float')

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

