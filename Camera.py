from Parameters import WIDTH, HEIGHT
from Utilities import *


class Camera:
    def __init__(self, euler_angles=None, position=None, K=Parameters.projection_matrix, add_default_scale=False, is_inverse_transform=False):

        # save scale
        self._scale = Parameters.DEFAULT_SCALE

        #define the projection (intrinsic) matrix
        self._K = K

        # define if this is inverse transform (for viewing camera)
        self._is_inverse_transform = is_inverse_transform

        # constructing translation vector
        if position is None:
            position = [0, 0, 0]
        else:
            position = position.copy()

        # adding default step back so the camera will be able to view the object
        if add_default_scale is True:
            position[2] += self._scale
        self._t = construct_translation_vector(position)

        # constructing rotation matrix
        if euler_angles is None:
            euler_angles = [0,0,0]
        else:
            euler_angles = euler_angles.copy()

        self._R = construct_rotation_from_euler(euler_angles)

        # construct T matrix [R | t]   (pose matrix)
        self._T = construct_pose_matrix(R=self._R, t=self._t)
        return

    # this function applies the transformation for a 3D point - [R | t] @ Point
    def apply_transformation(self, X_world, inverse_transformation = False):
        X_world = X_world.transpose()

        # augment the point to homogenous coordinates if needed
        if X_world.shape[0] == 3:
            X_world = np.matrix(list(X_world.A[0])+list(X_world.A[1])+list(X_world.A[2])+[1]).transpose()

        # apply transformation
        if inverse_transformation is True:
            T_inv = compute_pose_inverse(self._T)
            return T_inv @ X_world
        else:
            return self._T @ X_world

    #this function does the projection - K @ Point
    def project(self, X_camera):

        # project the points
        X_projected = self._K @ X_camera

        # move to pixel space
        px = X_projected[0][0] / X_projected[2][0]
        py = X_projected[1][0] / X_projected[2][0]

        return (px, py)

