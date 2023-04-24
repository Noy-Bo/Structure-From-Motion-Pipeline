import pygame
from Camera import Camera
from Parameters import WIDTH, HEIGHT, WHITE, BLACK, POINT_SIZE, RED
from Model import Model
from Utilities import *

class Simulator:

    def __init__(self):
        pygame.display.set_caption("simulator")
        self._screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self._model = Model()
        self._logger = {}

    # draws 2D points on screen
    def draw_points(self, points):
        for (px, py) in points:
            pygame.draw.circle(self._screen, RED, (px.item(), py.item()), POINT_SIZE)

    # draw lines - given points and matches between points draws the matches
    def draw_lines(self, points, lines_matches):
        for line in lines_matches:
            l1 = (points[line[0]][0].item(), points[line[0]][1].item())
            l2 = (points[line[1]][0].item(), points[line[1]][1].item())
            pygame.draw.line(self._screen, BLACK, l1, l2)

    # writes the metadata for each synthesized frame, includes -  3D points, 2D projection, camera's [R,t] object's [R,t]
    def write_frame_data(self, GT_points_3D, projections, viewing_camera, points_transformation, frame_idx, pts_scale):
        self._logger[frame_idx] = {}
        self._logger[frame_idx]['GT_points_3D'] = GT_points_3D.copy()
        self._logger[frame_idx]['projections'] = projections.copy()
        self._logger[frame_idx]['viewing_camera'] = viewing_camera
        self._logger[frame_idx]['points_transformation'] = points_transformation
        self._logger[frame_idx]['points_scale'] = pts_scale

    # this demo mode demonstrates a camera observing a rotating 3D object
    def rotation_demo(self):
        # running pygame screen. press ESC to quit.
        render_iterations = 0
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
            # init screen
            self._screen.fill(WHITE)

            # transform & project points
            projected_pts = []
            # define camera
            camera = Camera(add_default_scale=False, is_inverse_transform=True) # note - to apply camera perspective we have to apply the  inverse transformation

            for i, p in enumerate(self._model._points):
                euler_angles = [(render_iterations * 0.01)] * 3  # define the rotation
                points_transformation = Camera(add_default_scale=True, euler_angles=euler_angles, is_inverse_transform=False)
                p = points_transformation.apply_transformation(
                    p).transpose()

                if (p[0,2] != 0):  # check if non zeros (detect singularity)
                    (px, py) = camera.project(X_camera=p.transpose())
                    projected_pts.append((px, py))

            # draw points
            self.draw_points(points=projected_pts)

            # draw lines
            self.draw_lines(points=projected_pts, lines_matches=self._model._model_lines)

            # update renderer
            pygame.display.update()

            render_iterations += 1

    # playground mode enable to synthesize scene with full control over Camera & Object transformation.
    def playground(self, camera_object_transformations):
        # running pygame screen. press ESC to quit.
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
            # init screen
            self._screen.fill(WHITE)
            #extract camera and object rotation translation
            camera_euler_angles = camera_object_transformations[0]
            camera_position = camera_object_transformations[1]
            point_euler_angles = camera_object_transformations[2]
            point_position = camera_object_transformations[3]

            # transform & project points
            projected_pts = []
            # define camera
            viewing_camera = Camera(add_default_scale=False, position=camera_position, euler_angles=camera_euler_angles, is_inverse_transform=True) # note - to apply camera perspective we have to apply the  inverse transformation

            for i, p in enumerate(self._model._points):
                # apply transformation to points:
                points_transformation = Camera(add_default_scale=True, position=point_position,
                                               euler_angles=point_euler_angles, is_inverse_transform=False)

                # apply object transformation:
                p = points_transformation.apply_transformation(p).transpose()
                # apply camera transformation
                p = viewing_camera.apply_transformation(p,inverse_transformation=True)

                if (p[2,0] != 0):  # check if non zeros (detect singularity)
                    (px, py) = viewing_camera.project(X_camera=p)
                    projected_pts.append((px, py))

            # draw points
            self.draw_points(points=projected_pts)

            # draw lines
            self.draw_lines(points=projected_pts, lines_matches=self._model._model_lines)

            # update renderer
            pygame.display.update()

    # This mode will synthesize the scenes from the input camera poses consecutively.
    def simulate_poses(self, camera_poses, hold_poses):
        # running pygame screen. press ESC to quit.
        for frame_idx, pose in enumerate(camera_poses):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()
            # init screen
            self._screen.fill(WHITE)

            #extract position and euler angles
            camera_position = pose[0]
            camera_euler_angles = pose[1]

            #points scale (both camera + landmark scale)
            pts_scale = []
            # transform & project points
            projected_pts = []
            # define camera
            viewing_camera = Camera(add_default_scale=False, position=camera_position, euler_angles=camera_euler_angles, is_inverse_transform=True) # note - to apply camera perspective we have to apply the  inverse transformation

            for i, p in enumerate(self._model._points):
                # define the object transformation
                points_transformation = Camera(add_default_scale=True, is_inverse_transform=False)
                # apply object transformation:
                p = points_transformation.apply_transformation(p).transpose()
                # apply camera transformation
                p = viewing_camera.apply_transformation(p,inverse_transformation=True)

                # save transformed point scale
                pts_scale.append(p[2])

                if (p[2,0] != 0):  # check if non zeros (detect singularity)
                    (px, py) = viewing_camera.project(X_camera=p)
                    projected_pts.append((px, py))

            # draw points
            self.draw_points(points=projected_pts)

            # draw lines
            self.draw_lines(points=projected_pts, lines_matches=self._model._model_lines)

            # write data to log (save projections, cam & object rotation translation)
            self.write_frame_data(self._model._points, projected_pts,viewing_camera,points_transformation, frame_idx, pts_scale)

            # update renderer
            pygame.display.update()

            # hold on until next step
            if hold_poses is True:
                next = False
                while next is False:
                    event = pygame.event.wait()
                    if event.type == pygame.KEYDOWN:
                        next = True

    # main function where the simulation runs.
    def run_simulator(self, camera_poses, camera_object_transformations, mode, hold_poses):

        # this demo mode demonstrates a camera observing a rotating 3D object
        if mode == "rotation demo":
            self.rotation_demo()

        # playground mode enable to synthesize scene with full control over Camera & Object transformation. The control is via  the global lists above  (poses/point R,t), see playground section above
        if mode == "playground":
            self.playground(camera_object_transformations)

        # this mode will synthesize the scenes of the input poses consecutively. to get the next scene press DOWN_ARROW_KEY
        if mode == "simulate poses":
            self.simulate_poses(camera_poses, hold_poses)

        return self._logger