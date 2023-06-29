import numpy as np

## -------------------------------- MODEL -------------------------------- ##
# this list will
model_points = [] # holds the model 3D-points
model_lines = [] # holds the matching between points (lines)
# 8 point arc.
# model_points.append(np.matrix([-2,-1,1])) # 0   - DO NOT CHANGE THE ORDER
# model_points.append(np.matrix([2,-1,1])) # 1    - DO NOT CHANGE THE ORDER
# model_points.append(np.matrix([2,1,1])) # 2     - DO NOT CHANGE THE ORDER
# model_points.append(np.matrix([-2,1,1])) # 3    - DO NOT CHANGE THE ORDER
# model_points.append(np.matrix([-2,-1,-1])) # 4  - DO NOT CHANGE THE ORDER
# model_points.append(np.matrix([2,-1,-1])) # 5   - DO NOT CHANGE THE ORDER
# model_points.append(np.matrix([2,1,-1])) # 6    - DO NOT CHANGE THE ORDER
# model_points.append(np.matrix([-2,1,-1])) # 7   - DO NOT CHANGE THE ORDER
###########

model_points.append(np.matrix([-2,-1,-20])) # 0   - DO NOT CHANGE THE ORDER
model_points.append(np.matrix([2,-1,-20])) # 1    - DO NOT CHANGE THE ORDER
model_points.append(np.matrix([2,1,-20])) # 2     - DO NOT CHANGE THE ORDER
model_points.append(np.matrix([-2,1,-20])) # 3    - DO NOT CHANGE THE ORDER
model_points.append(np.matrix([-2,-1,-15])) # 4  - DO NOT CHANGE THE ORDER
model_points.append(np.matrix([2,-1,-15])) # 5   - DO NOT CHANGE THE ORDER
model_points.append(np.matrix([2,1,-15])) # 6    - DO NOT CHANGE THE ORDER
model_points.append(np.matrix([-2,1,-15])) # 7   - DO NOT CHANGE THE ORDER

#more points
model_points.append(np.matrix([-1,-1,-20]))
model_points.append(np.matrix([0.000000001,-1,-20]))
model_points.append(np.matrix([1,-1,-20]))
model_points.append(np.matrix([2,0.000000001,-20]))
model_points.append(np.matrix([0.000000001,1,-20]))
model_points.append(np.matrix([-2,0.000000001,-20]))
model_points.append(np.matrix([-1,-1,-15]))
model_points.append(np.matrix([0.000000001,-1,-15]))
model_points.append(np.matrix([1,-1,-15]))
model_points.append(np.matrix([1,1,-15]))
model_points.append(np.matrix([0.000000001,1,-15]))
model_points.append(np.matrix([-1,1,-15]))
model_points.append(np.matrix([1,1,-20]))
model_points.append(np.matrix([-1,1,-20]))
model_points.append(np.matrix([2,0.000000001, -15]))
model_points.append(np.matrix([-2,0.000000001,-15]))
# lines matching
model_lines.append((0,1))
model_lines.append((0,3))
model_lines.append((0,4))
model_lines.append((1,2))
model_lines.append((1,5))
model_lines.append((2,6))
model_lines.append((2,3))
model_lines.append((3,7))
model_lines.append((4,7))
model_lines.append((4,5))
model_lines.append((5,6))
model_lines.append((6,7))
## --------------------------------------------------------------------------- ""


class Model:

    def __init__(self):
        self._raw_points = model_points
        self._model_lines = model_lines
        self._points = []
        #self.add_mid_points()
        self.transform_points(self._raw_points)

    # TODO: in future cases where model is not normalize - write here normalization to the model such that it will have 1 scale ( z values should be between -1 to 1).
    #       This function currently does nothing as the object i am using is already normalized.
    def transform_points(self, points):
        for i, p in enumerate(points):
            x_ = (p.flat[0])
            y_ = (p.flat[1])
            z_ = (p.flat[2])
            self._points.append(np.matrix([x_, y_, z_]))
    def add_mid_points(self):

        for line in self._model_lines:
            point1  = (model_points[line[0]].flatten()[0][0, 0], model_points[line[0]].flatten()[0][0, 1], model_points[line[0]].flatten()[0][0, 2])
            point2  = (model_points[line[1]].flatten()[0][0, 0], model_points[line[1]].flatten()[0][0, 1], model_points[line[1]].flatten()[0][0, 2])
            x = [point1[0], point2[0]]
            x.sort()
            y = [point1[1], point2[1]]
            y.sort()
            z = [point1[2], point2[2]]
            z.sort()
            third_point = np.matrix([x[0]+(x[1]-x[0])/3, y[0]+(y[1]-y[0])/3, z[0]+(z[1]-z[0])/3])
            two_thirds_point = np.matrix([x[0]+2*(x[1]-x[0])/3, y[0]+2*(y[1]-y[0])/3, z[0]+2*(z[1]-z[0])/3])
            model_points.append(third_point)
            model_points.append(two_thirds_point)





