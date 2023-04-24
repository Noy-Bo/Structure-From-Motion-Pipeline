import numpy as np
## ------------ Renderer CFG ------------ ##
# colors
WHITE = (255,255,255)
RED = (255,0,0)
BLUE = (0,255,0)
GREEN = (0,0,255)
BLACK = (0,0,0)

# screen size
WIDTH = 800
HEIGHT = 600
ASPECT_RATIO = WIDTH/HEIGHT

# drawing point size
POINT_SIZE = 2
## --------------------------------------- ##

## ------------ Calculations ------------- ##
TOLERANCE = 1e-3
## --------------------------------------- ##

## ---- Projection (Intrinsic) matrix ---- ##
FOCAL_LENGTH = 500
projection_matrix = np.matrix([[FOCAL_LENGTH,0,WIDTH/2],
                              [0,FOCAL_LENGTH,HEIGHT/2],
                              [0,0,1]])
## ---------------------------------------- ##

## -------- camera configurtations -------- ##
# step back is used due to same starting point
# of the camera and object adding a stepback
# to camera will allow the camera# to
# observe the object.
DEFAULT_SCALE = -20
## ---------------------------------------- ##
