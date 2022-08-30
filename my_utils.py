import numpy as np
import math
from PIL import Image
from deepface.commons import distance


# this function aligns given face in img based on left and right eye coordinates
def alignment_procedure(img, left_eye, right_eye, bbox):

    # Crop Face
    x, y, w, h = bbox
    img_roi = img[y:y+h, x:x+w]

    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction

    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock

    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = distance.findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = distance.findEuclideanDistance(
        np.array(right_eye), np.array(point_3rd))
    c = distance.findEuclideanDistance(np.array(right_eye), np.array(left_eye))

    # -----------------------

    # apply cosine rule

    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        # img = Image.fromarray(img)
        # img = np.array(img.rotate(direction * angle))

        # Image ROI
        img_roi = Image.fromarray(img_roi)
        img_roi = np.array(img_roi.rotate(direction * angle))

    # -----------------------

    return img_roi  # return img anyway
