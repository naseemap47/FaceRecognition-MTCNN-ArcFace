import cv2
import glob
import argparse
import math
from my_utils import alignment_procedure
from retinaface import RetinaFace


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
# ap.add_argument("-c", "--classes", type=str, required=True,
#                 help="path to classes.txt")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save dir")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
# path_to_txt = args['classes']
path_to_save = args['save']


eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml') 


img_list = glob.glob(path_to_dir + '/*')
for img_path in img_list:
    # img = cv2.imread(img_path)
    resp = RetinaFace.detect_faces(img_path)
    
    # else:
    #     print(f'[INFO] Not detected Eyes in {img_path}')
    print('resp: ', resp)

    # right_eye = resp['face_1']['landmarks']['right_eye']
    # left_eye = resp['face_1']['landmarks']['left_eye']

    # right_eye = [int(x) for x in right_eye]
    # left_eye = [int(x) for x in left_eye]

    # cv2.circle(img, right_eye, 3, (0, 255, 0), 3)
    # cv2.circle(img, left_eye, 3, (0, 255, 0), 3)

    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    # norm_img = alignment_procedure(img, )