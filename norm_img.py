import cv2
import glob
import argparse
import math
from my_utils import alignment_procedure


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
    img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    eyes = eye_detector.detectMultiScale(gray_img)
    if len(eyes) != 0:
        for (x, y, w, h) in eyes:
            cv2.rectangle(
                img, (x, y), (x+w, y+h),
                (0, 255, 0), 2
            )
    else:
        print(f'[INFO] Not detected Eyes in {img_path}')
    print('eyes: ', eyes)

    cv2.imshow('img', img)
    cv2.waitKey(0)

    # norm_img = alignment_procedure(img, )