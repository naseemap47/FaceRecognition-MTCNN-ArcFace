import cv2
import glob
import argparse
from my_utils import alignment_procedure
from mtcnn import MTCNN


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save dir")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
path_to_save = args['save']

detector = MTCNN()


img_list = glob.glob(path_to_dir + '/*')
for img_path in img_list:
    img = cv2.imread(img_path)

    detections = detector.detect_faces(img)
    
    if len(detections)>0:
        right_eye = detections[0]['keypoints']['right_eye']
        left_eye = detections[0]['keypoints']['left_eye']
        bbox = detections[0]['box']
        norm_img = alignment_procedure(img, left_eye, right_eye, bbox)

        right_eye = [int(x) for x in right_eye]
        left_eye = [int(x) for x in left_eye]

        cv2.circle(img, right_eye, 3, (0, 255, 0), 3)
        cv2.circle(img, left_eye, 3, (0, 255, 0), 3)

        cv2.imshow('img', img)
        cv2.imshow('img_norm', norm_img)
        cv2.waitKey(0)

    else:
        print(f'[INFO] Not detected Eyes in {img_path}')

    print('detections: ', detections)
    