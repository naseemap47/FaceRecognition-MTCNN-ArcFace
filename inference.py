from keras.models import load_model
from mtcnn import MTCNN
from my_utils import alignment_procedure
import ArcFace
import cv2
import numpy as np
import pandas as pd
import math
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to saved .h5 model, eg: dir/model.h5")
# ap.add_argument("-c", "--conf", type=float, required=True,
#                 help="min prediction conf to detect pose class (0<conf<1)")
# ap.add_argument("-i", "--source", type=str, required=True,
#                 help="path to sample image")


args = vars(ap.parse_args())
# source = args["source"]
path_saved_model = args["model"]
# threshold = args["conf"]

# Load saved model
face_rec_model = load_model(path_saved_model, compile=True)

detector = MTCNN()

arcface_model = ArcFace.loadModel()
arcface_model.load_weights("arcface_weights.h5")
target_size = arcface_model.layers[0].input_shape[0][1:3]

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Error with Camera')
        break
    
    detections = detector.detect_faces(img)

    if len(detections)>0:
        for detect in detections:
            right_eye = detect['keypoints']['right_eye']
            left_eye = detect['keypoints']['left_eye']
            bbox = detect['box']
            norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)

            img_resize = cv2.resize(img, target_size)
            img_resize = np.reshape(img_resize, (1, 112, 112, 3))
            img_embedding = arcface_model.predict(img_resize)[0]

            data = pd.DataFrame([img_embedding], columns=np.arange(512))
            predict = face_rec_model.predict(data)[0]

            print(predict)

    else:
        print('[INFO] Eyes Not Detected!!')

    