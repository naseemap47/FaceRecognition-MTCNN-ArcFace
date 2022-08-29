from retinaface import RetinaFace
import cv2
# # import pandas as pd

# resp = RetinaFace.detect_faces("image/img1.jpg")
# print(resp['face_1'])

# import matplotlib.pyplot as plt
face = RetinaFace.extract_faces(img_path = "image/img1.jpg", align = True)

cv2.imshow('face', face[0])
cv2.waitKey(0)