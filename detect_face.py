from retinaface import RetinaFace
from deepface.commons import functions, distance as dst
import cv2
import ArcFace
# # import pandas as pd

# resp = RetinaFace.detect_faces("image/img1.jpg")
# print(resp['face_1'])

# face = RetinaFace.extract_faces(img_path = "image/img1.jpg", align = True)

# cv2.imshow('face', face[0])
# cv2.waitKey(0)

model = ArcFace.loadModel()
model.load_weights("arcface_weights.h5")
print("ArcFace expects ",model.layers[0].input_shape[0][1:]," inputs")
print("and it represents faces as ", model.layers[-1].output_shape[1:]," dimensional vectors")
target_size = model.layers[0].input_shape[0][1:3]
# print(target_size)
# backends = [
#   'opencv', 
#   'ssd', 
#   'dlib', 
#   'mtcnn', 
#   'retinaface', 
#   'mediapipe'
# ]

detector_backend = 'retinaface'

img1 = functions.preprocess_face("image/img1.jpg", target_size=target_size, detector_backend=detector_backend, align = True)

# metrics = ["cosine", "euclidean", "euclidean_l2"]

# metric = 'euclidean'
# def findThreshold(metric):
#     if metric == 'cosine':
#         return 0.6871912959056619
#     elif metric == 'euclidean':
#         return 4.1591468986978075
#     elif metric == 'euclidean_l2':
#         return 1.1315718048269017

img1_embedding = model.predict(img1)[0]

print(img1_embedding)
