# from deepface import DeepFace
from mtcnn import MTCNN
import cv2
from my_utils import alignment_procedure
import ArcFace
import numpy as np
from deepface.commons import distance as dst
from tensorflow.keras.preprocessing import image


# result = DeepFace.verify(img1_path = "data/Naseem/2.jpg", 
#                         img2_path = "image/img1.jpg", 
#                         model_name='ArcFace',
#                         detector_backend='mtcnn'
#                         # enforce_detection=False
#                         )
# print(result)


detector = MTCNN()
model = ArcFace.loadModel()
model.load_weights("arcface_weights.h5")
target_size = model.layers[0].input_shape[0][1:3]

img1 = cv2.imread('data/Naseem/2.jpg')
# img2 = cv2.imread("image/img1.jpg")
img2 = cv2.imread("data/Naseem/54.jpg")

detections1 = detector.detect_faces(img1)
detections2 = detector.detect_faces(img2)

right_eye = detections1[0]['keypoints']['right_eye']
left_eye = detections1[0]['keypoints']['left_eye']
bbox = detections1[0]['box']
img1 = alignment_procedure(img1, left_eye, right_eye, bbox)
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img1 = cv2.resize(img1, target_size)
# img_resize1 = np.reshape(img1_gray, (1, 112, 112, 3))
img_pixels = image.img_to_array(img1) #what this line doing? must?
img_pixels = np.expand_dims(img_pixels, axis = 0)
img1 = img_pixels/255 #normalize input in [0, 1]
img_embedding1 = model.predict(img1)[0]
print('img_embedding1: ', img_embedding1)


right_eye = detections2[0]['keypoints']['right_eye']
left_eye = detections2[0]['keypoints']['left_eye']
bbox = detections2[0]['box']
img2 = alignment_procedure(img2, left_eye, right_eye, bbox)
img2 = cv2.resize(img2, target_size)
# img2 = np.reshape(img2, (1, 112, 112, 3))
img_pixels = image.img_to_array(img2) #what this line doing? must?
img_pixels = np.expand_dims(img_pixels, axis = 0)
img2 = img_pixels/255 #normalize input in [0, 1]
img_embedding2 = model.predict(img2)[0]
print('img_embedding2: ', img_embedding2)

distance = dst.findCosineDistance(img_embedding1, img_embedding2)
print('distance: ', distance)

# cv2.imshow('img1', norm_img_roi1)
# cv2.imshow('img2', norm_img_roi2)
# cv2.waitKey(0)
