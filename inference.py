from keras.models import load_model
from mtcnn import MTCNN
from my_utils import alignment_procedure
import tensorflow as tf
import ArcFace
import cv2
import numpy as np
import pandas as pd
import argparse
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--source", type=str, required=True,
                help="path to Video or webcam")
ap.add_argument("-m", "--model", type=str, default='models/model.h5',
                help="path to saved .h5 model, eg: dir/model.h5")
ap.add_argument("-c", "--conf", type=float, default=0.9,
                help="min prediction conf (0<conf<1)")

# Liveness Model
ap.add_argument("-lm", "--liveness_model", type=str, default='models/liveness.model',
                help="path to liveness.model")
ap.add_argument("-le", "--label_encoder", type=str, default='models/le.pickle',
                help="path to label encoder")

args = vars(ap.parse_args())
source = args["source"]
path_saved_model = args["model"]
threshold = args["conf"]

if source.isnumeric():
    source = int(source)

# Load saved FaceRecognition Model
face_rec_model = load_model(path_saved_model, compile=True)

# Load MTCNN
detector = MTCNN()

# Load ArcFace Model
arcface_model = ArcFace.loadModel()
target_size = arcface_model.layers[0].input_shape[0][1:3]

# Liveness Model
liveness_model = tf.keras.models.load_model(args['liveness_model'])
label_encoder = pickle.loads(open(args["label_encoder"], "rb").read())

cap = cv2.VideoCapture(source)

while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Error with Camera')
        break

    detections = detector.detect_faces(img)
    if len(detections) > 0:
        for detect in detections:
            
            bbox = detect['box']
            xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), \
                    int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])
        
            # Liveness
            img_roi = img[ymin:ymax, xmin:xmax]
            face_resize = cv2.resize(img_roi, (32, 32))
            face_norm = face_resize.astype("float") / 255.0
            face_array = tf.keras.preprocessing.image.img_to_array(face_norm)
            face_prepro = np.expand_dims(face_array, axis=0)
            preds_liveness = liveness_model.predict(face_prepro)[0]
            decision = np.argmax(preds_liveness)
            
            # Liveness-Decision
            if decision == 0:
                # Show Decision
                cv2.rectangle(
                    img, (xmin, ymin), (xmax, ymax),
                    (0, 0, 255), 2
                )
                cv2.putText(
                    img, 'Fake',
                    (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN,
                    2, (0, 0, 255), 2
                )
                
            else:
                # Real
                right_eye = detect['keypoints']['right_eye']
                left_eye = detect['keypoints']['left_eye']
                
                norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)
                img_resize = cv2.resize(norm_img_roi, target_size)
                img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_norm = img_pixels/255  # normalize input in [0, 1]
                img_embedding = arcface_model.predict(img_norm)[0]

                data = pd.DataFrame([img_embedding], columns=np.arange(512))

                predict = face_rec_model.predict(data)[0]
                if max(predict) > threshold:
                    class_id = predict.argmax()
                    pose_class = label_encoder.classes_[class_id]
                    color = (0, 255, 0)
                else:
                    pose_class = 'Unkown Person'
                    color = (0, 0, 255)
                
                # Show Result
                cv2.rectangle(
                    img, (xmin, ymin), (xmax, ymax),
                    color, 2
                )
                cv2.putText(
                    img, f'{pose_class}',
                    (xmin, ymin-10), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 0, 255), 2
                )

    else:
        print('[INFO] Eyes Not Detected!!')

    cv2.imshow('Output Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
print('[INFO] Inference on Videostream is Ended...')
