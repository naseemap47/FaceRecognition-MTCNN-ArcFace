import tensorflow as tf
import numpy as np
import pickle
import cv2
import argparse


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default='../models/liveness.model',
	help="path to trained model")
ap.add_argument("-i", "--source", type=str, required=True,
                help="source - Video path or camera-id")
ap.add_argument("-c", "--conf", type=str, default=0.8,
                help="min prediction conf (0<conf<1)")
args = vars(ap.parse_args())

# Face Detection Caffe Model
opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="../models/deploy.prototxt",
                                            caffeModel="../models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

# Load Saved Model
print(f"[INFO] Loading Liveness Model from {args['model']}")
model = tf.keras.models.load_model(args['model'])
# le = pickle.loads(open('le.pickle', "rb").read())
class_names = ['Negative', 'Positive']

# Load Video or Camera
source = args['source']
if source.isnumeric():
    source = int(source)
cap = cv2.VideoCapture(source)

while True:
    success, frame = cap.read()
    h, w, _ = frame.shape
    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0)
    )
    opencv_dnn_model.setInput(blob)
    detections = opencv_dnn_model.forward()

    for face in detections[0][0]:
        face_confidence = face[2]
        if face_confidence > args['conf']:
            
            bbox = face[3:]
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)

            try:
                face_roi = frame[y1:y2, x1:x2]
                face_resize = cv2.resize(face_roi, (32, 32))
                face_norm = face_resize.astype("float") / 255.0
                face_array = tf.keras.preprocessing.image.img_to_array(face_norm)
                face_prepro = np.expand_dims(face_array, axis=0)

                preds = model.predict(face_prepro)[0]
                j = np.argmax(preds)
                label = class_names[j]

                # Color
                if j == 0:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                label = "{}: {:.1f}%".format(label, preds[j]*100)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                            color, 2)
            except:
                print('[INFO] Failed to Crop Face ROI')
            
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
