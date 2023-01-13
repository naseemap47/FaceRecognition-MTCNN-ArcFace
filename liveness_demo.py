import tensorflow as tf
import numpy as np
import pickle
import cv2


opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",
                                            caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel")


model = tf.keras.models.load_model('liveness.model')
le = pickle.loads(open('le.pickle', "rb").read())

cap = cv2.VideoCapture(0)
conf = 0.5

while True:

    success, frame = cap.read()
    # frame = cv2.resize(frame, 600)

    h, w, _ = frame.shape

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    opencv_dnn_model.setInput(blob)
    detections = opencv_dnn_model.forward()

    for face in detections[0][0]:
        face_confidence = face[2]
        if face_confidence > conf:
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
                label = le.classes_[j]

                # Color
                if j == 0:
                    color = (0, 0, 255)
                else:
                    color = (0, 255, 0)

                label = "{}: {:.4f}".format(label, preds[j])
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                            color, 2)
            except:
                print('[INFO] Failed to Crop Face ROI')
            
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
# do a bit of cleanup
cap.release()
cv2.destroyAllWindows()
