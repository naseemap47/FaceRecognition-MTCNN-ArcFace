import os
import cv2
import argparse


ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--dataset", type=str, required=True,
#                 help="path to dataset/dir")
# ap.add_argument("-c", "--classes", type=str, required=True,
#                 help="path to classes.txt")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save dir")


args = vars(ap.parse_args())
# path_to_dir = args["dataset"]
# path_to_txt = args['classes']
path_to_save = args['save']

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
count = 0
while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Cam NOT working!!')
        break

    faces = face_classifier.detectMultiScale(img)

    for (x, y, w, h) in faces:
        cv2.rectangle(
            img, (x, y), (x+w, y+h),
            (0, 255, 0), 2
        )
        img_roi = img[y:y+h, x:x+w]
        cv2.imwrite(f'{path_to_save}/{count}.jpg', img_roi)
        print(f'[INFO] Successfully Saved {count}.jpg')
        count += 1

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    if count == 50:
        print('[INFO] Collected 50 Images')
        cv2.destroyAllWindows()
        break

