import os
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, required=True,
                help="name of the person")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save dir")


args = vars(ap.parse_args())
name_of_person = args['name']
path_to_save = args['save']

os.makedirs((os.path.join(path_to_save, name_of_person)), exist_ok=True)
path_to_save = os.path.join(path_to_save, name_of_person)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
count = 0
while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Cam NOT working!!')
        break

    # Save Image
    cv2.imwrite(f'{path_to_save}/{count}.jpg', img)
    print(f'[INFO] Successfully Saved {count}.jpg')
    count += 1

    faces = face_classifier.detectMultiScale(img)

    for (x, y, w, h) in faces:
        cv2.rectangle(
            img, (x, y), (x+w, y+h),
            (0, 255, 0), 2
        )

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    if count == 100:
        print('[INFO] Collected 100 Images')
        cv2.destroyAllWindows()
        break
