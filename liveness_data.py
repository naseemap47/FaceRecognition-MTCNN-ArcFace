import os
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", type=str, required=True,
                choices=['positive', 'negative'],
                help="name of the person")
                

args = vars(ap.parse_args())
name_of_person = args['name']
path_to_save = 'Liveness'

os.makedirs((os.path.join(path_to_save, name_of_person)), exist_ok=True)
path_to_save_dir = os.path.join(path_to_save, name_of_person)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
count = 0

while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Cam NOT working!!')
        break
    
    img_name = len(os.listdir(path_to_save_dir))
    
    # Save Image
    if count % 5 == 0:
        cv2.imwrite(f'{path_to_save_dir}/{img_name}.jpg', img)
        print(f'[INFO] Successfully Saved {img_name}.jpg')
    count += 1

    faces = face_classifier.detectMultiScale(img)

    for (x, y, w, h) in faces:
        cv2.rectangle(
            img, (x, y), (x+w, y+h),
            (0, 255, 0), 2
        )

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        print(f'[INFO] Collected Image {name_of_person} Data')
        cv2.destroyAllWindows()
        break
