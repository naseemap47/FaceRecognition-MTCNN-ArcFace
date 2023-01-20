import os
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--source", type=str, required=True,
                help="RTSP link or webcam-id")
ap.add_argument("-n", "--name", type=str, required=True,
                help="name of the person")
ap.add_argument("-o", "--save", type=str, default='Data',
                help="path to save dir")
ap.add_argument("-c", "--conf", type=float, default=0.5,
                help="min prediction conf (0<conf<1)")
ap.add_argument("-x", "--number", type=int, default=100,
                help="number of data wants to collect")


args = vars(ap.parse_args())
source = args["source"]
name_of_person = args['name']
path_to_save = args['save']
min_confidence = args["conf"]

os.makedirs((os.path.join(path_to_save, name_of_person)), exist_ok=True)
path_to_save = os.path.join(path_to_save, name_of_person)

opencv_dnn_model = cv2.dnn.readNetFromCaffe(
    prototxt="models/deploy.prototxt",
    caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

if source.isnumeric():
    source = int(source)
cap = cv2.VideoCapture(source)
fps = cap.get(cv2.CAP_PROP_FPS)

count = 0
while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Cam NOT working!!')
        break

    # Save Image
    if count % int(fps/5) == 0:
        img_name = len(os.listdir(path_to_save))
        cv2.imwrite(f'{path_to_save}/{img_name}.jpg', img)
        print(f'[INFO] Successfully Saved {img_name}.jpg')
    count += 1

    # Caffe Model - Face Detection
    h, w, _ = img.shape
    preprocessed_image = cv2.dnn.blobFromImage(
        img, scalefactor=1.0, size=(300, 300),
        mean=(104.0, 117.0, 123.0), swapRB=False, crop=False
    )
    opencv_dnn_model.setInput(preprocessed_image)
    results = opencv_dnn_model.forward() 

    for face in results[0][0]:
        face_confidence = face[2]
        if face_confidence > min_confidence:
            bbox = face[3:]
            x1 = int(bbox[0] * w)
            y1 = int(bbox[1] * h)
            x2 = int(bbox[2] * w)
            y2 = int(bbox[3] * h)

            cv2.rectangle(
                img, pt1=(x1, y1), pt2=(x2, y2),
                color=(0, 255, 0), thickness=w//200
            )
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
    if img_name == args["number"]-1:
        print(f"[INFO] Collected {args['number']} Images")
        cv2.destroyAllWindows()
        break
