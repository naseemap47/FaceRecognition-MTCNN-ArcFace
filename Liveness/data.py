import os
import cv2
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--source", type=str, required=True,
                help="source - Video path or camera-id")
ap.add_argument("-n", "--name", type=str, required=True,
                choices=['positive', 'negative'],
                help="poitive or negative")
                

args = vars(ap.parse_args())
class_name = args['name']
path_to_vid = args['source']
min_confidence = 0.6

path_to_save_dir = os.path.join('data', class_name)
os.makedirs(path_to_save_dir, exist_ok=True)

# Face Detetcion - Caffe Model
opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="../models/deploy.prototxt",
                                            caffeModel="../models/res10_300x300_ssd_iter_140000_fp16.caffemodel")


if path_to_vid.isnumeric():
    path_to_vid = int(path_to_vid)

cap = cv2.VideoCapture(path_to_vid)
count = 0

while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Cam NOT working!!')
        break

    h, w, _ = img.shape
    
    # Face Detection - Caffe Model
    preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)
    
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

            # ROI
            img_roi = img[y1:y2, x1:x2]
            
            # Save ROI
            img_name = len(os.listdir(path_to_save_dir)) + 1
            path_to_save = os.path.join(path_to_save_dir, str(img_name)) + '.jpg'
            if count % 5 == 0:
                cv2.imwrite(path_to_save, img_roi)
                print(f'[INFO] {path_to_save} Saved Successfully')
            
            # Draw Rectangle
            cv2.rectangle(
                img, (x1, y1), (x2, y2),
                (0, 255, 0), w//200
            )
            count += 1

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        print(f'[INFO] Collected Image {class_name} Data')
        cv2.destroyAllWindows()
        break
