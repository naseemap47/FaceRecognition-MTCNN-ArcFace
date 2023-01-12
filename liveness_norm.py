import cv2
import os


path_to_img_dir = 'Liveness/cam'
path_to_img_save = 'Liveness/data'
min_confidence = 0.5


opencv_dnn_model = cv2.dnn.readNetFromCaffe(
    prototxt="models/deploy.prototxt",
    caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

for class_dir in os.listdir(path_to_img_dir):
    class_dir_path = os.path.join(path_to_img_dir, class_dir)
    for img_path in os.listdir(class_dir_path):
        img_path_full = os.path.join(class_dir_path, img_path)

        img = cv2.imread(img_path_full)
        height, width, _ = img.shape
        preprocessed_image = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

        opencv_dnn_model.setInput(preprocessed_image)
        results = opencv_dnn_model.forward()  
        for face in results[0][0]:
            face_confidence = face[2]
            if face_confidence > min_confidence:
                bbox = face[3:]
                x1 = int(bbox[0] * width)
                y1 = int(bbox[1] * height)
                x2 = int(bbox[2] * width)
                y2 = int(bbox[3] * height)

                # ROI
                img_roi = img[y1:y2, x1:x2]
                
                # Save ROI
                path_to_save = os.path.join(path_to_img_save, class_dir, img_path)
                cv2.imwrite(path_to_save, img_roi)
                print(f'[INFO] {path_to_save} Saved Successfully')

                # cv2.imshow('roi', img_roi)
                # if cv2.waitKey(0) & 0xFF == ord('q'):
                #     continue
