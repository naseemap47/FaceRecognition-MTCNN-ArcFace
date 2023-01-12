import cv2
import matplotlib.pyplot as plt
from time import time
from mtcnn import MTCNN


opencv_dnn_model = cv2.dnn.readNetFromCaffe(prototxt="models/deploy.prototxt",
                                            caffeModel="models/res10_300x300_ssd_iter_140000_fp16.caffemodel")

detector = MTCNN()


image = cv2.imread('Liveness/negative/5.jpg')
mtcnn_img = image.copy()


def cvDnnDetectFaces(image, opencv_dnn_model, min_confidence=0.5, display = True):
    image_height, image_width, _ = image.shape
    output_image = image.copy()
    preprocessed_image = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300),
                                               mean=(104.0, 117.0, 123.0), swapRB=False, crop=False)

    opencv_dnn_model.setInput(preprocessed_image)
    start = time()
    results = opencv_dnn_model.forward()    

    end = time()
    for face in results[0][0]:
        
        face_confidence = face[2]
        
        if face_confidence > min_confidence:

            bbox = face[3:]

            x1 = int(bbox[0] * image_width)
            y1 = int(bbox[1] * image_height)
            x2 = int(bbox[2] * image_width)
            y2 = int(bbox[3] * image_height)

            cv2.rectangle(output_image, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=image_width//200)

            cv2.rectangle(output_image, pt1=(x1, y1-image_width//20), pt2=(x1+image_width//16, y1),
                          color=(0, 255, 0), thickness=-1)

            cv2.putText(output_image, text=str(round(face_confidence, 1)), org=(x1, y1-25), 
                        fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=image_width//700,
                        color=(255,255,255), thickness=image_width//200)

    # if display:

        # cv2.putText(output_image, text='Time taken: '+str(round(end - start, 2))+' Seconds.', org=(10, 65),
        #             fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=image_width//700,
        #             color=(0,0,255), thickness=image_width//500)
        
        # plt.figure(figsize=[20,20])
        # plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        # plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output");plt.axis('off');
        
    # else:
        
    return output_image, results


output_image, results = cvDnnDetectFaces(image, opencv_dnn_model, min_confidence=0.5, display = False)
# print(output_image, results)

# MTCNN
detections = detector.detect_faces(image)
if len(detections) > 0:
    for detect in detections:
        right_eye = detect['keypoints']['right_eye']
        left_eye = detect['keypoints']['left_eye']
        bbox = detect['box']
        xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), \
                    int(bbox[2]+bbox[0]), int(bbox[3]+bbox[1])

        cv2.rectangle(
            mtcnn_img, (xmin, ymin), (xmax, ymax),
            (0, 255, 255), 2
        )

cv2.imshow('img', output_image)
cv2.imshow('MTCNN', mtcnn_img)

cv2.waitKey(0)