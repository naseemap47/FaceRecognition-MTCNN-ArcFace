import os
import cv2
import glob
import argparse
from my_utils import alignment_procedure
from mtcnn import MTCNN


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save dir")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
path_to_save = args['save']

Flage = True

detector = MTCNN()

# Checking that, Already All Data Normalized or NOT
# It will only Normalize, NOT normalized Data
# If NOT It will create new save Dir
class_list_update = []
if os.path.exists(path_to_save):
    class_list_save = os.listdir(path_to_save)
    class_list_dir = os.listdir(path_to_dir)
    class_list_update =  list(set(class_list_dir)^set(class_list_save))
else:
    os.makedirs(path_to_save)

if len(class_list_update) == 0:
    if (set(class_list_dir) == set(class_list_save)):
        Flage = False
    else:
        class_list = os.listdir(path_to_dir)
else:
    class_list = class_list_update


if Flage:
    class_list = sorted(class_list)
    for name in class_list:
        img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
        
        # Create Save Folder
        save_folder = os.path.join(path_to_save, name)
        os.makedirs(save_folder, exist_ok=True)
        
        for img_path in img_list:
            img = cv2.imread(img_path)

            detections = detector.detect_faces(img)
            
            if len(detections)>0:
                right_eye = detections[0]['keypoints']['right_eye']
                left_eye = detections[0]['keypoints']['left_eye']
                bbox = detections[0]['box']
                norm_img_roi = alignment_procedure(img, left_eye, right_eye, bbox)

                # Save Norm ROI
                cv2.imwrite(f'{save_folder}/{os.path.split(img_path)[1]}', norm_img_roi)
                print(f'[INFO] Successfully Normalised {img_path}')

            else:
                print(f'[INFO] Not detected Eyes in {img_path}')

        print(f'[INFO] Successfully Normalised All Images from {len(os.listdir(path_to_save))} Classes\n')
    print(f"[INFO] All Normalised Images Saved in '{path_to_save}'")

else:
    print('[INFO] Already Normalized All Data..')
