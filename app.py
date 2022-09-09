import streamlit as st
import cv2
import os
from my_utils import alignment_procedure
from mtcnn import MTCNN
import glob


st.title('Face Recognition System')
os.makedirs('data', exist_ok=True)
name_list = os.listdir('data')

st.sidebar.title('Data Collection')
webcam_channel = st.sidebar.selectbox(
    'Webcam Channel:',
    ('Select Channel', '0', '1', '2', '3')
)
name_person = st.text_input('Name of the Person:')
img_number = st.number_input('Number of Images:', 50)
FRAME_WINDOW = st.image([])

if not webcam_channel == 'Select Channel':
    take_img = st.button('Take Images')
    if take_img:
        if len(name_list)!= 0:
            for i in name_list:
                if i == name_person:
                    st.warning('The Name is Already Exist!!')
                    break
        os.mkdir(f'data/{name_person}')
        st.success(f'{name_person} added Successfully')

        if len(os.listdir(f'data/{name_person}'))==0:
            face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            cap = cv2.VideoCapture(int(webcam_channel))
            count = 0
            while True:
                success, img = cap.read()
                if not success:
                    st.error('[INFO] Cam NOT working!!')
                    break

                # Save Image
                cv2.imwrite(f'data/{name_person}/{count}.jpg', img)
                st.success(f'[INFO] Successfully Saved {count}.jpg')
                count += 1

                faces = face_classifier.detectMultiScale(img)
                for (x, y, w, h) in faces:
                    cv2.rectangle(
                        img, (x, y), (x+w, y+h),
                        (0, 255, 0), 2
                    )

                FRAME_WINDOW.image(img, channels='BGR')
                if count == img_number:
                    st.success(f'[INFO] Collected {img_number} Images')
                    break
                
            FRAME_WINDOW.image([])
            cap.release()
            cv2.destroyAllWindows()

    # 2nd Stage - Normalize Image Data
    st.sidebar.text('Go to Next Stage:')
    if st.sidebar.button('Completed', help='If Data Collection Completed'):
        path_to_dir = "data"
        path_to_save = 'norm_data'
              
        Flage = True
        detector = MTCNN()

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
                st.success(f"[INFO] Class '{name}' Started Normalising")
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
                        # st.success(f'[INFO] Successfully Normalised {img_path}')

                    else:
                        st.warning(f'[INFO] Not detected Eyes in {img_path}')

                st.success(f"[INFO] All Normalised Images from '{name}' Saved in '{path_to_save}'")
            st.success(f'[INFO] Successfully Normalised All Images from {len(os.listdir(path_to_dir))} Classes\n')

        else:
            st.warning('[INFO] Already Normalized All Data..')

