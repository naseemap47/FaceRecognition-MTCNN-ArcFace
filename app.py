import streamlit as st
import cv2
import os


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
                    st.warning('The Name is Already Exist')
                    break
        os.makedirs(f'data/{name_person}', exist_ok=True)
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
