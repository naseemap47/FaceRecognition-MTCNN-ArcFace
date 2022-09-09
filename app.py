import streamlit as st
import cv2
import os
from my_utils import alignment_procedure
from mtcnn import MTCNN
import glob
import ArcFace
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.preprocessing import image


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


    # 3rd Stage - Train Model
    st.sidebar.text('Go to Next Step:')
    if st.sidebar.button('Train Model'):
        path_to_dir = "norm_data"
        path_to_save = 'model.h5'
        
        # Load ArcFace Model
        model = ArcFace.loadModel()
        target_size = model.layers[0].input_shape[0][1:3]

        # Variable for store img Embedding
        x = []
        y = []

        names = os.listdir(path_to_dir)
        names = sorted(names)
        class_number = len(names)

        for name in names:
            img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
            img_list = sorted(img_list)
            st.success(f'[INFO] Started {name} Part')

            for img_path in img_list:
                img = cv2.imread(img_path)
                img_resize = cv2.resize(img, target_size)
                # what this line doing? must?
                img_pixels = image.img_to_array(img_resize)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_norm = img_pixels/255  # normalize input in [0, 1]
                img_embedding = model.predict(img_norm)[0]

                x.append(img_embedding)
                y.append(name)

            st.success(f'[INFO] Completed {name} Part')
        st.success('[INFO] Image Data Embedding Completed...')

        # Model Training
        # DataFrame
        df = pd.DataFrame(x, columns=np.arange(512))
        df['names'] = y

        x = df.copy()
        y = x.pop('names')
        y, _ = y.factorize()
        x = x.astype('float64')
        y = keras.utils.to_categorical(y)

        # Train Deep Neural Network
        x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                            test_size=0.2,
                                                            random_state=0)

        model = Sequential([
            layers.Dense(1024, activation='relu', input_shape=[512]),
            layers.Dense(512, activation='relu'),
            layers.Dense(class_number, activation="softmax")
        ])

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Add a checkpoint callback to store the checkpoint that has the highest
        # validation accuracy.
        checkpoint_path = path_to_save
        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                    monitor='val_accuracy',
                                                    verbose=1,
                                                    save_best_only=True,
                                                    mode='max')
        earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                    patience=20)

        st.success('[INFO] Model Training Started ...')
        # Start training
        history = model.fit(x_train, y_train,
                            epochs=200,
                            batch_size=16,
                            validation_data=(x_test, y_test),
                            callbacks=[checkpoint, earlystopping])

        st.success('[INFO] Model Training Completed')
        st.success(f'[INFO] Model Successfully Saved in ./{path_to_save}')

        # Plot History
        metric_loss = history.history['loss']
        metric_val_loss = history.history['val_loss']
        metric_accuracy = history.history['accuracy']
        metric_val_accuracy = history.history['val_accuracy']

        # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
        epochs = range(len(metric_loss))

        # Plot the Graph.
        plt.plot(epochs, metric_loss, 'blue', label=metric_loss)
        plt.plot(epochs, metric_val_loss, 'red', label=metric_val_loss)
        plt.plot(epochs, metric_accuracy, 'blue', label=metric_accuracy)
        plt.plot(epochs, metric_val_accuracy, 'green', label=metric_val_accuracy)

        # Add title to the plot.
        plt.title(str('Model Metrics'))

        # Add legend to the plot.
        plt.legend(['loss', 'val_loss', 'accuracy', 'val_accuracy'])

        # If the plot already exist, remove
        plot_png = os.path.exists('metrics.png')
        if plot_png:
            os.remove('metrics.png')
            plt.savefig('metrics.png', bbox_inches='tight')
        else:
            plt.savefig('metrics.png', bbox_inches='tight')
        st.success('[INFO] Successfully Saved metrics.png')

        