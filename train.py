import ArcFace
import argparse
import cv2
import glob
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import pandas as pd
import tensorflow as tf
import pickle


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, default='Norm',
                help="path to Norm/dir")
ap.add_argument("-o", "--save", type=str, default='models/model.h5',
                help="path to save .h5 model, eg: dir/model.h5")
ap.add_argument("-l", "--le", type=str, default='models/le.pickle',
	            help="path to label encoder")
ap.add_argument("-b", "--batch_size", type=int, default=16,
	            help="batch Size for model training")
ap.add_argument("-e", "--epochs", type=int, default=200,
	            help="Epochs for Model Training")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
checkpoint_path = args['save']

# Load ArcFace Model
model = ArcFace.loadModel()
model.load_weights("arcface_weights.h5")
print("ArcFace expects ", model.layers[0].input_shape[0][1:], " inputs")
print("and it represents faces as ",
      model.layers[-1].output_shape[1:], " dimensional vectors")
target_size = model.layers[0].input_shape[0][1:3]
print('target_size: ', target_size)

# Variable for store img Embedding
x = []
y = []

names = os.listdir(path_to_dir)
names = sorted(names)
class_number = len(names)

for name in names:
    img_list = glob.glob(os.path.join(path_to_dir, name) + '/*')
    img_list = sorted(img_list)

    for img_path in img_list:
        img = cv2.imread(img_path)
        img_resize = cv2.resize(img, target_size)
        # what this line doing? must?
        img_pixels = tf.keras.preprocessing.image.img_to_array(img_resize)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_norm = img_pixels/255  # normalize input in [0, 1]
        img_embedding = model.predict(img_norm)[0]

        x.append(img_embedding)
        y.append(name)
        print(f'[INFO] Embedding {img_path}')
    print(f'[INFO] Completed {name} Part')
print('[INFO] Image Data Embedding Completed...')

# DataFrame
df = pd.DataFrame(x, columns=np.arange(512))
x = df.copy()
x = x.astype('float64')

le = LabelEncoder()
labels = le.fit_transform(y)
labels = tf.keras.utils.to_categorical(labels, 2)

# Train Deep Neural Network
x_train, x_test, y_train, y_test = train_test_split(x, labels,
                                                    test_size=0.2,
                                                    random_state=0)

model = Sequential([
    layers.Dense(1024, activation='relu', input_shape=[512]),
    layers.Dense(512, activation='relu'),
    layers.Dense(class_number, activation="softmax")
])

# Model Summary
print('Model Summary: ', model.summary())

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Add a checkpoint callback to store the checkpoint that has the highest
# validation accuracy.
checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path,
                                             monitor='val_accuracy',
                                             verbose=1,
                                             save_best_only=True,
                                             mode='max')
earlystopping = keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                              patience=20)

print('[INFO] Model Training Started ...')
# Start training
history = model.fit(x_train, y_train,
                    epochs=args['epochs'],
                    batch_size=args['batch_size'],
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint, earlystopping])

print('[INFO] Model Training Completed')
print(f'[INFO] Model Successfully Saved in /{checkpoint_path}')

# save label encoder
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()
print('[INFO] Successfully Saved models/le.pickle')

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
print('[INFO] Successfully Saved metrics.png')
