import ArcFace
import argparse
import cv2
import glob
import numpy as np
import keras
from keras import layers, Sequential
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.preprocessing import Normalizer


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--dataset", type=str, required=True,
                help="path to dataset/dir")
ap.add_argument("-o", "--save", type=str, required=True,
                help="path to save .h5 model, eg: dir/model.h5")


args = vars(ap.parse_args())
path_to_dir = args["dataset"]
path_to_save = args['save']


model = ArcFace.loadModel()
model.load_weights("arcface_weights.h5")
print("ArcFace expects ",model.layers[0].input_shape[0][1:]," inputs")
print("and it represents faces as ", model.layers[-1].output_shape[1:]," dimensional vectors")
target_size = model.layers[0].input_shape[0][1:3]
print('target_size: ', target_size)


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
        img_resize = np.reshape(img_resize, (1, 112, 112, 3))
        img_embedding = model.predict(img_resize)[0]
        
        x.append(img_embedding)
        y.append(name)
        print(f'[INFO] Embedding {img_path}')
    print(f'[INFO] Completed {name} Part')
print('[INFO] Image Data Embedding Completed...')

# DataFrame
df = pd.DataFrame(x, columns=np.arange(512))

# Normalise
x = df.values #returns a numpy array
normalizer = Normalizer().fit(x)
x_norm = normalizer.transform(x)

df = pd.DataFrame(x_norm)

df['names'] = y
# print(df)

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

# Model Summary
print('Model Summary: ', model.summary())

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

print('[INFO] Model Training Started ...')
# Start training
history = model.fit(x_train, y_train,
                    epochs=200,
                    batch_size=16,
                    validation_data=(x_test, y_test),
                    callbacks=[checkpoint, earlystopping])

print('[INFO] Model Training Completed')
print(f'[INFO] Model Successfully Saved in /{path_to_save}')

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
