from deepface.commons import functions
import ArcFace


model = ArcFace.loadModel()
model.load_weights("arcface_weights.h5")
print("ArcFace expects ",model.layers[0].input_shape[0][1:]," inputs")
print("and it represents faces as ", model.layers[-1].output_shape[1:]," dimensional vectors")
target_size = model.layers[0].input_shape[0][1:3]

detector_backend = 'retinaface'

img1 = functions.preprocess_face("image/img1.jpg", target_size=target_size, detector_backend=detector_backend, align = True)

img1_embedding = model.predict(img1)[0]

print(img1_embedding)

