import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("classifier_model.h5")

with open("labels.txt") as f:
    labels = f.read().splitlines()

img_path = input("Enter image path: ")

img = image.load_img(img_path, target_size=(224, 224))
img = image.img_to_array(img) / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
idx = np.argmax(pred)

print("Class:", labels[idx])
print("Confidence:", round(pred[0][idx]*100, 2), "%")
