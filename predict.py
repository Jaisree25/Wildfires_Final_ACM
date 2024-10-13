import tensorflow as tf
import keras
import cv2 as cv
import numpy as np
import os

user_input = input("Please enter satellite image file: ")
print("You entered:", user_input)

model = keras.models.load_model('wildfire_model.h5')

def process_resize_image(image_path):
    image = cv.imread(image_path)
    image = cv.resize(image,(64, 64))
    image = np.array(image)
    image = image/ 255
    image = np.expand_dims(image, axis=0)

    return image

def predict_image(input):
    processed_image = process_resize_image(input)
    prediction = model.predict(processed_image)
    predicted_class = (prediction > 0.5).astype(int)
    return predicted_class[0][0]

predicted_class = predict_image(user_input)
if predicted_class == 1:
    print("HIGH of Wildfire according to the Topography of the area")
if predicted_class == 0:
    print("LOW Risk of Wildfire according to the Topography of the area")