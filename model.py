import cv2 as cv
import keras
import numpy as np
import os
import sys
import tensorflow as tf

train_dir = 'C:/Users/drjai/Wildfires_ACM/data/train'
valid_dir = 'C:/Users/drjai/Wildfires_ACM/data/valid'
test_dir = 'C:/Users/drjai/Wildfires_ACM/data/test'

x_train = []
y_train = []
x_valid = []
y_valid = []
x_test = []
y_test = []

for classes in os.listdir(train_dir):
    print("Loading dataset training")
    for file in os.listdir(os.path.join(train_dir,classes)):
        image_path = os.path.join(train_dir,classes,file)
        image = cv.imread(image_path)
        image = cv.resize(image, (64,64))
        image = np.array(image)
        image = image/255
        x_train.append(image)
        y_train.append(classes)

for classes in os.listdir(valid_dir):
    print("Loading dataset validating")
    for file in os.listdir(os.path.join(valid_dir,classes)):
        image_path = os.path.join(valid_dir,classes,file)
        image = cv.imread(image_path)
        image = cv.resize(image, (64,64))
        image = np.array(image)
        image = image/255
        x_valid.append(image)
        y_valid.append(classes)

for classes in os.listdir(test_dir):
    print("Loading dataset testing")
    for file in os.listdir(os.path.join(test_dir,classes)):
        image_path = os.path.join(test_dir,classes,file)
        image = cv.imread(image_path)
        image = cv.resize(image, (64,64))
        image = np.array(image)
        image = image/255
        x_test.append(image)
        y_test.append(classes)


x_train = np.array(x_train)
x_valid = np.array(x_valid)
x_test = np.array(x_test)

y_train_binary = []
y_valid_binary = []
y_test_binary = []

for label in y_train:
    if label == 'wildfire':
        y_train_binary.append(1)
    else:
        y_train_binary.append(0)

for label in y_valid:
    if label == 'wildfire':
        y_valid_binary.append(1)
    else:
        y_valid_binary.append(0)

for label in y_test:
    if label == 'wildfire':
        y_test_binary.append(1)
    else:
        y_test_binary.append(0)


y_train = np.array(y_train_binary)
y_valid = np.array(y_valid_binary)
y_test = np.array(y_test_binary)


model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),

        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(x_train, y_train, epochs=20)

test_loss, test_accuracy = model.evaluate(x_test, y_test)

print("Test Accuracy:", test_accuracy)

model.save('wildfire_model.h5')