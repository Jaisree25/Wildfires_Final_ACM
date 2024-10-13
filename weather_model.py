import tensorflow as tf
from sklearn.model_selection import train_test_split
from meteostat_data import overall_data, overall_labels
import numpy as np
from sklearn.preprocessing import MinMaxScaler

x = np.array(overall_data)
y = np.array(overall_labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,epochs=100)

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print("Test accuracy:", test_accuracy)

model.save('weather_wildfire_model.h5')