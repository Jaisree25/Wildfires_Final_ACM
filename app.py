from flask import Flask, render_template, request
import keras
import numpy as np
import cv2 as cv

app = Flask(__name__)

model = keras.models.load_model('wildfire_model.h5')
model_weather = keras.models.load_model('weather_wildfire_model.h5')

def process_resize_image(image_path):
    #image = cv.imread(image_path)
    file_bytes = np.frombuffer(image_path.read(), np.uint8)
    image = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    image = cv.resize(image, (64, 64))
    image = np.array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)

    return image

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    processed_image = process_resize_image(file)

    prediction = model.predict(processed_image)
    predicted_class = (prediction > 0.5).astype(int)
    predicted_class = predicted_class[0][0]

    if predicted_class == 1:
        result = "HIGH"
    if predicted_class == 0:
        result = "LOW"

    return render_template('result.html', predicted_class=result)

@app.route('/predict_weather', methods=['POST'])
def predict_weather():
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    wind_speed = float(request.form['wind_speed'])

    input_data = np.array([[temperature, humidity, wind_speed]], dtype=np.float32)

    #scaler = StandardScaler()
    #input_data_scaled = scaler.fit_transform(input_data)
    #prediction = model.predict(input_data_scaled)

    prediction = model_weather.predict(input_data)

    predicted_class = (prediction > 0.5).astype(int)
    predicted_class = predicted_class[0][0]

    if predicted_class == 1:
        result = "HIGH"
    if predicted_class == 0:
        result = "LOW"

    return render_template('result_weather.html', predicted_class=result)

if __name__ == '__main__':
    app.run(debug=True)