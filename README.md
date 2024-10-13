## Wildfires Prediction Program (ACMHacks 2024)

## Inspiration (Problem)
Wildfires are a big problem:
- Harmful air pollutants such as “PM2.5, NO2, ozone, aromatic hydrocarbons, or lead” (WHO) which affect humans as well as wildlife
- Releases a lot of Carbon Dioxide and other Greenhouse gasses contributing to climate change
- Soil erosion that leads to damaged ecosystems
- Contaminates water supplies and impacts aquatic ecosystems
- Humans cause 85% of all wildfires yearly (National Park Service)
- Over 7.5 million acres of the wild were consumed by flames in 2022

## About our project (Solution)
We trained two AI models. Our 1st model takes in a 350x350 pixel satellite/topography image of a specific region and outputs whether the area is at risk of a wildfire. Our 2nd model takes in numerical weather parameters (temperature, humidity, and wind speed) and outputs whether the conditions signify a risk of a wildfire. To make interactions with the models more user-friendly, we integrated and established the models on a web application (localhost).

This project allows users to predict whether a certain region is at risk of being ravaged by a wildfire. With a predictive tool like this, preventive measures can be taken to prevent fires. An example of a preventive measure is controlled burning which is the practice of purposely setting fire to a bounded region to reduce the risk of rampant wildfires.

## How we built it
# Satellite/Topography Image Model:
- Used Kaggle’s “Wildfire Prediction Dataset (Satellite Images)” 
- Images of the topography of regions in Quebec, Canada labeled either “wildfire” (where wildfires occurred “in the future”) or “nowildfire” (wildfires haven’t occurred)
- https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset 
- Built a CNN model using Tensorflow and Keras API which classifies a given satellite/topography image into two classes which are “low risk of wildfires” (“nowildfires”) and “high risk of wildfires” (“wildfires”)
- Trained and evaluated the model which has about 94% accuracy

# Numerical Weather Parameters Model:
- Used Kaggle’s dataset: https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires which documents the location and time of various wildfires 
- Used Meteostat API to retrieve weather conditions (temperature, humidity, wind speeds) at the time and location of the wildfires (wildfire data)
- Also generated some control data (no wildfire data) →  random regions and their normal weather conditions
- Built an AI model using Tensorflow and Keras API which outputs whether a weather condition (temp, humidity, wind speed) is of  “low risk of wildfires” (“nowildfires”) and “high risk of wildfires” (“wildfires”)
- Trained and evaluated the model

Used the Flask framework to create the web application that hosts the models and allows users to easily interact with the models.

## Challenges we ran into
- It was challenging to find the data we needed to train the AI models. For instance, the data the Satellite/Topography Image Model was trained on was only images of regions in Quebec, Canada.
- The AI models were time-consuming to make since there were lots of parameters to tweak and neural network layers to order and rearrange to achieve the best results.
-  The weather APIs needed to access the data to train the Numerical Weather Parameters Model were difficult to work with and didn’t always have the data available
	- We ended up using the Meteostat API
- Cleaning up and organizing large amounts of data and generating labels, etc. for the AI models (especially for the Numerical Weather Parameters Model) was extremely time-consuming and tedious
- The Numerical Weather Parameters Model has about “100%” accuracy, though this is due to overfitting and insufficient data. The model actually isn’t very accurate as of now due to the challenges in finding proper data to train the model.

## What we learned
- Learned the complexity of making AI models
- Learned that data and data collection is a huge part of making AI models
- Learned how to use APIs
- Learned how to implement Flask
- Experience how it feels to collaborate as a team on a coding project   

## What's Next for Wildfire Prediction
- Train the models on better and varied data
- Have real-time satellite images and numerical weather data being passed into the models (though at a certain time interval so the models aren’t overloaded or running constantly)
- In addition to classifying whether the real-time satellite images/data have a risk or not of wildfires, we would like for the output to trigger an alert and notify the relevant departments/organizations (fire departments, etc.) so action can be taken such as controlled burning, etc.
