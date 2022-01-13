# Disaster Response Pipeline Project

## Project Motivation
In this project, the main aim is to classify the messages into different categories. Instead of training the model in a conventional method, here we use pipeline to structure the models.
Given real world disaster messages as dataset, first ETL pipeline is run through the data, thereby cleaning the dataset in order to facilaitate seamless classification. Then a ML pipeline is built to fit and train the dataset. Following a gridsearch paradigm to tune the hyperparameters, accuracy is reported for the model predicted classes.

## File Structure

app

| - template

| |- master.html # main page of web app

| |- go.html # classification result page of web app

|- run.py # Flask file that runs app

data

|- disaster_categories.csv # data to process

|- disaster_messages.csv # data to process

|- process_data.py # data cleaning pipeline

|- InsertDatabaseName.db # database to save clean data to

models

|- train_classifier.py # machine learning pipeline

|- classifier.pkl # saved model

README.md

## Installations

1. sys
2. numpy
3. nltk
4. pandas
5. sklearn
6. sqlalchemy
7. re

# Instructions to implement the project

1. In order to run the ETL pipeline, run the command inside data directory: python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
2. In order to run the ML pipeline, run the command inside models directory: python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
3. Run the command: python run.py inside app's directory to run the web app
4. Go to the webpage [http://0.0.0.0:3001/](http://0.0.0.0:3001/)

## Visualisation Using Web App

The flask web app is deisgned in a way to output the categories of the given disaster concerning message. The app also has the means to display data visualisations.
Following images depicts the UI and the output from the app.
<img width="798" alt="app1" src="https://user-images.githubusercontent.com/81457427/149317689-8451cd6e-b7f5-4b18-a295-7f614ded9802.png">

<img width="769" alt="app 2" src="https://user-images.githubusercontent.com/81457427/149317685-f88de464-de35-483a-8e10-c25fb5f850f3.png">

<img width="1280" alt="app 3" src="https://user-images.githubusercontent.com/81457427/149317679-4dc51a4c-f633-4fd8-a7d2-66196e792c50.png">


## Acknowledgement

I thank Udacity for the web app code and Figure Eight for their disaster response dataset.

