from flask import Flask, session, render_template, request, make_response, redirect, flash, jsonify
from flask.helpers import url_for
# from flask_mysql import MySQL
# from flaskext.mysql import MySQL
# from datetime import date
# from flask_mail import *
# from random import *
# import pdfkit
import os
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

app = Flask(__name__)
@app.route('/')
def home():
    return render_template("home.html")

@app.route('/upload', methods=['POST'])
def upload():
    model_file = request.form['modelfile']
    
    # Work on the model using the model_file
    
    # Redirect to www.google.com
    return redirect("http://www.google.com")

@app.route('/application', methods=['POST'])
def application():
    return render_template('dashboard.html')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/login', methods=['POST'])
def login():
    file = request.form['file']
    dataset = pd.read_csv(file)

    X_train = pd.read_csv("/Users/sanskar.verma/Downloads/Credit_risk_detection/train.csv")
    scaler = StandardScaler().fit(X_train)

    # Perform any necessary preprocessing on the dataset
    data = scaler.transform(dataset)

    # Make predictions using the loaded model
    predictions = model.predict(data)
    print('ok')
    return render_template('dashboard.html', predictions=predictions)

if __name__ == "__main__":
    app.run(debug=True)