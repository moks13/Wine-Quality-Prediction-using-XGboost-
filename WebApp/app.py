import warnings
warnings.simplefilter("ignore")
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Load the saved model and scaler using pickle
with open('xgboost_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file, encoding='latin1') 

with open('scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)

# Function to take user input and make predictions
def predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,total_sulfur_dioxide, density, pH, sulphates, alcohol):

    # Creating a new data point based on user inputs
    user_data = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, 
                           total_sulfur_dioxide, density, pH, sulphates, alcohol]])

    # Apply the saved scaler to the new input data
    user_data_scaled = loaded_scaler.transform(user_data)

    # Make a prediction using the loaded model
    prediction = loaded_model.predict(user_data_scaled)

    print(prediction)

    return prediction
#===============================================================================================================

from flask import Flask, render_template, request,send_file,jsonify
from flask_cors import CORS,cross_origin

app = Flask(__name__)

@app.route('/',methods=['GET'])
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
@cross_origin()
def index():
    if request.method == 'POST':
        try:
            fixed_acidity=float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            # predictions using the loaded model file and scaler file
            prediction = predict(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, total_sulfur_dioxide, density, pH, sulphates, alcohol)

            print('prediction is', prediction)

            if prediction == 1:
                return render_template('good_quality.html')
            elif prediction == 0:
                return render_template('bad_quality.html')
            else:
                return "something is wrong"

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')

if __name__ == "__main__":
	app.run(debug=True)