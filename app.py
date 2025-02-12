from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load both models
with open('random_forest_model.pkl', 'rb') as file:
    basketball_model = pickle.load(file)

with open('logistic_regression_model.pkl', 'rb') as file:
    football_model = joblib.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/basketball')
def basketball():
    return render_template('basketball.html', prediction=None)

@app.route('/football')
def football():
    return render_template('football.html', prediction=None)

@app.route('/predict_basketball', methods=['POST'])
def predict_basketball():
    if request.method == 'POST':
        # Get values from the form
        features = {
            'Age': float(request.form['age']),
            'Games_Played': float(request.form['games_played']),
            'Wins': float(request.form['wins']),
            'Losses': float(request.form['losses']),
            'Minutes_Played': float(request.form['minutes_played']),
            'Field_Goal_Percentage': float(request.form['field_goal_percentage']),
            'Free_Throws_Made': float(request.form['free_throws_made']),
            'Free_Throw_Percentage': float(request.form['free_throw_percentage']),
            'Total_Rebounds': float(request.form['total_rebounds']),
            'Assists': float(request.form['assists']),
            'Turnovers': float(request.form['turnovers']),
            'Steals': float(request.form['steals']),
            'Blocks': float(request.form['blocks']),
            'Personal_Fouls': float(request.form['personal_fouls']),
            'Plus_Minus': float(request.form['plus_minus'])
        }
        
        input_df = pd.DataFrame([features])
        prediction = basketball_model.predict(input_df)[0]
        prediction = round(prediction, 2)

        return render_template('basketball.html', prediction=prediction)

@app.route('/predict_football', methods=['POST'])
def predict_football():
    if request.method == 'POST':
        # Get values from the form
        features = {
            'HTGD': float(request.form['htgd']),
            'ATGD': float(request.form['atgd']),
            'HTP': float(request.form['htp']),
            'ATP': float(request.form['atp']),
            'DiffFormPts': float(request.form['diff_form_pts']),
            'DiffPts': float(request.form['diff_pts'])
        }
        
        # Create form history features
        form_columns = ['H1', 'H2', 'H3', 'H4', 'H5', 'A1', 'A2', 'A3', 'A4', 'A5']
        for col in form_columns:
            features[f"{col}_D"] = 0
            features[f"{col}_L"] = 0
            features[f"{col}_M"] = 0
            features[f"{col}_W"] = 0
            
            form_value = request.form[col.lower()]
            if form_value in ['D', 'L', 'M', 'W']:
                features[f"{col}_{form_value}"] = 1

        input_df = pd.DataFrame([features])
        prediction = football_model.predict(input_df)[0]
        prediction_text = "Home Win" if prediction == 1 else "Away Win or Draw"

        return render_template('football.html', prediction=prediction_text)

if __name__ == '__main__':
    app.run(debug=True) 