import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, session, g
from functools import wraps
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'your_super_secret_key_here_please_change_me'

USER_CREDENTIALS = {
    'user': 'password123',
    'admin': 'securepassword'
}

# --- DECORATOR TO PROTECT ROUTES ---
def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if 'logged_in' not in session:
            # If not logged in, redirect to the home/login page
            return redirect(url_for('home'))
        return view(**kwargs)
    return wrapped_view

# --- NEW HOME/LOGIN PAGE ROUTE ---
@app.route('/', methods=['GET', 'POST'])
def home():
    # If the user is already logged in, redirect them to the prediction form
    if 'logged_in' in session:
        return redirect(url_for('predict_form'))

    # Handle the login form submission
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if USER_CREDENTIALS.get(username) == password:
            session['logged_in'] = True
            session['username'] = username
            return redirect(url_for('predict_form'))
        else:
            # On failed login, re-render the home page with an error message
            return render_template('home.html', error='Invalid username or password.')
    
    # Handle the initial GET request (displaying the home page)
    return render_template('home.html')

@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('home'))

# --- PROTECTED ROUTES ---
# The prediction form and prediction logic are now protected
@app.route('/predict_form')
@login_required
def predict_form():
    prediction_text = session.pop('prediction_text', None)
    probabilities = session.pop('probabilities', None)
    status = session.pop('status', None)
    form_data = session.pop('form_data', None)

    return render_template('predict_form.html',
                           prediction_text=prediction_text,
                           probabilities=probabilities,
                           status=status,
                           form_data=form_data
                           )

# --- Define the rest of your original protected routes as before ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'cirrhosis_prediction_model.pkl')
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

feature_columns = ['N_Days', 'Drug', 'Age', 'Sex', 'Ascites', 'Hepatomegaly',
                   'Spiders', 'Edema', 'Bilirubin', 'Cholesterol', 'Albumin',
                   'Copper', 'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets',
                   'Prothrombin', 'Stage']
class_mapping = {0: 'C', 1: 'CL', 2: 'D'}

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if model is None:
        session['prediction_text'] = "Error: Model not loaded. Please check server logs."
        session['status'] = "error"
        session['form_data'] = request.form.to_dict()
        return redirect(url_for('predict_form'))

    try:
        data = request.form.to_dict()
        input_data = {}
        for col in feature_columns:
            if col in ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper',
                       'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 'Stage']:
                try:
                    input_data[col] = float(data.get(col, 0.0))
                except ValueError:
                    input_data[col] = np.nan
            else:
                input_data[col] = data.get(col, '')

        input_df = pd.DataFrame([input_data], columns=feature_columns)
        probabilities = model.predict_proba(input_df)[0]
        prediction_results = {}
        for idx, prob in enumerate(probabilities):
            label = class_mapping.get(idx, f"Unknown_{idx}")
            prediction_results[f'Status_{label}'] = f"{prob*100:.2f}%"

        predicted_class_index = np.argmax(probabilities)
        predicted_status = class_mapping.get(predicted_class_index, "Unknown")
        
        session['prediction_text'] = f"Predicted Cirrhosis Status: {predicted_status}"
        session['probabilities'] = prediction_results
        session['status'] = "success"
        session['form_data'] = input_data
        
        return redirect(url_for('predict_form'))

    except Exception as e:
        session['prediction_text'] = f"Prediction Error: {e}"
        session['status'] = "error"
        session['form_data'] = request.form.to_dict()
        return redirect(url_for('predict_form'))

if __name__ == '__main__':
    app.run(debug=True)