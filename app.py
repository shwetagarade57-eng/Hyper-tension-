
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import pickle
import os

# Initialize Flask application
app = Flask(__name__)

# Load the trained Logistic Regression model
# Updated filename to match logreg_model.pkl as per your project files
try:
    model_path = os.path.join(os.path.dirname(__file__), 'logreg_model.pkl')
    model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    print("Error: logreg_model.pkl not found. Ensure the trained model is saved in the root directory.")
except Exception as e:
    print(f"Error loading model: {e}")

def get_recommendation(prediction):
    """
    Maps model output to clinical recommendations and UI styles.
    0: Normal
    1: Stage 1 Hypertension
    2: Stage 2 Hypertension
    3: Hypertensive Crisis
    """
    recommendations = {
        0: {
            "stage": "Normal Blood Pressure",
            "color": "success",
            "advice": "Maintain a healthy lifestyle with balanced nutrition and regular exercise. Schedule annual check-ups."
        },
        1: {
            "stage": "Stage 1 Hypertension",
            "color": "warning",
            "advice": "Adopt a DASH diet (low sodium), increase physical activity to 150 min/week, and monitor BP at home."
        },
        2: {
            "stage": "Stage 2 Hypertension",
            "color": "orange",
            "advice": "Medical consultation recommended for potential pharmacological intervention and lifestyle modification."
        },
        3: {
            "stage": "Hypertensive Crisis (Severe)",
            "color": "danger",
            "advice": "EMERGENCY: Seek immediate medical attention at the nearest emergency department or call 911."
        }
    }
    return recommendations.get(prediction, recommendations[0])

@app.route('/')
def home():
    """Renders the landing page with the diagnostic form."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Processes form data and returns prediction results."""
    try:
        # Extract features from form input
        # Features: Age, Gender, BMI, Systolic, Diastolic, Smoking
        # Note: Gender and Smoking should be encoded as 0/1 in the frontend
        features = [[
            int(request.form['Age']),
            str(request.form['History']),
            str(request.form['Patient']),
            str(request.form['TakeMedication']),
            str(request.form['Severity']),
            str(request.form['BreathShortness']),
            str(request.form['VisualChanges']),
            str(request.form['NoseBleeding']),
            int(request.form['Whendiagnoused']),
            int(request.form['Systolic']),
            int(request.form['Diastolic']),
            str(request.form['ControlledDiet'])]]
        
        # Prepare input for model
        final_features = [np.array(features)]
        
        # Make prediction using the loaded Logistic Regression model
        prediction = model.predict(final_features)[0]
        
        # Get corresponding clinical recommendation
        result = get_recommendation(prediction)
        
        return render_template('index.html', 
                               prediction_text=result['stage'],
                               advice=result['advice'],
                               color=result['color'],
                               show_result=True)
    
    except Exception as e:
        # Graceful error handling for missing inputs or invalid data types
        return render_template('index.html', 
                               prediction_text="Processing Error",
                               advice="Please ensure all health parameters are filled with valid numeric values.",
                               color="secondary",
                               show_result=True)

if __name__ == "__main__":
    # Start the Flask development server
    app.run(debug=True)
