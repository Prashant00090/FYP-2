import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and results
try:
    model = joblib.load('fraud_detection_model.joblib')
    with open('model_results.pkl', 'rb') as f:
        model_results = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    model_results = None

# Define the expected columns based on the notebook
STRUCTURED_COLS = [
    "job_title", "location", "industry", "salary_range",
    "company_profile", "employment_type"
]
TEXT_COLS = ["job_desc", "skills_desc", "text"]

def build_all_text(job_desc, skills_desc, text):
    """Combine text fields as done in the notebook"""
    parts = []
    for val in [job_desc, skills_desc, text]:
        if pd.isna(val) or val is None:
            val = ""
        parts.append(str(val))
    return " ".join(parts)

def preprocess_input(data):
    """Preprocess input data to match model expectations"""
    # Create DataFrame with the input data
    df = pd.DataFrame([data])
    
    # Build all_text column
    df["all_text"] = build_all_text(
        data.get("job_desc", ""),
        data.get("skills_desc", ""),
        data.get("text", "")
    )
    
    # Select only the columns needed for prediction
    feature_cols = STRUCTURED_COLS + ["all_text"]
    df = df[feature_cols]
    
    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Get form data
        data = {
            'job_title': request.form.get('job_title', ''),
            'location': request.form.get('location', ''),
            'industry': request.form.get('industry', ''),
            'salary_range': request.form.get('salary_range', ''),
            'company_profile': request.form.get('company_profile', ''),
            'employment_type': request.form.get('employment_type', ''),
            'job_desc': request.form.get('job_desc', ''),
            'skills_desc': request.form.get('skills_desc', ''),
            'text': request.form.get('text', '')
        }
        
        # Preprocess the input
        df = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(df)[0]
        prediction_proba = model.predict_proba(df)[0]
        
        # Get confidence scores
        confidence_legitimate = prediction_proba[0] * 100
        confidence_fraudulent = prediction_proba[1] * 100
        
        result = {
            'prediction': int(prediction),
            'prediction_text': 'Fraudulent' if prediction == 1 else 'Legitimate',
            'confidence_legitimate': round(confidence_legitimate, 2),
            'confidence_fraudulent': round(confidence_fraudulent, 2),
            'risk_level': 'High' if confidence_fraudulent > 70 else 'Medium' if confidence_fraudulent > 30 else 'Low'
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_info')
def model_info():
    """Return model performance information"""
    if model_results is None:
        return jsonify({'error': 'Model results not available'}), 500
    
    try:
        info = {
            'auc_score': round(model_results['auc_score'], 4),
            'confusion_matrix': model_results['confusion_matrix'].tolist()
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)