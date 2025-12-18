#!/usr/bin/env python3
"""
Job Fraud Detection Web Application
Run this script to start the Flask web server
"""

import os
import sys
from app import app

if __name__ == '__main__':
    # Check if model files exist
    if not os.path.exists('fraud_detection_model.joblib'):
        print("Error: fraud_detection_model.joblib not found!")
        print("Please make sure the model file is in the same directory as this script.")
        sys.exit(1)
    
    if not os.path.exists('model_results.pkl'):
        print("Warning: model_results.pkl not found!")
        print("Model performance information will not be available.")
    
    print("Starting Job Fraud Detection Web Application...")
    print("Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)