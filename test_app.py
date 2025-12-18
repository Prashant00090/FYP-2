#!/usr/bin/env python3
"""
Test script for the Job Fraud Detection application
"""

import requests
import json

def test_prediction():
    """Test the prediction endpoint with sample data"""
    
    # Sample legitimate job posting
    legitimate_job = {
        'job_title': 'Software Engineer',
        'location': 'San Francisco, CA',
        'industry': 'Technology',
        'salary_range': '80000-120000',
        'company_profile': 'Established tech company with 500+ employees',
        'employment_type': 'Full-Time',
        'job_desc': 'We are looking for a skilled software engineer to join our development team. You will work on building scalable web applications using modern technologies.',
        'skills_desc': 'Bachelor\'s degree in Computer Science, 3+ years experience with Python, JavaScript, SQL. Experience with cloud platforms preferred.',
        'text': 'Competitive salary, health benefits, 401k matching, flexible work arrangements.'
    }
    
    # Sample suspicious job posting
    suspicious_job = {
        'job_title': 'Data Entry Specialist',
        'location': 'Remote',
        'industry': 'Various',
        'salary_range': '5000-10000',
        'company_profile': 'New company established 2023',
        'employment_type': 'Part-Time',
        'job_desc': 'Easy work from home opportunity! Earn $5000 per week immediately! No experience required! Just basic computer skills needed!',
        'skills_desc': 'Basic knowledge required. No degree needed. Flexible schedule.',
        'text': 'Earn money fast! Work from home! No experience needed! Contact us now!'
    }
    
    url = 'http://localhost:5000/predict'
    
    print("Testing Job Fraud Detection API...")
    print("=" * 50)
    
    # Test legitimate job
    print("\n1. Testing Legitimate Job Posting:")
    print(f"Job Title: {legitimate_job['job_title']}")
    try:
        response = requests.post(url, data=legitimate_job)
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction_text']}")
            print(f"Confidence - Legitimate: {result['confidence_legitimate']}%")
            print(f"Confidence - Fraudulent: {result['confidence_fraudulent']}%")
            print(f"Risk Level: {result['risk_level']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the Flask app is running.")
        return
    
    # Test suspicious job
    print("\n2. Testing Suspicious Job Posting:")
    print(f"Job Title: {suspicious_job['job_title']}")
    try:
        response = requests.post(url, data=suspicious_job)
        if response.status_code == 200:
            result = response.json()
            print(f"Prediction: {result['prediction_text']}")
            print(f"Confidence - Legitimate: {result['confidence_legitimate']}%")
            print(f"Confidence - Fraudulent: {result['confidence_fraudulent']}%")
            print(f"Risk Level: {result['risk_level']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the Flask app is running.")
        return
    
    print("\n" + "=" * 50)
    print("Test completed!")

def test_model_info():
    """Test the model info endpoint"""
    url = 'http://localhost:5000/model_info'
    
    print("\n3. Testing Model Info Endpoint:")
    try:
        response = requests.get(url)
        if response.status_code == 200:
            result = response.json()
            print(f"AUC Score: {result['auc_score']}")
            print(f"Confusion Matrix: {result['confusion_matrix']}")
        else:
            print(f"Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the server. Make sure the Flask app is running.")

if __name__ == '__main__':
    print("Job Fraud Detection API Test")
    print("Make sure the Flask application is running on http://localhost:5000")
    print("You can start it by running: python run.py")
    input("Press Enter to continue with the test...")
    
    test_prediction()
    test_model_info()