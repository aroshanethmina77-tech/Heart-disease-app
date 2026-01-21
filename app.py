from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from google import genai
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# 1. SETUP GEMINI
GEMINI_API_KEY = "AIzaSyB0Q1QSKBa1DwqbfTXEKiaZTWQYlNyF8i4"
client = genai.Client(api_key=GEMINI_API_KEY)

# 2. LOAD MODELS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FOLDER = os.path.join(BASE_DIR, 'save model')
FEATURE_NAMES = [
    'Age', 'Sex', 'Chest pain type', 'BP', 'Cholesterol', 
    'FBS over 120', 'EKG results', 'Max HR', 'Exercise angina', 
    'ST depression', 'Slope of ST', 'Number of vessels fluro', 'Thallium'
]

try:
    model = joblib.load(os.path.join(MODEL_FOLDER, 'heart_disease_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_FOLDER, 'scaler.pkl'))
    le = joblib.load(os.path.join(MODEL_FOLDER, 'label_encoder.pkl'))
    print("✅ Success: ML Models and Gemini Client Ready.")
except Exception as e:
    print(f"❌ Error Loading Files: {e}")

# 3. ENHANCED AI DOCTOR LOGIC
def get_ai_doctor_advice(data, prediction):
    """
    Generates a deep medical review and detailed advice.
    """
    is_risk = prediction.lower() == 'presence'
    
    # Enhanced Prompt for deeper medical insight
    prompt = f"""
    Role: Senior Consultant Cardiologist.
    ML Prediction: The patient has been screened for heart disease with a result of '{prediction}'.
    
    Patient Metrics:
    - Age: {data['Age']}
    - Blood Pressure: {data['BP']} mmHg
    - Cholesterol: {data['Cholesterol']} mg/dL
    - Max Heart Rate: {data['Max HR']} bpm
    
    Task: Provide a highly detailed and professional medical review.
    1. Determine a RISK SCORE (1-10). (Presence: 8-10, Absence: 1-3).
    2. Provide a 'REVIEW' which is a deep analysis. Explain how the combination of Age {data['Age']} and BP {data['BP']} affects their specific cardiovascular risk profile according to the {prediction} result.
    3. Provide 'TIPS' as a detailed action plan. Include clinical advice (e.g., further tests), dietary adjustments, and physical activity guidelines.
    
    YOU MUST RETURN THE RESPONSE IN THIS EXACT FORMAT:
    SCORE: [Number]/10
    REVIEW: [Deep Clinical Analysis]
    TIPS: [Tip 1, Tip 2, Tip 3, Tip 4, Tip 5]
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        score = "9/10" if is_risk else "2/10"
        return f"SCORE: {score}\nREVIEW: Deep analysis currently unavailable.\nTIPS: Consult a specialist immediately, Conduct an EKG/ECG, Monitor sodium intake, Regular cardio exercise, Avoid tobacco."

# 4. ROUTES
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_values = [float(data[f]) for f in FEATURE_NAMES]
        features_df = pd.DataFrame([input_values], columns=FEATURE_NAMES)
        
        scaled_data = scaler.transform(features_df)
        prediction_numeric = model.predict(scaled_data)
        result_label = le.inverse_transform(prediction_numeric)[0]
        
        # Get the detailed AI Advice
        ai_advice = get_ai_doctor_advice(data, result_label)
        
        return jsonify({
            'prediction': result_label,
            'ai_data': ai_advice,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)