# Student Performance Predictor by Rinku Prajapati
# AICTE Microsoft Azure Internship Project - Feb 2026
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

print("📁 Loading model...")
with open('student_model.pkl', 'rb') as f:
    model, le_parent = pickle.load(f)
print("✅ Model loaded!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = float(request.form['age'])
        study_hours = float(request.form['study_hours'])
        absences = float(request.form['absences'])
        parent_edu = request.form['parent_education']
        
        parent_encoded = le_parent.transform([parent_edu])[0]
        features = np.array([[age, study_hours, absences, parent_encoded]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        result = 'PASS' if prediction == 1 else 'FAIL'
        confidence = f"{max(probability)*100:.1f}%"
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'pass_prob': f"{probability[1]*100:.1f}%",
            'fail_prob': f"{probability[0]*100:.1f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    print("🌐 Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5001)
