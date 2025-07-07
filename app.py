from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load the trained model for /analyze endpoint
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'ml_model', 'customer_classifier.pkl')
model_classifier = joblib.load(MODEL_PATH)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    # Extract features in the correct order
    features = [
        data['age'],
        data['annual_salary'],
        data['num_accounts'],
        data['loan_interest'],
        data['num_loans'],
        data['days_overdue'],
        data['num_late_payments'],
        data['total_debt']
    ]
    X = np.array([features])
    prediction = model_classifier.predict(X)[0]
    # Garante que o score é inteiro
    try:
        prediction = int(prediction)
    except Exception:
        pass
    return jsonify({'credit_score': prediction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)