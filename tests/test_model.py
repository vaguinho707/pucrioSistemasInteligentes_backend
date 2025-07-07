import pytest
import numpy as np
import joblib
import os

@pytest.fixture(scope="module")
def model():
    model_path = os.path.join(os.path.dirname(__file__), '../ml_model/customer_classifier.pkl')
    return joblib.load(model_path)

@pytest.fixture
def input_data():
    return {
        'age': [25, 34, 46],
        'annual_salary': [19000.33, 34148.50, 30144.22],
        'num_accounts': [1, 3, 2],
        'loan_interest': [29.55, 15.33, 6.38],
        'num_loans': [15, 9, 3],
        'days_overdue': [60, 20, 4],
        'num_late_payments': [30, 9, 1],
        'total_debt': [20345.33, 1000.55, 835.00],
    }

def test_model_performance(model, input_data):
    # Arrange features in the same order as in the app
    features = [
        input_data['age'],
        input_data['annual_salary'],
        input_data['num_accounts'],
        input_data['loan_interest'],
        input_data['num_loans'],
        input_data['days_overdue'],
        input_data['num_late_payments'],
        input_data['total_debt']
    ]
    X = np.array(features).T 
    # Act
    predictions = model.predict(X)

    # Assert: cada previsão deve corresponder à classe esperada
    expected = ['Poor', 'Fair', 'Excelent']
    assert list(predictions) == expected, f"Esperado {expected}, mas obteve {list(predictions)}"
