import pytest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference

@pytest.fixture
def sample_data():
    """Provides a small dummy dataset for testing."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5],
        'salary': [0, 1, 0, 1, 0]
    })

def test_train_model_algorithm(sample_data):
    """Test that the model trained is a RandomForestClassifier."""
    X = sample_data[['feature1']]
    y = sample_data['salary']
    model = train_model(X, y)
    assert isinstance(model, RandomForestClassifier)

def test_compute_model_metrics():
    """Test that metric calculations return values between 0 and 1."""
    y = np.array([0, 1, 1, 0])
    preds = np.array([0, 1, 0, 1])
    precision, recall, fbeta = compute_model_metrics(y, preds)
    assert 0 <= precision <= 1
    assert 0 <= recall <= 1
    assert 0 <= fbeta <= 1

def test_inference_returns_predictions(sample_data):
    """Test that inference returns an array of the correct length."""
    X = sample_data[['feature1']]
    y = sample_data['salary']
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(sample_data)
    assert isinstance(preds, np.ndarray)