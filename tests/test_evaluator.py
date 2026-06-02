"""
Unit tests for src/evaluator.py
"""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from evaluator import evaluate_classifier, evaluate_regressor, compare_models
from sklearn.dummy import DummyClassifier, DummyRegressor


@pytest.fixture
def clf_data():
    rng = np.random.default_rng(42)
    X = rng.random((100, 4))
    y = (X[:, 0] > 0.5).astype(int)
    split = 80
    return X[:split], X[split:], y[:split], y[split:]


@pytest.fixture
def reg_data():
    rng = np.random.default_rng(42)
    X = rng.random((100, 3))
    y = X[:, 0] * 2 + rng.normal(0, 0.1, 100)
    split = 80
    return X[:split], X[split:], y[:split], y[split:]


def test_classifier_returns_required_keys(clf_data):
    X_tr, X_te, y_tr, y_te = clf_data
    clf = DummyClassifier(strategy='most_frequent')
    clf.fit(X_tr, y_tr)
    res = evaluate_classifier(clf, X_te, y_te, verbose=False)
    assert 'accuracy' in res
    assert 'f1_weighted' in res
    assert 'confusion_matrix' in res


def test_regressor_returns_required_keys(reg_data):
    X_tr, X_te, y_tr, y_te = reg_data
    reg = DummyRegressor()
    reg.fit(X_tr, y_tr)
    res = evaluate_regressor(reg, X_te, y_te, verbose=False)
    assert 'mae' in res
    assert 'rmse' in res
    assert 'r2' in res


def test_compare_models_classification(clf_data):
    X_tr, X_te, y_tr, y_te = clf_data
    models = {
        'Dummy Most Freq': DummyClassifier(strategy='most_frequent'),
        'Dummy Uniform':   DummyClassifier(strategy='uniform', random_state=0),
    }
    df = compare_models(models, X_tr, X_te, y_tr, y_te, task='classification', cv=3)
    assert len(df) == 2
    assert 'Accuracy' in df.columns


def test_compare_models_regression(reg_data):
    X_tr, X_te, y_tr, y_te = reg_data
    models = {'Dummy Mean': DummyRegressor(strategy='mean')}
    df = compare_models(models, X_tr, X_te, y_tr, y_te, task='regression')
    assert 'R²' in df.columns
