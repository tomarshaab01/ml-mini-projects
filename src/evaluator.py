"""
evaluator.py
------------
Standardized model evaluation utilities:
  - Classification metrics (accuracy, F1, confusion matrix)
  - Regression metrics (MAE, MSE, RMSE, R²)
  - Cross-validation helper
  - Model comparison table
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, mean_absolute_error,
    mean_squared_error, r2_score
)
from sklearn.model_selection import cross_val_score
from typing import Dict, Any


def evaluate_classifier(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray = None,
    y_train: np.ndarray = None,
    cv: int = 5,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Full classification evaluation.
    Returns dict with accuracy, F1, confusion matrix, and optional CV score.
    """
    y_pred = model.predict(X_test)
    results = {
        'accuracy':         round(accuracy_score(y_test, y_pred), 4),
        'f1_weighted':      round(f1_score(y_test, y_pred, average='weighted'), 4),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }
    if X_train is not None and y_train is not None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv)
        results['cv_mean'] = round(cv_scores.mean(), 4)
        results['cv_std']  = round(cv_scores.std(), 4)

    if verbose:
        print(f"Accuracy : {results['accuracy']}")
        print(f"F1 Score : {results['f1_weighted']}")
        if 'cv_mean' in results:
            print(f"CV ({cv}-fold) : {results['cv_mean']} ± {results['cv_std']}")
        print('\nClassification Report:')
        print(classification_report(y_test, y_pred))

    return results


def evaluate_regressor(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    verbose: bool = True,
) -> Dict[str, float]:
    """Full regression evaluation — MAE, RMSE, R²."""
    y_pred = model.predict(X_test)
    results = {
        'mae':  round(mean_absolute_error(y_test, y_pred), 4),
        'rmse': round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        'r2':   round(r2_score(y_test, y_pred), 4),
    }
    if verbose:
        print(f"MAE  : {results['mae']}")
        print(f"RMSE : {results['rmse']}")
        print(f"R²   : {results['r2']}")
    return results


def compare_models(
    models: Dict[str, Any],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task: str = 'classification',
    cv: int = 5,
) -> pd.DataFrame:
    """
    Train + evaluate multiple models and return a sorted comparison DataFrame.
    task: 'classification' or 'regression'
    """
    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        if task == 'classification':
            res = evaluate_classifier(model, X_test, y_test, X_train, y_train, cv=cv, verbose=False)
            rows.append({'Model': name, 'Accuracy': res['accuracy'],
                         'F1': res['f1_weighted'], 'CV Mean': res.get('cv_mean', '-')})
        else:
            res = evaluate_regressor(model, X_test, y_test, verbose=False)
            rows.append({'Model': name, 'MAE': res['mae'], 'RMSE': res['rmse'], 'R²': res['r2']})

    df = pd.DataFrame(rows)
    sort_col = 'Accuracy' if task == 'classification' else 'R²'
    return df.sort_values(sort_col, ascending=False).reset_index(drop=True)
