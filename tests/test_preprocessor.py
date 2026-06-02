"""
Unit tests for src/preprocessor.py
"""
import numpy as np
import pandas as pd
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessor import DataPreprocessor


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'age':    [25, np.nan, 35, 45, 28, 32],
        'salary': [50000, 60000, np.nan, 80000, 55000, 62000],
        'dept':   ['IT', 'HR', 'IT', np.nan, 'Finance', 'HR'],
        'hired':  [1, 0, 1, 1, 0, 1]
    })


def test_fit_transform_returns_four_arrays(sample_df):
    pp = DataPreprocessor()
    X_tr, X_te, y_tr, y_te = pp.fit_transform(sample_df, target_col='hired')
    assert X_tr is not None
    assert X_te is not None
    assert len(y_tr) + len(y_te) == len(sample_df)


def test_no_missing_after_imputation(sample_df):
    pp = DataPreprocessor()
    X_tr, X_te, y_tr, y_te = pp.fit_transform(sample_df.copy(), target_col='hired')
    assert not np.isnan(X_tr).any(), 'NaN found in training data after preprocessing'
    assert not np.isnan(X_te).any(), 'NaN found in test data after preprocessing'


def test_transform_without_fit_raises(sample_df):
    pp = DataPreprocessor()
    with pytest.raises(RuntimeError):
        pp.transform(sample_df.drop(columns=['hired']))


def test_minmax_scaler():
    pp = DataPreprocessor(scale_method='minmax')
    df = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0, 5.0], 'y': [0, 1, 0, 1, 0]})
    X_tr, X_te, _, _ = pp.fit_transform(df, target_col='y', test_size=0.2)
    assert X_tr.min() >= -0.01  # min-max values should be in [0, 1]
    assert X_tr.max() <= 1.01
