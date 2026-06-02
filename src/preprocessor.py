"""
preprocessor.py
---------------
Reusable preprocessing pipeline used across all notebooks.
Handles: missing values, encoding, scaling, train-test split.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from typing import Tuple, List, Optional


class DataPreprocessor:
    """End-to-end preprocessing pipeline for tabular ML data."""

    def __init__(self, scale_method: str = 'standard'):
        """
        Args:
            scale_method: 'standard' (zero mean, unit var) or 'minmax' (0-1 range)
        """
        self.num_imputer = SimpleImputer(strategy='mean')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler() if scale_method == 'standard' else MinMaxScaler()
        self.label_encoders: dict = {}
        self._fitted = False

    # ------------------------------------------------------------------ #

    def fit_transform(
        self,
        df: pd.DataFrame,
        target_col: str,
        cat_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Full pipeline: impute → encode → scale → split.

        Returns: X_train, X_test, y_train, y_test
        """
        df = df.copy()
        y = df.pop(target_col).values

        # Auto-detect categorical columns if not provided
        if cat_cols is None:
            cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        num_cols = [c for c in df.columns if c not in cat_cols]

        # Impute
        if num_cols:
            df[num_cols] = self.num_imputer.fit_transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = self.cat_imputer.fit_transform(df[cat_cols])

        # Encode categoricals
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Scale
        X = self.scaler.fit_transform(df.values)
        self._fitted = True

        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def transform(self, df: pd.DataFrame, cat_cols: Optional[List[str]] = None) -> np.ndarray:
        """Transform new data using already-fitted pipeline."""
        if not self._fitted:
            raise RuntimeError('Call fit_transform() before transform().')
        df = df.copy()
        if cat_cols is None:
            cat_cols = list(self.label_encoders.keys())
        num_cols = [c for c in df.columns if c not in cat_cols]
        if num_cols:
            df[num_cols] = self.num_imputer.transform(df[num_cols])
        if cat_cols:
            df[cat_cols] = self.cat_imputer.transform(df[cat_cols])
        for col in cat_cols:
            if col in self.label_encoders:
                df[col] = self.label_encoders[col].transform(df[col].astype(str))
        return self.scaler.transform(df.values)
