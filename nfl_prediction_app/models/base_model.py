"""Base model interface for NFL prediction."""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import joblib
from pathlib import Path


class BaseNFLModel(ABC):
    """Abstract base class for NFL prediction models."""

    def __init__(self, name: str):
        """
        Initialize the model.

        Args:
            name: Name of the model
        """
        self.name = name
        self.model = None
        self.feature_columns = None
        self.is_fitted = False

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model.

        Args:
            X: Feature DataFrame
            y: Target Series
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict probabilities (for classification models).

        Args:
            X: Feature DataFrame

        Returns:
            Array of probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X[self.feature_columns])
        else:
            raise NotImplementedError("Model does not support probability predictions")

    def save(self, path: Path) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'name': self.name,
        }
        joblib.dump(model_data, path)

    def load(self, path: Path) -> None:
        """
        Load the model from disk.

        Args:
            path: Path to the saved model
        """
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.feature_columns = model_data['feature_columns']
        self.name = model_data['name']
        self.is_fitted = True

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance if available.

        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            return importance
        return None
