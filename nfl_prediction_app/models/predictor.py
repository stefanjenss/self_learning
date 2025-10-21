"""Prediction models for NFL games."""
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from .base_model import BaseNFLModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogisticModel(BaseNFLModel):
    """Logistic Regression baseline model."""

    def __init__(self, name: str = "Logistic"):
        super().__init__(name)
        self.scaler = StandardScaler()

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train logistic regression model."""
        self.feature_columns = X.columns.tolist()

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model = LogisticRegression(
            random_state=42,
            max_iter=1000,
            solver='lbfgs'
        )
        self.model.fit(X_scaled, y)
        self.is_fitted = True

        logger.info(f"{self.name} model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict(X_scaled)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities."""
        X_scaled = self.scaler.transform(X[self.feature_columns])
        return self.model.predict_proba(X_scaled)


class RandomForestModel(BaseNFLModel):
    """Random Forest model."""

    def __init__(self, name: str = "RandomForest", n_estimators: int = 100):
        super().__init__(name)
        self.n_estimators = n_estimators

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train random forest model."""
        self.feature_columns = X.columns.tolist()

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=42,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            n_jobs=-1
        )
        self.model.fit(X[self.feature_columns], y)
        self.is_fitted = True

        logger.info(f"{self.name} model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X[self.feature_columns])


class XGBoostModel(BaseNFLModel):
    """XGBoost gradient boosting model."""

    def __init__(self, name: str = "XGBoost"):
        super().__init__(name)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train XGBoost model."""
        self.feature_columns = X.columns.tolist()

        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        self.model.fit(X[self.feature_columns], y)
        self.is_fitted = True

        logger.info(f"{self.name} model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X[self.feature_columns])


class LightGBMModel(BaseNFLModel):
    """LightGBM gradient boosting model."""

    def __init__(self, name: str = "LightGBM"):
        super().__init__(name)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train LightGBM model."""
        self.feature_columns = X.columns.tolist()

        self.model = LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        self.model.fit(X[self.feature_columns], y)
        self.is_fitted = True

        logger.info(f"{self.name} model trained on {len(X)} samples")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X[self.feature_columns])


class EnsembleModel(BaseNFLModel):
    """Ensemble of multiple models."""

    def __init__(self, models: list, name: str = "Ensemble"):
        super().__init__(name)
        self.models = models
        self.weights = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Train all models in the ensemble."""
        self.feature_columns = X.columns.tolist()

        for model in self.models:
            model.train(X, y)

        # Equal weights by default
        self.weights = np.ones(len(self.models)) / len(self.models)
        self.is_fitted = True

        logger.info(f"{self.name} trained with {len(self.models)} models")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities by averaging model outputs."""
        probas = []
        for model in self.models:
            probas.append(model.predict_proba(X))

        # Weighted average
        ensemble_proba = np.average(probas, axis=0, weights=self.weights)
        return ensemble_proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        probas = self.predict_proba(X)
        return (probas[:, 1] > 0.5).astype(int)
