"""Model training and evaluation pipeline."""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from typing import Dict, List, Tuple
import logging
from pathlib import Path

from .predictor import LogisticModel, RandomForestModel, XGBoostModel, LightGBMModel, EnsembleModel
from config.config import SAVED_MODELS_DIR, RANDOM_STATE, TEST_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate NFL prediction models."""

    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        exclude_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for modeling.

        Args:
            df: Full dataset
            target_col: Name of target column
            exclude_cols: Columns to exclude from features

        Returns:
            Tuple of (features, target)
        """
        if exclude_cols is None:
            exclude_cols = [
                'game_id', 'home_team', 'away_team', 'season', 'week',
                'home_score', 'away_score', 'home_win', 'point_diff',
                'beat_spread', 'over', 'total_points'
            ]

        # Get feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Handle missing values
        X = X.fillna(X.mean())

        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(X.mean())

        logger.info(f"Prepared {len(X)} samples with {len(feature_cols)} features")

        return X, y

    def train_model(
        self,
        model,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Train a single model and evaluate.

        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics
        """
        # Train
        model.train(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]

        # Evaluate
        results = {
            'model': model.name,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'train_precision': precision_score(y_train, y_pred_train),
            'test_precision': precision_score(y_test, y_pred_test),
            'train_recall': recall_score(y_train, y_pred_train),
            'test_recall': recall_score(y_test, y_pred_test),
            'train_f1': f1_score(y_train, y_pred_train),
            'test_f1': f1_score(y_test, y_pred_test),
            'train_auc': roc_auc_score(y_train, y_proba_train),
            'test_auc': roc_auc_score(y_test, y_proba_test),
            'train_logloss': log_loss(y_train, y_proba_train),
            'test_logloss': log_loss(y_test, y_proba_test),
        }

        logger.info(f"{model.name} - Test Accuracy: {results['test_accuracy']:.4f}, Test AUC: {results['test_auc']:.4f}")

        return results

    def train_all_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> pd.DataFrame:
        """
        Train all model types and compare.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            DataFrame with all results
        """
        # Initialize models
        models_to_train = [
            LogisticModel(),
            RandomForestModel(n_estimators=100),
            XGBoostModel(),
            LightGBMModel(),
        ]

        results_list = []

        for model in models_to_train:
            results = self.train_model(model, X_train, y_train, X_test, y_test)
            results_list.append(results)
            self.models[model.name] = model

        # Create ensemble
        ensemble = EnsembleModel(list(self.models.values()))
        ensemble.feature_columns = X_train.columns.tolist()
        ensemble.is_fitted = True
        ensemble_results = self.train_model(ensemble, X_train, y_train, X_test, y_test)
        results_list.append(ensemble_results)
        self.models['Ensemble'] = ensemble

        # Convert to DataFrame
        results_df = pd.DataFrame(results_list)

        # Find best model
        best_idx = results_df['test_auc'].idxmax()
        self.best_model = self.models[results_df.loc[best_idx, 'model']]

        logger.info(f"\nBest model: {self.best_model.name}")

        return results_df

    def save_models(self, output_dir: Path = SAVED_MODELS_DIR):
        """
        Save all trained models.

        Args:
            output_dir: Directory to save models
        """
        for name, model in self.models.items():
            model_path = output_dir / f"{name.lower()}_model.joblib"
            model.save(model_path)
            logger.info(f"Saved {name} model to {model_path}")

    def load_model(self, model_name: str, model_dir: Path = SAVED_MODELS_DIR):
        """
        Load a saved model.

        Args:
            model_name: Name of the model
            model_dir: Directory containing saved models
        """
        model_path = model_dir / f"{model_name.lower()}_model.joblib"

        # Determine model type
        if 'logistic' in model_name.lower():
            model = LogisticModel()
        elif 'random' in model_name.lower():
            model = RandomForestModel()
        elif 'xgboost' in model_name.lower():
            model = XGBoostModel()
        elif 'lightgbm' in model_name.lower():
            model = LightGBMModel()
        elif 'ensemble' in model_name.lower():
            model = EnsembleModel([])
        else:
            raise ValueError(f"Unknown model type: {model_name}")

        model.load(model_path)
        self.models[model_name] = model
        logger.info(f"Loaded {model_name} model from {model_path}")

        return model


if __name__ == "__main__":
    # Example usage
    from config.config import PROCESSED_DATA_DIR

    # Load processed features
    features = pd.read_parquet(PROCESSED_DATA_DIR / "game_features.parquet")

    # Train models
    trainer = ModelTrainer()
    X, y = trainer.prepare_data(features, target_col='home_win')

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=False
    )

    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)
    print("\nModel Results:")
    print(results[['model', 'test_accuracy', 'test_auc', 'test_logloss']])

    # Save models
    trainer.save_models()
