"""Tests for prediction models."""
import unittest
import pandas as pd
import numpy as np
from models.predictor import LogisticModel, RandomForestModel, XGBoostModel


class TestModels(unittest.TestCase):
    """Test model functionality."""

    def setUp(self):
        """Create sample data for testing."""
        np.random.seed(42)
        self.X_train = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100),
        })
        self.y_train = pd.Series(np.random.randint(0, 2, 100))

        self.X_test = pd.DataFrame({
            'feature1': np.random.randn(20),
            'feature2': np.random.randn(20),
            'feature3': np.random.randn(20),
        })

    def test_logistic_model(self):
        """Test logistic regression model."""
        model = LogisticModel()
        model.train(self.X_train, self.y_train)

        self.assertTrue(model.is_fitted)
        self.assertEqual(len(model.feature_columns), 3)

        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), 20)
        self.assertTrue(all(p in [0, 1] for p in predictions))

        probas = model.predict_proba(self.X_test)
        self.assertEqual(probas.shape, (20, 2))
        self.assertTrue(np.allclose(probas.sum(axis=1), 1.0))

    def test_random_forest_model(self):
        """Test random forest model."""
        model = RandomForestModel(n_estimators=10)
        model.train(self.X_train, self.y_train)

        self.assertTrue(model.is_fitted)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), 20)

        # Test feature importance
        importance = model.get_feature_importance()
        self.assertIsNotNone(importance)
        self.assertEqual(len(importance), 3)

    def test_xgboost_model(self):
        """Test XGBoost model."""
        model = XGBoostModel()
        model.train(self.X_train, self.y_train)

        self.assertTrue(model.is_fitted)
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), 20)


if __name__ == '__main__':
    unittest.main()
