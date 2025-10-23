"""Tests for betting strategies."""
import unittest
import pandas as pd
import numpy as np
from backtesting.strategies import (
    FlatBettingStrategy,
    KellyCriterionStrategy,
    ConfidenceThresholdStrategy
)


class TestStrategies(unittest.TestCase):
    """Test betting strategy functionality."""

    def setUp(self):
        """Create sample game data."""
        self.game = pd.Series({
            'game_id': '2024_01_KC_BUF',
            'home_team': 'KC',
            'away_team': 'BUF',
            'home_spread': -3.5,
            'home_spread_odds': -110,
        })

    def test_flat_betting(self):
        """Test flat betting strategy."""
        strategy = FlatBettingStrategy(bet_amount=100, bankroll=10000)

        # Should bet when confident
        bet_size = strategy.calculate_bet_size(self.game, 1, 0.65)
        self.assertEqual(bet_size, 100)

        # Should not bet when not confident
        bet_size = strategy.calculate_bet_size(self.game, 0, 0.45)
        self.assertEqual(bet_size, 0)

    def test_kelly_criterion(self):
        """Test Kelly Criterion strategy."""
        strategy = KellyCriterionStrategy(
            fraction=0.25,
            min_edge=0.02,
            bankroll=10000
        )

        # Should bet when edge exists
        bet_size = strategy.calculate_bet_size(self.game, 1, 0.60)
        self.assertGreater(bet_size, 0)
        self.assertLess(bet_size, 500)  # Should be reasonable

        # Should not bet with no edge
        bet_size = strategy.calculate_bet_size(self.game, 1, 0.50)
        self.assertEqual(bet_size, 0)

    def test_confidence_threshold(self):
        """Test confidence threshold strategy."""
        strategy = ConfidenceThresholdStrategy(
            confidence_threshold=0.60,
            bet_percentage=0.02,
            bankroll=10000
        )

        # Should bet when confidence exceeds threshold
        bet_size = strategy.calculate_bet_size(self.game, 1, 0.65)
        self.assertEqual(bet_size, 200)  # 2% of 10000

        # Should not bet below threshold
        bet_size = strategy.calculate_bet_size(self.game, 1, 0.55)
        self.assertEqual(bet_size, 0)


if __name__ == '__main__':
    unittest.main()
