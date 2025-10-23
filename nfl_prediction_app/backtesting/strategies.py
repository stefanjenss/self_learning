"""Betting strategies for NFL games."""
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import Dict, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BettingStrategy(ABC):
    """Abstract base class for betting strategies."""

    def __init__(self, name: str, bankroll: float = 10000):
        """
        Initialize betting strategy.

        Args:
            name: Strategy name
            bankroll: Starting bankroll
        """
        self.name = name
        self.initial_bankroll = bankroll
        self.current_bankroll = bankroll

    @abstractmethod
    def calculate_bet_size(
        self,
        game: pd.Series,
        prediction: float,
        confidence: float
    ) -> float:
        """
        Calculate bet size for a game.

        Args:
            game: Game information
            prediction: Model prediction
            confidence: Prediction confidence

        Returns:
            Bet size in dollars
        """
        pass

    def reset(self):
        """Reset bankroll to initial value."""
        self.current_bankroll = self.initial_bankroll


class FlatBettingStrategy(BettingStrategy):
    """Flat betting - same amount every bet."""

    def __init__(self, bet_amount: float = 100, bankroll: float = 10000):
        super().__init__("Flat Betting", bankroll)
        self.bet_amount = bet_amount

    def calculate_bet_size(
        self,
        game: pd.Series,
        prediction: float,
        confidence: float
    ) -> float:
        """Always bet the same amount."""
        if confidence > 0.5:  # Only bet if confident
            return min(self.bet_amount, self.current_bankroll)
        return 0


class PercentageBettingStrategy(BettingStrategy):
    """Bet a fixed percentage of bankroll."""

    def __init__(self, percentage: float = 0.02, bankroll: float = 10000):
        super().__init__("Percentage Betting", bankroll)
        self.percentage = percentage

    def calculate_bet_size(
        self,
        game: pd.Series,
        prediction: float,
        confidence: float
    ) -> float:
        """Bet a percentage of current bankroll."""
        if confidence > 0.5:
            return self.current_bankroll * self.percentage
        return 0


class KellyCriterionStrategy(BettingStrategy):
    """Kelly Criterion - optimal bet sizing based on edge."""

    def __init__(
        self,
        fraction: float = 0.25,
        min_edge: float = 0.02,
        max_bet_pct: float = 0.05,
        bankroll: float = 10000
    ):
        """
        Initialize Kelly Criterion strategy.

        Args:
            fraction: Fractional Kelly (e.g., 0.25 for quarter Kelly)
            min_edge: Minimum edge required to place bet
            max_bet_pct: Maximum bet as percentage of bankroll
            bankroll: Starting bankroll
        """
        super().__init__("Kelly Criterion", bankroll)
        self.fraction = fraction
        self.min_edge = min_edge
        self.max_bet_pct = max_bet_pct

    def calculate_bet_size(
        self,
        game: pd.Series,
        prediction: float,
        confidence: float
    ) -> float:
        """
        Calculate Kelly Criterion bet size.

        Formula: f = (bp - q) / b
        where:
            f = fraction of bankroll to bet
            b = odds received (decimal - 1)
            p = probability of winning
            q = probability of losing (1 - p)
        """
        # American odds to decimal
        if 'home_spread_odds' in game and not pd.isna(game['home_spread_odds']):
            american_odds = game['home_spread_odds']
            if american_odds > 0:
                decimal_odds = (american_odds / 100) + 1
            else:
                decimal_odds = (100 / abs(american_odds)) + 1
        else:
            # Default -110 odds
            decimal_odds = (100 / 110) + 1

        b = decimal_odds - 1
        p = confidence
        q = 1 - p

        # Calculate edge
        edge = (p * decimal_odds) - 1

        # Only bet if we have sufficient edge
        if edge < self.min_edge:
            return 0

        # Kelly formula
        kelly_fraction = (b * p - q) / b

        # Apply fractional Kelly
        kelly_fraction *= self.fraction

        # Cap at max bet percentage
        kelly_fraction = min(kelly_fraction, self.max_bet_pct)

        # Ensure non-negative
        kelly_fraction = max(kelly_fraction, 0)

        bet_size = self.current_bankroll * kelly_fraction

        return bet_size


class ConfidenceThresholdStrategy(BettingStrategy):
    """Only bet when confidence exceeds a threshold."""

    def __init__(
        self,
        confidence_threshold: float = 0.6,
        bet_percentage: float = 0.02,
        bankroll: float = 10000
    ):
        super().__init__("Confidence Threshold", bankroll)
        self.confidence_threshold = confidence_threshold
        self.bet_percentage = bet_percentage

    def calculate_bet_size(
        self,
        game: pd.Series,
        prediction: float,
        confidence: float
    ) -> float:
        """Bet only when confidence exceeds threshold."""
        if confidence >= self.confidence_threshold:
            return self.current_bankroll * self.bet_percentage
        return 0


class ValueBettingStrategy(BettingStrategy):
    """Bet when model probability differs significantly from market odds."""

    def __init__(
        self,
        min_value: float = 0.05,
        base_bet_pct: float = 0.02,
        bankroll: float = 10000
    ):
        """
        Initialize value betting strategy.

        Args:
            min_value: Minimum value (model prob - implied prob) to bet
            base_bet_pct: Base bet as percentage of bankroll
            bankroll: Starting bankroll
        """
        super().__init__("Value Betting", bankroll)
        self.min_value = min_value
        self.base_bet_pct = base_bet_pct

    def calculate_bet_size(
        self,
        game: pd.Series,
        prediction: float,
        confidence: float
    ) -> float:
        """Bet when we find value vs market odds."""
        # Get implied probability from odds
        if 'home_spread_odds' in game and not pd.isna(game['home_spread_odds']):
            american_odds = game['home_spread_odds']
            if american_odds > 0:
                implied_prob = 100 / (american_odds + 100)
            else:
                implied_prob = abs(american_odds) / (abs(american_odds) + 100)
        else:
            implied_prob = 0.5238  # -110 odds

        # Calculate value
        value = confidence - implied_prob

        if value >= self.min_value:
            # Scale bet size by value
            bet_pct = self.base_bet_pct * (1 + value)
            return self.current_bankroll * bet_pct

        return 0


class MarketBiasStrategy(BettingStrategy):
    """Exploit known market biases (e.g., home favorites)."""

    def __init__(
        self,
        target_bias: str = "home_underdog",
        bet_percentage: float = 0.02,
        bankroll: float = 10000
    ):
        """
        Initialize market bias strategy.

        Args:
            target_bias: Type of bias to exploit
                - "home_underdog": Bet on home underdogs
                - "away_favorite": Bet on away favorites
                - "division_games": Bet on division game underdogs
            bet_percentage: Bet size as percentage of bankroll
            bankroll: Starting bankroll
        """
        super().__init__(f"Market Bias ({target_bias})", bankroll)
        self.target_bias = target_bias
        self.bet_percentage = bet_percentage

    def calculate_bet_size(
        self,
        game: pd.Series,
        prediction: float,
        confidence: float
    ) -> float:
        """Bet based on market bias."""
        should_bet = False

        if 'home_spread' in game and not pd.isna(game['home_spread']):
            home_spread = game['home_spread']

            if self.target_bias == "home_underdog":
                # Home team is underdog (positive spread)
                should_bet = home_spread > 0 and confidence > 0.5

            elif self.target_bias == "away_favorite":
                # Away team is favorite (negative home spread)
                should_bet = home_spread < 0 and confidence < 0.5

            elif self.target_bias == "division_games":
                is_division = game.get('is_division_game', False)
                should_bet = is_division and home_spread > 0 and confidence > 0.5

        if should_bet:
            return self.current_bankroll * self.bet_percentage

        return 0
