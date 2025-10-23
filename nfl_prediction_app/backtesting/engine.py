"""Backtesting engine for betting strategies."""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from .strategies import BettingStrategy
from .metrics import BettingMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """Engine to backtest betting strategies on historical data."""

    def __init__(self):
        self.results = []
        self.bet_history = []

    def run_backtest(
        self,
        strategy: BettingStrategy,
        games: pd.DataFrame,
        predictions: pd.Series,
        confidences: pd.Series,
        bet_type: str = "spread"
    ) -> Dict:
        """
        Run backtest for a betting strategy.

        Args:
            strategy: Betting strategy to test
            games: DataFrame with game information
            predictions: Model predictions (0 or 1 for classification)
            confidences: Prediction confidences (probabilities)
            bet_type: Type of bet ("spread", "moneyline", "total")

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {strategy.name}")

        # Reset strategy
        strategy.reset()

        bet_history = []
        bankroll_history = [strategy.initial_bankroll]

        for idx, (_, game) in enumerate(games.iterrows()):
            prediction = predictions.iloc[idx]
            confidence = confidences.iloc[idx]

            # Calculate bet size
            bet_size = strategy.calculate_bet_size(game, prediction, confidence)

            if bet_size > 0:
                # Determine bet outcome
                bet_won = self._evaluate_bet(
                    game, prediction, confidence, bet_type
                )

                # Calculate profit/loss
                if bet_won:
                    # Win (assuming -110 odds)
                    profit = bet_size * (100 / 110)
                else:
                    # Loss
                    profit = -bet_size

                # Update bankroll
                strategy.current_bankroll += profit

                # Record bet
                bet_record = {
                    'game_id': game.get('game_id'),
                    'season': game.get('season'),
                    'week': game.get('week'),
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'bet_type': bet_type,
                    'prediction': prediction,
                    'confidence': confidence,
                    'bet_size': bet_size,
                    'won': bet_won,
                    'profit': profit,
                    'bankroll': strategy.current_bankroll,
                }
                bet_history.append(bet_record)

            bankroll_history.append(strategy.current_bankroll)

        # Calculate metrics
        bet_df = pd.DataFrame(bet_history)

        if len(bet_df) > 0:
            metrics = BettingMetrics.calculate_all_metrics(
                bet_df,
                initial_bankroll=strategy.initial_bankroll
            )
        else:
            metrics = {
                'total_bets': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0,
            }

        results = {
            'strategy': strategy.name,
            'bet_type': bet_type,
            'metrics': metrics,
            'bet_history': bet_df,
            'bankroll_history': bankroll_history,
        }

        self.results.append(results)
        logger.info(f"Backtest complete - Win rate: {metrics.get('win_rate', 0):.2%}, ROI: {metrics.get('roi', 0):.2%}")

        return results

    def _evaluate_bet(
        self,
        game: pd.Series,
        prediction: float,
        confidence: float,
        bet_type: str
    ) -> bool:
        """
        Evaluate if a bet won.

        Args:
            game: Game information
            prediction: Model prediction
            confidence: Prediction confidence
            bet_type: Type of bet

        Returns:
            True if bet won, False otherwise
        """
        if bet_type == "spread":
            # Betting on home team to cover
            if 'home_spread' in game and not pd.isna(game['home_spread']):
                actual_diff = game.get('home_score', 0) - game.get('away_score', 0)
                home_covered = actual_diff + game['home_spread'] > 0

                # Our prediction: 1 = home wins, 0 = away wins
                bet_home = prediction == 1 or confidence > 0.5
                return bet_home == home_covered
            return False

        elif bet_type == "moneyline":
            # Straight up winner
            home_won = game.get('home_score', 0) > game.get('away_score', 0)
            bet_home = prediction == 1 or confidence > 0.5
            return bet_home == home_won

        elif bet_type == "total":
            # Over/under
            if 'total' in game and not pd.isna(game['total']):
                total_points = game.get('home_score', 0) + game.get('away_score', 0)
                went_over = total_points > game['total']

                # Prediction: 1 = over, 0 = under
                bet_over = prediction == 1 or confidence > 0.5
                return bet_over == went_over
            return False

        return False

    def compare_strategies(
        self,
        strategies: List[BettingStrategy],
        games: pd.DataFrame,
        predictions: pd.Series,
        confidences: pd.Series,
        bet_type: str = "spread"
    ) -> pd.DataFrame:
        """
        Compare multiple betting strategies.

        Args:
            strategies: List of strategies to compare
            games: Game data
            predictions: Model predictions
            confidences: Prediction confidences
            bet_type: Type of bet

        Returns:
            DataFrame comparing strategy performance
        """
        comparison_results = []

        for strategy in strategies:
            result = self.run_backtest(
                strategy, games, predictions, confidences, bet_type
            )

            metrics = result['metrics']
            comparison_results.append({
                'strategy': strategy.name,
                'total_bets': metrics.get('total_bets', 0),
                'win_rate': metrics.get('win_rate', 0),
                'total_profit': metrics.get('total_profit', 0),
                'roi': metrics.get('roi', 0),
                'final_bankroll': metrics.get('final_bankroll', 0),
                'max_drawdown': metrics.get('max_drawdown', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            })

        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('roi', ascending=False)

        return comparison_df

    def get_bet_history(self) -> pd.DataFrame:
        """Get complete bet history from all backtests."""
        all_bets = []
        for result in self.results:
            bet_df = result['bet_history'].copy()
            bet_df['strategy'] = result['strategy']
            all_bets.append(bet_df)

        if all_bets:
            return pd.concat(all_bets, ignore_index=True)
        return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    from config.config import PROCESSED_DATA_DIR, SAVED_MODELS_DIR
    from models.trainer import ModelTrainer
    from backtesting.strategies import (
        FlatBettingStrategy,
        KellyCriterionStrategy,
        ConfidenceThresholdStrategy,
        ValueBettingStrategy
    )

    # Load data
    features = pd.read_parquet(PROCESSED_DATA_DIR / "game_features.parquet")

    # Load model
    trainer = ModelTrainer()
    trainer.load_model('ensemble')
    model = trainer.models['ensemble']

    # Prepare data
    X, y = trainer.prepare_data(features, target_col='home_win')

    # Get predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Define strategies
    strategies = [
        FlatBettingStrategy(bet_amount=100),
        PercentageBettingStrategy(percentage=0.02),
        KellyCriterionStrategy(fraction=0.25),
        ConfidenceThresholdStrategy(confidence_threshold=0.6),
        ValueBettingStrategy(min_value=0.05),
    ]

    # Run backtest
    engine = BacktestEngine()
    results = engine.compare_strategies(
        strategies, features, predictions, probabilities
    )

    print("\nStrategy Comparison:")
    print(results)
