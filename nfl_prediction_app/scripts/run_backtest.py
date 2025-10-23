"""Run backtests on historical data."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.trainer import ModelTrainer
from backtesting.engine import BacktestEngine
from backtesting.strategies import (
    FlatBettingStrategy,
    PercentageBettingStrategy,
    KellyCriterionStrategy,
    ConfidenceThresholdStrategy,
    ValueBettingStrategy,
    MarketBiasStrategy
)
from backtesting.metrics import BettingMetrics
from config.config import PROCESSED_DATA_DIR, STARTING_BANKROLL
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run comprehensive backtest analysis."""
    logger.info("=" * 60)
    logger.info("NFL BETTING STRATEGY BACKTEST")
    logger.info("=" * 60)

    # Load data
    logger.info("\nLoading data...")
    features = pd.read_parquet(PROCESSED_DATA_DIR / "game_features.parquet")
    logger.info(f"Loaded {len(features)} games")

    # Load model
    logger.info("\nLoading trained model...")
    trainer = ModelTrainer()
    trainer.load_model('ensemble')
    model = trainer.models['ensemble']

    # Prepare data
    X, y = trainer.prepare_data(features, target_col='home_win')

    # Get predictions
    logger.info("\nGenerating predictions...")
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Define strategies to test
    strategies = [
        FlatBettingStrategy(bet_amount=100, bankroll=STARTING_BANKROLL),
        PercentageBettingStrategy(percentage=0.02, bankroll=STARTING_BANKROLL),
        KellyCriterionStrategy(fraction=0.25, min_edge=0.02, bankroll=STARTING_BANKROLL),
        KellyCriterionStrategy(fraction=0.5, min_edge=0.02, bankroll=STARTING_BANKROLL),
        ConfidenceThresholdStrategy(confidence_threshold=0.55, bankroll=STARTING_BANKROLL),
        ConfidenceThresholdStrategy(confidence_threshold=0.60, bankroll=STARTING_BANKROLL),
        ConfidenceThresholdStrategy(confidence_threshold=0.65, bankroll=STARTING_BANKROLL),
        ValueBettingStrategy(min_value=0.03, bankroll=STARTING_BANKROLL),
        ValueBettingStrategy(min_value=0.05, bankroll=STARTING_BANKROLL),
        MarketBiasStrategy(target_bias="home_underdog", bankroll=STARTING_BANKROLL),
    ]

    # Run backtests
    logger.info("\nRunning backtests...")
    engine = BacktestEngine()
    comparison = engine.compare_strategies(
        strategies, features, predictions, probabilities
    )

    # Display results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    print("\nStrategy Comparison (sorted by ROI):")
    print(comparison.to_string(index=False))

    # Detailed results for best strategy
    best_strategy_name = comparison.iloc[0]['strategy']
    logger.info(f"\n\nDetailed results for best strategy: {best_strategy_name}")

    best_result = [r for r in engine.results if r['strategy'] == best_strategy_name][0]
    BettingMetrics.print_metrics_report(best_result['metrics'])

    # Analysis by season
    if len(best_result['bet_history']) > 0:
        print("\n" + "=" * 60)
        print("PERFORMANCE BY SEASON")
        print("=" * 60)

        bet_history = best_result['bet_history']
        season_results = bet_history.groupby('season').agg({
            'won': ['sum', 'count', 'mean'],
            'profit': 'sum',
            'bet_size': 'sum'
        }).round(2)

        season_results.columns = ['Wins', 'Total Bets', 'Win Rate', 'Profit', 'Total Wagered']
        season_results['ROI'] = (season_results['Profit'] / season_results['Total Wagered'] * 100).round(2)

        print(season_results.to_string())

    # Save results
    output_path = Path(__file__).parent.parent / "backtest_results.csv"
    comparison.to_csv(output_path, index=False)
    logger.info(f"\n\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
