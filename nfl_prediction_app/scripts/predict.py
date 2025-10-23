"""Make predictions for upcoming games."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.trainer import ModelTrainer
from config.config import PROCESSED_DATA_DIR
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def predict_games(week: int = None, season: int = None):
    """
    Make predictions for games in a specific week/season.

    Args:
        week: Week number (if None, uses most recent)
        season: Season year (if None, uses most recent)
    """
    logger.info("Loading data and model...")

    # Load data
    features = pd.read_parquet(PROCESSED_DATA_DIR / "game_features.parquet")

    # Load model
    trainer = ModelTrainer()
    trainer.load_model('ensemble')
    model = trainer.models['ensemble']

    # Filter to specific week/season if provided
    if season:
        features = features[features['season'] == season]
    if week:
        features = features[features['week'] == week]
    else:
        # Get most recent week
        features = features.tail(20)

    if len(features) == 0:
        logger.error("No games found for specified criteria")
        return

    # Prepare features
    X, y = trainer.prepare_data(features, target_col='home_win')

    # Make predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    # Display predictions
    print("\n" + "=" * 80)
    print("NFL GAME PREDICTIONS")
    print("=" * 80)

    for idx, (_, game) in enumerate(features.iterrows()):
        home_team = game['home_team']
        away_team = game['away_team']
        season = game['season']
        week = game['week']

        prediction = predictions[idx]
        confidence = probabilities[idx]

        predicted_winner = home_team if prediction == 1 else away_team
        win_prob = confidence if prediction == 1 else (1 - confidence)

        print(f"\nSeason {season}, Week {week}")
        print(f"  {away_team} @ {home_team}")
        print(f"  Predicted Winner: {predicted_winner}")
        print(f"  Win Probability: {win_prob:.1%}")

        if 'home_spread' in game and pd.notna(game['home_spread']):
            spread = game['home_spread']
            print(f"  Spread: {home_team} {spread:+.1f}")

        # Show actual result if available
        if 'home_score' in game and pd.notna(game['home_score']):
            home_score = game['home_score']
            away_score = game['away_score']
            actual_winner = home_team if home_score > away_score else away_team
            correct = "✓" if actual_winner == predicted_winner else "✗"
            print(f"  Actual: {home_team} {int(home_score)} - {away_team} {int(away_score)} {correct}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Make NFL game predictions')
    parser.add_argument('--week', type=int, help='Week number')
    parser.add_argument('--season', type=int, help='Season year')

    args = parser.parse_args()

    predict_games(week=args.week, season=args.season)
