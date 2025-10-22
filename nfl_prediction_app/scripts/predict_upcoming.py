"""Make predictions for upcoming NFL games using real-time data."""
import sys
from pathlib import Path
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.realtime_sources import NFLDataIntegration, TheOddsAPIClient, ESPNAPIClient
from models.trainer import ModelTrainer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Predict upcoming NFL games."""
    logger.info("=" * 80)
    logger.info("UPCOMING NFL GAME PREDICTIONS")
    logger.info("=" * 80)

    # Check for API key
    odds_api_key = os.getenv('ODDS_API_KEY')

    if not odds_api_key:
        logger.warning("\n⚠️  No ODDS_API_KEY found in environment variables")
        logger.warning("To get upcoming games with live odds:")
        logger.warning("1. Sign up at https://the-odds-api.com/ (free tier available)")
        logger.warning("2. Set environment variable: export ODDS_API_KEY='your_key'")
        logger.warning("\nFalling back to ESPN API (free, but limited odds data)\n")

    # Initialize data integration
    integration = NFLDataIntegration(odds_api_key=odds_api_key)

    # Get upcoming games
    logger.info("\nFetching upcoming games...")
    upcoming_games = integration.get_games_for_prediction(upcoming_only=True)

    if upcoming_games.empty:
        logger.error("\n❌ No upcoming games found")
        logger.info("\nPossible reasons:")
        logger.info("- It's the off-season")
        logger.info("- All games this week have been played")
        logger.info("- API connection issue")
        return

    logger.info(f"\n✅ Found {len(upcoming_games)} upcoming games\n")

    # Display games
    print("\n" + "=" * 80)
    print("UPCOMING GAMES")
    print("=" * 80)

    for idx, game in upcoming_games.iterrows():
        print(f"\n{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}")

        if 'commence_time' in game:
            game_time = game['commence_time']
            print(f"  Kickoff: {game_time}")

        if 'home_spread' in game and game['home_spread'] is not None:
            spread = game['home_spread']
            print(f"  Spread: {game['home_team']} {spread:+.1f}")

        if 'total' in game and game['total'] is not None:
            print(f"  Total: {game['total']}")

        if 'home_moneyline' in game and game['home_moneyline'] is not None:
            print(f"  Moneyline: {game['home_team']} {game['home_moneyline']:+d}, "
                  f"{game['away_team']} {game.get('away_moneyline', 0):+d}")

    # Check if model exists
    print("\n" + "=" * 80)
    print("MODEL PREDICTIONS")
    print("=" * 80)

    try:
        trainer = ModelTrainer()
        trainer.load_model('ensemble')
        logger.info("\n✅ Model loaded successfully")

        logger.warning("\n⚠️  Note: Model predictions require feature engineering")
        logger.warning("Current implementation shows games but doesn't calculate features yet.")
        logger.warning("\nTo enable predictions, you need to:")
        logger.warning("1. Fetch current season team statistics")
        logger.warning("2. Calculate rolling averages and EPA metrics")
        logger.warning("3. Create matchup features matching training data")
        logger.warning("\nSee docs/REALTIME_DATA.md for complete workflow")

    except Exception as e:
        logger.error(f"\n❌ Could not load model: {e}")
        logger.info("\nPlease run the training pipeline first:")
        logger.info("  python scripts/run_pipeline.py")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
