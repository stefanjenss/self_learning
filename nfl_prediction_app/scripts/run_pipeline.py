"""Complete pipeline to collect data, engineer features, and train models."""
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data.collectors import NFLDataCollector
from data.features import FeatureEngineeer
from models.trainer import ModelTrainer
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, NFL_START_YEAR, NFL_CURRENT_YEAR
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run the complete pipeline."""
    logger.info("=" * 60)
    logger.info("NFL PREDICTION MODEL PIPELINE")
    logger.info("=" * 60)

    # Step 1: Collect data
    logger.info("\nStep 1: Collecting NFL data from NFLfastR...")
    collector = NFLDataCollector()

    try:
        data = collector.collect_all_data(
            start_year=NFL_START_YEAR,
            end_year=NFL_CURRENT_YEAR
        )
        collector.save_data(data, RAW_DATA_DIR)
        logger.info("Data collection complete!")
    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        logger.info("Attempting to load existing data...")
        try:
            data = {
                'pbp': pd.read_parquet(RAW_DATA_DIR / "pbp.parquet"),
                'schedule': pd.read_parquet(RAW_DATA_DIR / "schedule.parquet"),
                'team_stats': pd.read_parquet(RAW_DATA_DIR / "team_stats.parquet"),
            }
            logger.info("Loaded existing data successfully")
        except Exception as load_error:
            logger.error(f"Could not load existing data: {load_error}")
            return

    # Step 2: Feature engineering
    logger.info("\nStep 2: Engineering features...")
    engineer = FeatureEngineeer()

    try:
        features = engineer.engineer_all_features(
            schedule=data['schedule'],
            pbp=data['pbp'],
            weekly_stats=data['team_stats']
        )

        # Save features
        output_path = PROCESSED_DATA_DIR / "game_features.parquet"
        features.to_parquet(output_path, index=False)
        logger.info(f"Features saved to {output_path}")
        logger.info(f"Created {len(features)} game records with {len(features.columns)} features")
    except Exception as e:
        logger.error(f"Error in feature engineering: {e}")
        return

    # Step 3: Train models
    logger.info("\nStep 3: Training models...")
    trainer = ModelTrainer()

    try:
        X, y = trainer.prepare_data(features, target_col='home_win')

        # Split data chronologically (no shuffle for time series)
        from sklearn.model_selection import train_test_split
        from config.config import TEST_SIZE, RANDOM_STATE

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, shuffle=False
        )

        logger.info(f"Training set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")

        # Train all models
        results = trainer.train_all_models(X_train, y_train, X_test, y_test)

        logger.info("\nModel Results:")
        print(results[['model', 'test_accuracy', 'test_auc', 'test_logloss']].to_string(index=False))

        # Save models
        trainer.save_models()
        logger.info("\nModels saved successfully!")

    except Exception as e:
        logger.error(f"Error training models: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 4: Summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 60)
    logger.info(f"\nData collected: {NFL_START_YEAR} - {NFL_CURRENT_YEAR}")
    logger.info(f"Features created: {len(features)} games")
    logger.info(f"Models trained: {len(trainer.models)}")
    logger.info(f"Best model: {trainer.best_model.name}")
    logger.info("\nYou can now:")
    logger.info("1. Run the web app: python web_app/app.py")
    logger.info("2. Run backtests: python scripts/run_backtest.py")
    logger.info("3. Make predictions: python scripts/predict.py")


if __name__ == "__main__":
    main()
