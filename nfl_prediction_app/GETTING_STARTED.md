# Getting Started with NFL Game Predictor

This guide will help you get up and running with the NFL Game Prediction and Betting Strategy Backtesting application.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection (for downloading NFL data)

## Installation

### 1. Clone or Download the Repository

```bash
cd nfl_prediction_app
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Linux/Mac:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install all required packages including:
- nfl-data-py (NFLfastR data access)
- pandas, numpy (data processing)
- scikit-learn, xgboost, lightgbm (machine learning)
- flask (web application)
- plotly, matplotlib (visualizations)

## Quick Start

### Option 1: Run Complete Pipeline (Recommended)

This will collect data, engineer features, and train models automatically:

```bash
python scripts/run_pipeline.py
```

This process may take 30-60 minutes depending on your internet connection and computer speed.

### Option 2: Step-by-Step

If you prefer to run each step individually:

#### Step 1: Collect Data

```python
python -c "
from data.collectors import NFLDataCollector
from config.config import RAW_DATA_DIR

collector = NFLDataCollector()
data = collector.collect_all_data(start_year=2015, end_year=2024)
collector.save_data(data, RAW_DATA_DIR)
"
```

#### Step 2: Engineer Features

```python
python -c "
from data.features import FeatureEngineeer
from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
import pandas as pd

schedule = pd.read_parquet(RAW_DATA_DIR / 'schedule.parquet')
pbp = pd.read_parquet(RAW_DATA_DIR / 'pbp.parquet')
team_stats = pd.read_parquet(RAW_DATA_DIR / 'team_stats.parquet')

engineer = FeatureEngineeer()
features = engineer.engineer_all_features(schedule, pbp, team_stats)
features.to_parquet(PROCESSED_DATA_DIR / 'game_features.parquet', index=False)
"
```

#### Step 3: Train Models

```bash
python models/trainer.py
```

## Using the Application

### 1. Web Interface

Start the web application:

```bash
python web_app/app.py
```

Then open your browser to: `http://localhost:5000`

The web interface allows you to:
- View game predictions
- Run backtests with different strategies
- Compare strategy performance
- View model information

### 2. Command Line Tools

#### Make Predictions

Predict all recent games:
```bash
python scripts/predict.py
```

Predict specific week:
```bash
python scripts/predict.py --week 5 --season 2024
```

#### Run Backtests

```bash
python scripts/run_backtest.py
```

This will:
- Test multiple betting strategies
- Show detailed performance metrics
- Save results to CSV

### 3. Jupyter Notebooks

Explore the data interactively:

```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Understanding the Results

### Prediction Confidence Levels

- **High (>65%)**: Strong confidence in prediction
- **Medium (55-65%)**: Moderate confidence
- **Low (<55%)**: Weak confidence, close to coin flip

### Betting Strategy Metrics

- **Win Rate**: Percentage of bets won
- **ROI**: Return on investment (profit / total wagered)
- **Max Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Risk of Ruin**: Estimated probability of losing all bankroll

### Recommended Strategies

1. **Kelly Criterion**: Optimal bet sizing based on edge (recommended for experienced bettors)
2. **Confidence Threshold**: Only bet when model is confident (good for conservative approach)
3. **Value Betting**: Bet when model finds value vs market odds

## Configuration

Edit `config/config.py` to customize:

- Data range (years to collect)
- Model parameters
- Betting parameters (starting bankroll, Kelly fraction, etc.)
- Web app settings

## Troubleshooting

### Data Download Issues

If data download fails:
1. Check your internet connection
2. Try downloading a smaller date range
3. Use the retry logic built into the collectors

### Memory Issues

If you run out of memory:
1. Reduce the date range in config
2. Process data in chunks
3. Use a machine with more RAM

### Model Training Slow

If training is too slow:
1. Reduce the number of models
2. Decrease data size
3. Use fewer features
4. Consider using a GPU

## Next Steps

1. **Customize Models**: Experiment with different model parameters
2. **Create New Strategies**: Implement your own betting strategies
3. **Add Features**: Engineer new predictive features
4. **Live Predictions**: Set up automated predictions for upcoming games

## Important Notes

- **For Educational Purposes**: This tool is for learning and entertainment
- **Gamble Responsibly**: Never bet more than you can afford to lose
- **Past Performance**: Historical results don't guarantee future performance
- **Market Efficiency**: Real betting markets are highly efficient

## Support

For issues or questions:
1. Check the documentation
2. Review example notebooks
3. Open an issue on GitHub

## License

MIT License - See LICENSE file for details
