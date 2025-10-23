# NFL Game Prediction & Betting Strategy Backtester

A comprehensive application for predicting NFL game outcomes and backtesting betting strategies using historical data from NFLfastR.

## Features

- **Game Outcome Prediction**: Machine learning models to predict NFL game winners, point spreads, and totals
- **Historical Data Access**: Integration with NFLfastR for comprehensive NFL play-by-play and game data (2015-2024)
- **Betting Strategy Backtesting**: Framework to test various betting strategies on historical data
- **Performance Metrics**: Calculate win rates, ROI, Kelly criterion, Sharpe ratio, and other betting metrics
- **Web Interface**: User-friendly interface to view predictions and backtest results
- **Visualization**: Interactive charts and graphs for model performance and betting results
- **Multiple Strategies**: Test flat betting, percentage betting, Kelly Criterion, confidence thresholds, value betting, and more

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline (data collection, feature engineering, model training)
python scripts/run_pipeline.py

# 3. Start the web application
python web_app/app.py

# 4. Open browser to http://localhost:5000
```

For detailed setup instructions, see [GETTING_STARTED.md](GETTING_STARTED.md)

## Project Structure

```
nfl_prediction_app/
├── data/                   # Data collection and processing
│   ├── collectors.py       # NFLfastR data collection
│   ├── features.py         # Feature engineering
│   ├── raw/                # Raw data storage
│   └── processed/          # Processed features
├── models/                 # Predictive models
│   ├── base_model.py       # Base model interface
│   ├── predictor.py        # Model implementations
│   ├── trainer.py          # Model training pipeline
│   └── saved_models/       # Trained models
├── backtesting/            # Betting strategy backtesting
│   ├── strategies.py       # Betting strategies (Kelly, flat, etc.)
│   ├── engine.py           # Backtesting engine
│   └── metrics.py          # Performance metrics
├── web_app/                # Flask web application
│   ├── app.py              # Main application
│   ├── static/             # CSS, JS files
│   └── templates/          # HTML templates
├── scripts/                # Utility scripts
│   ├── run_pipeline.py     # Complete data pipeline
│   ├── run_backtest.py     # Run backtests
│   └── predict.py          # Make predictions
├── config/                 # Configuration files
│   └── config.py           # App settings
├── notebooks/              # Jupyter notebooks for analysis
│   └── exploratory_analysis.ipynb
└── tests/                  # Unit tests
    ├── test_models.py
    └── test_strategies.py
```

## Usage Examples

### Command Line

#### Make Predictions
```bash
# Predict recent games
python scripts/predict.py

# Predict specific week
python scripts/predict.py --week 5 --season 2024
```

#### Run Backtests
```bash
python scripts/run_backtest.py
```

### Python API

#### Collect Data
```python
from data.collectors import NFLDataCollector

collector = NFLDataCollector()
data = collector.collect_all_data(start_year=2020, end_year=2024)
```

#### Train Models
```python
from models.trainer import ModelTrainer

trainer = ModelTrainer()
X, y = trainer.prepare_data(features, target_col='home_win')
results = trainer.train_all_models(X_train, y_train, X_test, y_test)
```

#### Backtest Strategy
```python
from backtesting.engine import BacktestEngine
from backtesting.strategies import KellyCriterionStrategy

strategy = KellyCriterionStrategy(fraction=0.25, bankroll=10000)
engine = BacktestEngine()
results = engine.run_backtest(strategy, games, predictions, confidences)
```

## Betting Strategies Implemented

1. **Flat Betting**: Fixed bet size for every game
2. **Percentage Betting**: Bet a fixed percentage of current bankroll
3. **Kelly Criterion**: Optimal bet sizing based on edge and odds
4. **Confidence Threshold**: Only bet when model confidence exceeds threshold
5. **Value Betting**: Bet when model probability differs from market odds
6. **Market Bias**: Exploit known market inefficiencies

## Machine Learning Models

- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with advanced features
- **LightGBM**: Fast gradient boosting
- **Ensemble**: Weighted combination of all models

## Key Metrics

### Prediction Metrics
- Accuracy, Precision, Recall, F1 Score
- ROC AUC, Log Loss
- Calibration plots

### Betting Metrics
- Win Rate (hit rate)
- Return on Investment (ROI)
- Total Profit/Loss
- Maximum Drawdown
- Sharpe Ratio (risk-adjusted return)
- Risk of Ruin
- Profit Factor

## Data Sources

### Historical Data (Training)
All historical data comes from [NFLfastR](https://www.nflfastr.com/) via the [nfl_data_py](https://github.com/nflverse/nfl_data_py) package:
- Play-by-play data (2015-2024)
- Game schedules and results
- Weekly team statistics
- Roster information
- Injury reports (where available)
- **Updates**: Within 15 minutes after each game

### Real-Time Data (Predictions)
For upcoming games and live odds, integrate with:
- **[The Odds API](https://the-odds-api.com/)**: Live betting lines (free tier: 500 req/month)
- **ESPN API**: Unofficial but free game schedules
- **[SportsDataIO](https://sportsdata.io/)**: Professional-grade data (paid)

See [docs/REALTIME_DATA.md](docs/REALTIME_DATA.md) for complete integration guide.

## Configuration

Edit `config/config.py` to customize:

```python
# Data settings
NFL_START_YEAR = 2015
NFL_CURRENT_YEAR = 2024

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Betting settings
STARTING_BANKROLL = 10000
KELLY_FRACTION = 0.25
MIN_EDGE = 0.02
```

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Or with coverage:
```bash
pytest tests/ --cov=. --cov-report=html
```

## Web Application

The Flask web app provides:
- **Predictions Tab**: View recent game predictions with confidence levels
- **Backtest Tab**: Run individual strategy backtests
- **Compare Tab**: Compare all strategies side-by-side
- **Models Tab**: View model information and feature importance

Access at `http://localhost:5000` after starting with:
```bash
python web_app/app.py
```

## Performance

Based on historical backtests (2015-2024):
- Prediction accuracy: ~55-60% (baseline: 52% home win rate)
- Best strategies show positive ROI on historical data
- Kelly Criterion and Value Betting typically outperform flat betting
- Performance varies by season and betting market efficiency

**Note**: Past performance does not guarantee future results.

## Contributing

Contributions welcome! Areas for improvement:
- Additional prediction features (weather, injuries, rest days)
- More sophisticated models (neural networks, time series)
- Real-time odds integration
- Live game predictions
- Enhanced visualizations

## Roadmap

- [ ] Integration with live odds APIs
- [ ] Player-level predictions (props)
- [ ] In-game live betting models
- [ ] Automated bet placement (paper trading)
- [ ] Mobile app interface
- [ ] Advanced time series models
- [ ] Injury impact modeling

## License

MIT License - See LICENSE file for details

## Disclaimer

**IMPORTANT**: This application is for educational and entertainment purposes only.

- Sports betting involves risk and you may lose money
- Past performance does not indicate future results
- No prediction model is perfect
- Always gamble responsibly and within your means
- This is not financial advice
- Check local laws regarding sports betting

The authors are not responsible for any financial losses incurred from using this software.

## Acknowledgments

- [NFLfastR](https://www.nflfastr.com/) for providing comprehensive NFL data
- [nfl_data_py](https://github.com/nflverse/nfl_data_py) for the Python wrapper
- The open source ML community for excellent tools

## Support

For questions or issues:
1. Check [GETTING_STARTED.md](GETTING_STARTED.md)
2. Review the example notebooks
3. Open an issue on GitHub

---

**Remember**: Bet responsibly. This is a learning tool, not a guaranteed profit system.
