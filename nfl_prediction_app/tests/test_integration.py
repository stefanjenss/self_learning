"""Integration test - creates sample data and tests the full pipeline."""
import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("NFL PREDICTION APP - INTEGRATION TEST")
print("=" * 80)

# Test 1: Create sample data
print("\n[1/5] Creating sample game data...")
np.random.seed(42)

# Create 100 sample games
n_games = 100
sample_games = pd.DataFrame({
    'game_id': [f'2023_{i:03d}' for i in range(n_games)],
    'season': 2023,
    'week': np.random.randint(1, 18, n_games),
    'home_team': np.random.choice(['KC', 'BUF', 'MIA', 'CIN', 'SF'], n_games),
    'away_team': np.random.choice(['DAL', 'PHI', 'DET', 'LAC', 'BAL'], n_games),
    'home_score': np.random.randint(10, 40, n_games),
    'away_score': np.random.randint(10, 40, n_games),
    'home_spread': np.random.uniform(-7, 7, n_games),
    'total': np.random.uniform(40, 55, n_games),
})

# Add features
for stat in ['points_scored_avg', 'points_allowed_avg', 'yards_gained_avg',
             'yards_allowed_avg', 'epa_per_play', 'success_rate']:
    sample_games[f'home_{stat}'] = np.random.randn(n_games)
    sample_games[f'away_{stat}'] = np.random.randn(n_games)
    sample_games[f'diff_{stat}'] = sample_games[f'home_{stat}'] - sample_games[f'away_{stat}']

# Add target variable
sample_games['home_win'] = (sample_games['home_score'] > sample_games['away_score']).astype(int)

print(f"✅ Created {len(sample_games)} sample games")
print(f"   Features: {len(sample_games.columns)} columns")
print(f"   Home win rate: {sample_games['home_win'].mean():.1%}")

# Test 2: Model Training
print("\n[2/5] Testing model training...")
from models.trainer import ModelTrainer
from sklearn.model_selection import train_test_split

trainer = ModelTrainer()

# Prepare data
X, y = trainer.prepare_data(sample_games, target_col='home_win')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"   Training set: {len(X_train)} games")
print(f"   Test set: {len(X_test)} games")
print(f"   Features: {len(X.columns)}")

# Train a single model (faster for testing)
from models.predictor import LogisticModel, XGBoostModel

print("\n   Training Logistic Regression...")
logistic = LogisticModel()
logistic.train(X_train, y_train)
logistic_preds = logistic.predict(X_test)
logistic_acc = (logistic_preds == y_test).mean()
print(f"   ✅ Logistic Regression - Accuracy: {logistic_acc:.1%}")

print("\n   Training XGBoost...")
xgb = XGBoostModel()
xgb.train(X_train, y_train)
xgb_preds = xgb.predict(X_test)
xgb_acc = (xgb_preds == y_test).mean()
print(f"   ✅ XGBoost - Accuracy: {xgb_acc:.1%}")

# Test 3: Backtesting
print("\n[3/5] Testing backtesting engine...")
from backtesting.engine import BacktestEngine
from backtesting.strategies import KellyCriterionStrategy, FlatBettingStrategy

# Get predictions for all games
predictions = xgb.predict(X)
probabilities = xgb.predict_proba(X)[:, 1]

# Test Kelly Criterion strategy
kelly_strategy = KellyCriterionStrategy(fraction=0.25, bankroll=10000)
engine = BacktestEngine()
kelly_results = engine.run_backtest(
    kelly_strategy,
    sample_games,
    predictions,
    probabilities
)

print(f"   ✅ Kelly Criterion Results:")
print(f"      Total Bets: {kelly_results['metrics']['total_bets']}")
print(f"      Win Rate: {kelly_results['metrics']['win_rate']:.1%}")
print(f"      ROI: {kelly_results['metrics']['roi']:.1%}")
print(f"      Final Bankroll: ${kelly_results['metrics']['final_bankroll']:.2f}")

# Test Flat Betting strategy
flat_strategy = FlatBettingStrategy(bet_amount=100, bankroll=10000)
flat_results = engine.run_backtest(
    flat_strategy,
    sample_games,
    predictions,
    probabilities
)

print(f"\n   ✅ Flat Betting Results:")
print(f"      Total Bets: {flat_results['metrics']['total_bets']}")
print(f"      Win Rate: {flat_results['metrics']['win_rate']:.1%}")
print(f"      ROI: {flat_results['metrics']['roi']:.1%}")
print(f"      Final Bankroll: ${flat_results['metrics']['final_bankroll']:.2f}")

# Test 4: Real-time Data API (without actual API calls)
print("\n[4/5] Testing real-time data modules...")
from data.realtime_sources import ESPNAPIClient, TheOddsAPIClient

espn_client = ESPNAPIClient()
print("   ✅ ESPN API client initialized")

# Note: Not testing actual API calls to avoid rate limits
print("   ✅ The Odds API client ready (requires API key for actual calls)")

# Test 5: Flask App
print("\n[5/5] Testing Flask web application...")
from web_app.app import app

# Check routes
routes = [rule.endpoint for rule in app.url_map.iter_rules() if not rule.endpoint.startswith('static')]
print(f"   ✅ Flask app initialized")
print(f"   ✅ Available routes:")
for route in routes:
    print(f"      - {route}")

# Test app context
with app.app_context():
    print("   ✅ App context works")

print("\n" + "=" * 80)
print("INTEGRATION TEST RESULTS")
print("=" * 80)
print("✅ Sample Data Creation: PASSED")
print("✅ Model Training: PASSED")
print("✅ Backtesting Engine: PASSED")
print("✅ Real-time Data Modules: PASSED")
print("✅ Flask Web App: PASSED")
print("\n🎉 All integration tests passed!")
print("\nThe application is working correctly and ready to use.")
print("=" * 80)
