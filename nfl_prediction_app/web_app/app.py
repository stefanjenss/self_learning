"""Main Flask application."""
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
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
    ValueBettingStrategy
)
from config.config import FLASK_HOST, FLASK_PORT, FLASK_DEBUG, PROCESSED_DATA_DIR, SAVED_MODELS_DIR

app = Flask(__name__)
CORS(app)

# Global variables to cache loaded data
model_trainer = None
data_cache = {}


def load_models():
    """Load trained models."""
    global model_trainer
    if model_trainer is None:
        model_trainer = ModelTrainer()
        try:
            model_trainer.load_model('ensemble')
        except Exception as e:
            print(f"Could not load ensemble model: {e}")
            print("Please train models first by running: python models/trainer.py")
    return model_trainer


def load_data():
    """Load processed data."""
    global data_cache
    if 'features' not in data_cache:
        try:
            features_path = PROCESSED_DATA_DIR / "game_features.parquet"
            if features_path.exists():
                data_cache['features'] = pd.read_parquet(features_path)
            else:
                print("Processed features not found. Please run feature engineering first.")
                data_cache['features'] = pd.DataFrame()
        except Exception as e:
            print(f"Error loading data: {e}")
            data_cache['features'] = pd.DataFrame()
    return data_cache['features']


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/predictions')
def get_predictions():
    """Get predictions for upcoming/recent games."""
    try:
        trainer = load_models()
        data = load_data()

        if data.empty:
            return jsonify({'error': 'No data available'}), 404

        if not trainer.models:
            return jsonify({'error': 'Models not loaded'}), 404

        # Get recent games
        recent_games = data.tail(20).copy()

        # Prepare features
        X, y = trainer.prepare_data(recent_games, target_col='home_win')

        # Get predictions
        model = trainer.models.get('ensemble') or list(trainer.models.values())[0]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Prepare response
        results = []
        for idx, (_, game) in enumerate(recent_games.iterrows()):
            result = {
                'game_id': str(game.get('game_id', '')),
                'season': int(game.get('season', 0)),
                'week': int(game.get('week', 0)),
                'home_team': str(game.get('home_team', '')),
                'away_team': str(game.get('away_team', '')),
                'prediction': int(predictions[idx]),
                'confidence': float(probabilities[idx]),
                'predicted_winner': game['home_team'] if predictions[idx] == 1 else game['away_team'],
                'home_score': int(game.get('home_score', 0)) if pd.notna(game.get('home_score')) else None,
                'away_score': int(game.get('away_score', 0)) if pd.notna(game.get('away_score')) else None,
            }
            results.append(result)

        return jsonify(results)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest with specified strategy."""
    try:
        params = request.get_json()
        strategy_name = params.get('strategy', 'kelly')
        bankroll = params.get('bankroll', 10000)

        trainer = load_models()
        data = load_data()

        if data.empty or not trainer.models:
            return jsonify({'error': 'Data or models not available'}), 404

        # Prepare data
        X, y = trainer.prepare_data(data, target_col='home_win')

        # Get predictions
        model = trainer.models.get('ensemble') or list(trainer.models.values())[0]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Select strategy
        if strategy_name == 'flat':
            strategy = FlatBettingStrategy(bet_amount=100, bankroll=bankroll)
        elif strategy_name == 'percentage':
            strategy = PercentageBettingStrategy(percentage=0.02, bankroll=bankroll)
        elif strategy_name == 'kelly':
            strategy = KellyCriterionStrategy(fraction=0.25, bankroll=bankroll)
        elif strategy_name == 'confidence':
            strategy = ConfidenceThresholdStrategy(confidence_threshold=0.6, bankroll=bankroll)
        elif strategy_name == 'value':
            strategy = ValueBettingStrategy(min_value=0.05, bankroll=bankroll)
        else:
            return jsonify({'error': 'Invalid strategy'}), 400

        # Run backtest
        engine = BacktestEngine()
        results = engine.run_backtest(strategy, data, predictions, probabilities)

        # Format response
        metrics = results['metrics']
        bet_history = results['bet_history']

        response = {
            'strategy': results['strategy'],
            'metrics': {
                'total_bets': int(metrics['total_bets']),
                'win_rate': float(metrics['win_rate']),
                'total_profit': float(metrics['total_profit']),
                'roi': float(metrics['roi']),
                'final_bankroll': float(metrics['final_bankroll']),
                'max_drawdown': float(metrics['max_drawdown']),
                'sharpe_ratio': float(metrics.get('sharpe_ratio', 0)),
            },
            'bankroll_history': results['bankroll_history'],
            'bet_count': len(bet_history),
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/compare_strategies', methods=['POST'])
def compare_strategies():
    """Compare multiple betting strategies."""
    try:
        params = request.get_json()
        bankroll = params.get('bankroll', 10000)

        trainer = load_models()
        data = load_data()

        if data.empty or not trainer.models:
            return jsonify({'error': 'Data or models not available'}), 404

        # Prepare data
        X, y = trainer.prepare_data(data, target_col='home_win')

        # Get predictions
        model = trainer.models.get('ensemble') or list(trainer.models.values())[0]
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]

        # Define strategies to compare
        strategies = [
            FlatBettingStrategy(bet_amount=100, bankroll=bankroll),
            PercentageBettingStrategy(percentage=0.02, bankroll=bankroll),
            KellyCriterionStrategy(fraction=0.25, bankroll=bankroll),
            ConfidenceThresholdStrategy(confidence_threshold=0.6, bankroll=bankroll),
            ValueBettingStrategy(min_value=0.05, bankroll=bankroll),
        ]

        # Run comparison
        engine = BacktestEngine()
        comparison_df = engine.compare_strategies(strategies, data, predictions, probabilities)

        # Convert to dict
        comparison = comparison_df.to_dict('records')

        return jsonify(comparison)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model_info')
def model_info():
    """Get information about loaded models."""
    try:
        trainer = load_models()

        if not trainer.models:
            return jsonify({'error': 'No models loaded'}), 404

        model_list = []
        for name, model in trainer.models.items():
            info = {
                'name': name,
                'fitted': model.is_fitted,
                'features': len(model.feature_columns) if model.feature_columns else 0,
            }

            # Get feature importance if available
            importance = model.get_feature_importance()
            if importance is not None:
                info['top_features'] = importance.head(10).to_dict('records')

            model_list.append(info)

        return jsonify(model_list)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=FLASK_DEBUG)
