# Real-Time Data Integration Guide

This guide explains how to get current/upcoming game data for making predictions.

## The Problem

NFLfastR provides excellent **historical** data but only for **completed games**. For predicting upcoming games, you need:

1. Current week's schedule
2. Live betting lines (spreads, totals, moneylines)
3. Recent injury updates
4. Current team form/statistics

## Recommended Solutions

### Option 1: The Odds API (Recommended)

**Best for**: Betting lines and upcoming games
**Cost**: Free tier (500 requests/month) - sufficient for personal use
**Signup**: https://the-odds-api.com/

#### Features:
- ✅ Upcoming games with start times
- ✅ Live odds from 50+ sportsbooks
- ✅ Spreads, totals, moneylines
- ✅ Simple JSON API
- ✅ Free tier available

#### Setup:

```bash
# 1. Sign up at https://the-odds-api.com/
# 2. Get your API key
# 3. Set environment variable
export ODDS_API_KEY="your_api_key_here"
```

#### Usage:

```python
from data.realtime_sources import TheOddsAPIClient

client = TheOddsAPIClient(api_key="your_key")
upcoming = client.get_upcoming_games()

print(upcoming[['home_team', 'away_team', 'home_spread', 'total']])
```

### Option 2: ESPN API (Free, Unofficial)

**Best for**: Schedules and basic game info
**Cost**: Free
**Limitations**: Unofficial API, may change without notice

#### Features:
- ✅ Current week's games
- ✅ Scores and schedules
- ✅ Basic odds (not comprehensive)
- ✅ No API key required

#### Usage:

```python
from data.realtime_sources import ESPNAPIClient

client = ESPNAPIClient()
games = client.get_scoreboard()

print(games[['home_team', 'away_team', 'spread', 'over_under']])
```

### Option 3: SportsDataIO (Most Comprehensive)

**Best for**: Professional/commercial use
**Cost**: Starts at $25/month
**URL**: https://sportsdata.io/nfl-api

#### Features:
- ✅ Real-time scores
- ✅ Advanced statistics
- ✅ Injury reports
- ✅ Weather data
- ✅ Player props
- ✅ Commercial license

## Workflow for Predictions

### Step 1: Train Models on Historical Data (One-time)

```bash
python scripts/run_pipeline.py
```

This uses NFLfastR data (2015-2024) to train models.

### Step 2: Get Upcoming Games (Weekly)

```python
from data.realtime_sources import NFLDataIntegration

# Initialize with your API key
integration = NFLDataIntegration(odds_api_key="your_key")

# Get this week's games
upcoming_games = integration.get_games_for_prediction()
```

### Step 3: Fetch Current Team Stats

You need recent team performance data:

```python
import nfl_data_py as nfl

# Get current season data
current_year = 2024
recent_stats = nfl.import_weekly_data([current_year])

# Get last 3 weeks for each team
# Calculate rolling averages, EPA, etc.
```

### Step 4: Engineer Features

Convert real-time data to model format:

```python
from data.features import FeatureEngineeer

engineer = FeatureEngineeer()

# You'll need to adapt this to work with current season data
# Calculate same features as training: rolling averages, EPA, etc.
features = engineer.create_matchup_features(upcoming_games, recent_stats)
```

### Step 5: Make Predictions

```python
from models.trainer import ModelTrainer

trainer = ModelTrainer()
trainer.load_model('ensemble')

predictions = trainer.models['ensemble'].predict(features)
probabilities = trainer.models['ensemble'].predict_proba(features)
```

## Complete Example

Here's a full workflow script:

```python
import os
from datetime import datetime
import nfl_data_py as nfl
from data.realtime_sources import NFLDataIntegration
from data.features import FeatureEngineeer
from models.trainer import ModelTrainer

# 1. Get API key
odds_api_key = os.getenv('ODDS_API_KEY')

# 2. Get upcoming games
integration = NFLDataIntegration(odds_api_key=odds_api_key)
upcoming = integration.get_games_for_prediction()

print(f"Found {len(upcoming)} upcoming games")

# 3. Get current season stats
current_year = datetime.now().year
recent_stats = nfl.import_weekly_data([current_year])

# 4. Engineer features
engineer = FeatureEngineeer()
# Note: You'll need to adapt feature engineering for real-time data
# This is a simplified example
features = engineer.create_prediction_features(upcoming, recent_stats)

# 5. Load model and predict
trainer = ModelTrainer()
trainer.load_model('ensemble')
model = trainer.models['ensemble']

predictions = model.predict(features)
probabilities = model.predict_proba(features)[:, 1]

# 6. Display predictions
for idx, game in upcoming.iterrows():
    pred = predictions[idx]
    conf = probabilities[idx]
    winner = game['home_team'] if pred == 1 else game['away_team']

    print(f"{game['away_team']} @ {game['home_team']}")
    print(f"  Prediction: {winner} ({conf:.1%} confidence)")
    print(f"  Spread: {game['home_spread']}")
    print()
```

## Comparison of Options

| Feature | NFLfastR | The Odds API | ESPN API | SportsDataIO |
|---------|----------|--------------|----------|--------------|
| **Historical Data** | ✅ Excellent | ❌ No | ❌ Limited | ✅ Yes |
| **Upcoming Games** | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| **Betting Lines** | ✅ Historical | ✅ Live | ⚠️ Basic | ✅ Live |
| **Free Tier** | ✅ Yes | ✅ Yes (500/mo) | ✅ Yes | ❌ No |
| **Official/Supported** | ✅ Yes | ✅ Yes | ❌ No | ✅ Yes |
| **Best For** | Training | Predictions | Schedules | Production |

## Recommended Approach

**For Personal Use:**
1. Use **NFLfastR** for historical data and model training
2. Use **The Odds API** (free tier) for upcoming games and live odds
3. Use **ESPN API** as backup for schedules

**For Production/Commercial:**
1. Use **NFLfastR** for historical data
2. Use **SportsDataIO** for real-time data, odds, and comprehensive stats
3. Consider **The Odds API** (paid tier) for multi-sportsbook odds comparison

## Next Steps

1. **Sign up for The Odds API**: https://the-odds-api.com/
2. **Update the prediction script** to fetch real-time data
3. **Create weekly automation** to pull current stats and make predictions
4. **Add injury data integration** for more accurate predictions
5. **Consider weather data** for outdoor games

## Important Notes

- NFLfastR updates within 15 minutes of game completion ✅
- The Odds API free tier gives 500 requests/month (about 2 per day) ⚠️
- ESPN API is unofficial - use at your own risk ⚠️
- Always respect API rate limits and terms of service ⚠️
- For high-frequency updates, consider a paid service 💰
