"""Configuration settings for NFL prediction app."""
import os
from pathlib import Path
from typing import List

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directories
MODEL_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODEL_DIR / "saved_models"

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, SAVED_MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# NFL Data settings
NFL_START_YEAR = 2015
NFL_CURRENT_YEAR = 2024

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Features to use for prediction
TEAM_STATS_FEATURES = [
    'points_scored_avg',
    'points_allowed_avg',
    'yards_gained_avg',
    'yards_allowed_avg',
    'turnovers_avg',
    'third_down_pct',
    'red_zone_pct',
    'sack_rate',
    'pressure_rate',
]

# Betting settings
STARTING_BANKROLL = 10000
KELLY_FRACTION = 0.25  # Fractional Kelly (conservative)
MIN_EDGE = 0.02  # Minimum 2% edge to place bet

# Web app settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True
