# Makefile for NFL Prediction App

.PHONY: help install setup data features train backtest web test clean

help:
	@echo "NFL Prediction App - Available Commands:"
	@echo ""
	@echo "  make install     - Install Python dependencies"
	@echo "  make setup       - Create virtual environment and install dependencies"
	@echo "  make pipeline    - Run complete data pipeline (collect, features, train)"
	@echo "  make data        - Collect NFL data from NFLfastR"
	@echo "  make features    - Engineer features from raw data"
	@echo "  make train       - Train prediction models"
	@echo "  make backtest    - Run betting strategy backtests"
	@echo "  make web         - Start web application"
	@echo "  make predict     - Make predictions for recent games"
	@echo "  make test        - Run unit tests"
	@echo "  make clean       - Remove generated files"

install:
	pip install -r requirements.txt

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

pipeline:
	python scripts/run_pipeline.py

data:
	python data/collectors.py

features:
	python data/features.py

train:
	python models/trainer.py

backtest:
	python scripts/run_backtest.py

web:
	python web_app/app.py

predict:
	python scripts/predict.py

test:
	pytest tests/ -v

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -f backtest_results.csv
