"""Integration with real-time data sources for upcoming games.

This module provides integrations with various APIs to get current/upcoming game data.
"""
import requests
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TheOddsAPIClient:
    """
    Client for The Odds API - provides upcoming games and betting lines.

    Free tier: 500 requests/month
    Signup: https://the-odds-api.com/
    """

    BASE_URL = "https://api.the-odds-api.com/v4"

    def __init__(self, api_key: str):
        """
        Initialize The Odds API client.

        Args:
            api_key: Your API key from the-odds-api.com
        """
        self.api_key = api_key
        self.sport = "americanfootball_nfl"

    def get_upcoming_games(self, days_ahead: int = 7) -> pd.DataFrame:
        """
        Get upcoming NFL games with odds.

        Args:
            days_ahead: Number of days ahead to fetch

        Returns:
            DataFrame with upcoming games and odds
        """
        url = f"{self.BASE_URL}/sports/{self.sport}/odds"

        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american',
            'dateFormat': 'iso'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            games = response.json()
            logger.info(f"Retrieved {len(games)} upcoming games")

            # Parse games into DataFrame
            parsed_games = []
            for game in games:
                game_data = {
                    'game_id': game['id'],
                    'commence_time': game['commence_time'],
                    'home_team': game['home_team'],
                    'away_team': game['away_team'],
                }

                # Extract odds from bookmakers
                if game.get('bookmakers'):
                    bookmaker = game['bookmakers'][0]  # Use first bookmaker

                    for market in bookmaker.get('markets', []):
                        if market['key'] == 'spreads':
                            # Find home team spread
                            for outcome in market['outcomes']:
                                if outcome['name'] == game['home_team']:
                                    game_data['home_spread'] = outcome['point']
                                    game_data['home_spread_odds'] = outcome['price']

                        elif market['key'] == 'totals':
                            game_data['total'] = market['outcomes'][0]['point']
                            game_data['over_odds'] = market['outcomes'][0]['price']

                        elif market['key'] == 'h2h':
                            for outcome in market['outcomes']:
                                if outcome['name'] == game['home_team']:
                                    game_data['home_moneyline'] = outcome['price']
                                else:
                                    game_data['away_moneyline'] = outcome['price']

                parsed_games.append(game_data)

            df = pd.DataFrame(parsed_games)

            # Check remaining requests
            remaining = response.headers.get('x-requests-remaining')
            logger.info(f"Requests remaining: {remaining}")

            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching odds: {e}")
            return pd.DataFrame()


class ESPNAPIClient:
    """
    Unofficial ESPN API client - free but no official support.

    No API key required, but rate limits apply.
    """

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

    def get_scoreboard(self, week: Optional[int] = None, season: Optional[int] = None) -> pd.DataFrame:
        """
        Get current week's scoreboard with game info.

        Args:
            week: Week number (None for current week)
            season: Season year (None for current season)

        Returns:
            DataFrame with games
        """
        url = f"{self.BASE_URL}/scoreboard"

        params = {}
        if week:
            params['week'] = week
        if season:
            params['seasontype'] = 2  # Regular season
            params['season'] = season

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()

            data = response.json()
            games = []

            for event in data.get('events', []):
                competition = event['competitions'][0]

                game_data = {
                    'game_id': event['id'],
                    'date': event['date'],
                    'name': event['name'],
                    'week': event.get('week', {}).get('number'),
                    'season': event.get('season', {}).get('year'),
                    'status': competition['status']['type']['name'],
                }

                # Get teams
                for team in competition['competitors']:
                    prefix = 'home' if team['homeAway'] == 'home' else 'away'
                    game_data[f'{prefix}_team'] = team['team']['abbreviation']
                    game_data[f'{prefix}_team_full'] = team['team']['displayName']
                    game_data[f'{prefix}_score'] = team.get('score')

                # Get odds if available
                if competition.get('odds'):
                    odds = competition['odds'][0]
                    game_data['spread'] = odds.get('details')
                    game_data['over_under'] = odds.get('overUnder')

                games.append(game_data)

            df = pd.DataFrame(games)
            logger.info(f"Retrieved {len(df)} games from ESPN")
            return df

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching from ESPN: {e}")
            return pd.DataFrame()

    def get_team_injuries(self, team_abbr: str) -> pd.DataFrame:
        """
        Get injury report for a team.

        Args:
            team_abbr: Team abbreviation (e.g., 'KC', 'BUF')

        Returns:
            DataFrame with injury information
        """
        # Note: ESPN's injury endpoint requires team ID, not abbreviation
        # You'd need to map abbreviations to ESPN team IDs
        url = f"{self.BASE_URL}/teams/{team_abbr}/injuries"

        try:
            response = requests.get(url)
            response.raise_for_status()
            return pd.DataFrame(response.json())
        except:
            logger.warning(f"Could not fetch injuries for {team_abbr}")
            return pd.DataFrame()


class NFLDataIntegration:
    """
    Combines NFLfastR historical data with real-time APIs.
    """

    def __init__(self, odds_api_key: Optional[str] = None):
        """
        Initialize data integration.

        Args:
            odds_api_key: API key for The Odds API (optional)
        """
        self.odds_client = TheOddsAPIClient(odds_api_key) if odds_api_key else None
        self.espn_client = ESPNAPIClient()

    def get_games_for_prediction(self, upcoming_only: bool = True) -> pd.DataFrame:
        """
        Get games that need predictions.

        Args:
            upcoming_only: If True, only return games that haven't been played

        Returns:
            DataFrame with games ready for prediction
        """
        games = pd.DataFrame()

        # Try The Odds API first (has best odds data)
        if self.odds_client:
            try:
                games = self.odds_client.get_upcoming_games()
                logger.info(f"Got {len(games)} games from The Odds API")
            except Exception as e:
                logger.warning(f"The Odds API failed: {e}")

        # Fallback to ESPN
        if games.empty:
            try:
                games = self.espn_client.get_scoreboard()
                if upcoming_only:
                    games = games[games['status'].isin(['STATUS_SCHEDULED', 'STATUS_POSTPONED'])]
                logger.info(f"Got {len(games)} games from ESPN")
            except Exception as e:
                logger.error(f"ESPN API failed: {e}")

        return games

    def prepare_for_model(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Convert real-time game data to format expected by model.

        Args:
            games: Raw game data from APIs

        Returns:
            DataFrame formatted for model prediction
        """
        # Map team abbreviations
        team_mapping = {
            'KC': 'KC', 'BUF': 'BUF', 'MIA': 'MIA', # etc.
            # Add full mapping as needed
        }

        # TODO: Fetch current season stats and create features
        # This would require getting recent game stats for each team
        # and calculating rolling averages, EPA, etc.

        logger.info("Note: Feature engineering from real-time data not yet implemented")
        logger.info("You'll need to fetch recent team stats and calculate features")

        return games


# Example usage
if __name__ == "__main__":
    import os

    # Get API key from environment variable
    api_key = os.getenv('ODDS_API_KEY')

    # Initialize integration
    integration = NFLDataIntegration(odds_api_key=api_key)

    # Get upcoming games
    upcoming = integration.get_games_for_prediction()

    if not upcoming.empty:
        print("\nUpcoming NFL Games:")
        print(upcoming[['home_team', 'away_team', 'home_spread', 'total']].to_string())
    else:
        print("No upcoming games found")
