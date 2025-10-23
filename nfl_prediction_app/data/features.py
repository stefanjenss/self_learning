"""Feature engineering for NFL game predictions."""
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineeer:
    """Create features for NFL game prediction."""

    def __init__(self):
        self.team_features = {}

    def calculate_rolling_stats(
        self,
        df: pd.DataFrame,
        group_col: str,
        stat_cols: List[str],
        windows: List[int] = [3, 5, 10]
    ) -> pd.DataFrame:
        """
        Calculate rolling averages for team statistics.

        Args:
            df: DataFrame with team statistics
            group_col: Column to group by (usually 'team')
            stat_cols: Columns to calculate rolling stats for
            windows: Window sizes for rolling averages

        Returns:
            DataFrame with rolling statistics
        """
        df = df.copy()
        df = df.sort_values(['season', 'week'])

        for window in windows:
            for col in stat_cols:
                roll_col = f"{col}_roll_{window}"
                df[roll_col] = df.groupby(group_col)[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean()
                )

        return df

    def calculate_team_efficiency_stats(self, pbp_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate advanced efficiency statistics from play-by-play data.

        Args:
            pbp_df: Play-by-play DataFrame

        Returns:
            DataFrame with efficiency stats by team and game
        """
        # Filter to regular plays
        plays = pbp_df[
            (pbp_df['play_type'].isin(['run', 'pass'])) &
            (pbp_df['two_point_attempt'] == 0)
        ].copy()

        # Calculate EPA (Expected Points Added) stats
        team_stats = []

        for game_id in plays['game_id'].unique():
            game_plays = plays[plays['game_id'] == game_id]

            for team_type in ['posteam', 'defteam']:
                teams = game_plays[team_type].unique()

                for team in teams:
                    if team_type == 'posteam':
                        team_plays = game_plays[game_plays['posteam'] == team]
                        epa_col = 'epa'
                    else:
                        team_plays = game_plays[game_plays['defteam'] == team]
                        epa_col = 'epa'

                    if len(team_plays) > 0:
                        stats = {
                            'game_id': game_id,
                            'team': team,
                            'type': team_type,
                            'epa_per_play': team_plays[epa_col].mean(),
                            'success_rate': team_plays['success'].mean(),
                            'pass_epa': team_plays[team_plays['play_type'] == 'pass'][epa_col].mean(),
                            'rush_epa': team_plays[team_plays['play_type'] == 'run'][epa_col].mean(),
                            'explosive_play_rate': (team_plays[epa_col] > 1.0).mean(),
                        }
                        team_stats.append(stats)

        return pd.DataFrame(team_stats)

    def create_matchup_features(
        self,
        schedule: pd.DataFrame,
        team_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create features for game matchups.

        Args:
            schedule: Schedule DataFrame with home/away teams
            team_stats: Team statistics DataFrame

        Returns:
            DataFrame with matchup features
        """
        matchups = []

        for _, game in schedule.iterrows():
            game_id = game['game_id']
            home_team = game['home_team']
            away_team = game['away_team']
            week = game['week']
            season = game['season']

            # Get stats for both teams up to this game
            home_stats = team_stats[
                (team_stats['team'] == home_team) &
                (team_stats['season'] == season) &
                (team_stats['week'] < week)
            ]

            away_stats = team_stats[
                (team_stats['team'] == away_team) &
                (team_stats['season'] == season) &
                (team_stats['week'] < week)
            ]

            if len(home_stats) > 0 and len(away_stats) > 0:
                # Get most recent stats
                home_recent = home_stats.iloc[-1]
                away_recent = away_stats.iloc[-1]

                matchup = {
                    'game_id': game_id,
                    'season': season,
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': game.get('home_score'),
                    'away_score': game.get('away_score'),
                    'home_spread': game.get('spread_line'),
                    'total': game.get('total_line'),
                }

                # Add differential features
                for col in home_recent.index:
                    if col not in ['team', 'season', 'week', 'game_id']:
                        try:
                            matchup[f'home_{col}'] = home_recent[col]
                            matchup[f'away_{col}'] = away_recent[col]
                            matchup[f'diff_{col}'] = home_recent[col] - away_recent[col]
                        except:
                            pass

                matchups.append(matchup)

        return pd.DataFrame(matchups)

    def add_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add target variables for prediction.

        Args:
            df: DataFrame with game data

        Returns:
            DataFrame with target variables
        """
        df = df.copy()

        # Home team won
        df['home_win'] = (df['home_score'] > df['away_score']).astype(int)

        # Point differential
        df['point_diff'] = df['home_score'] - df['away_score']

        # Beat the spread (if spread available)
        if 'home_spread' in df.columns:
            df['beat_spread'] = (df['point_diff'] + df['home_spread'] > 0).astype(int)

        # Over/under (if total available)
        if 'total' in df.columns:
            df['total_points'] = df['home_score'] + df['away_score']
            df['over'] = (df['total_points'] > df['total']).astype(int)

        return df

    def engineer_all_features(
        self,
        schedule: pd.DataFrame,
        pbp: pd.DataFrame,
        weekly_stats: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.

        Args:
            schedule: Game schedule
            pbp: Play-by-play data
            weekly_stats: Weekly team statistics

        Returns:
            DataFrame ready for modeling
        """
        logger.info("Starting feature engineering...")

        # Calculate efficiency stats from play-by-play
        logger.info("Calculating efficiency stats...")
        efficiency_stats = self.calculate_team_efficiency_stats(pbp)

        # Pivot efficiency stats for offensive and defensive
        offense_stats = efficiency_stats[efficiency_stats['type'] == 'posteam'].copy()
        defense_stats = efficiency_stats[efficiency_stats['type'] == 'defteam'].copy()

        # Merge with weekly stats
        offense_stats = offense_stats.rename(columns={
            'epa_per_play': 'off_epa_per_play',
            'success_rate': 'off_success_rate',
            'pass_epa': 'off_pass_epa',
            'rush_epa': 'off_rush_epa',
        })

        defense_stats = defense_stats.rename(columns={
            'epa_per_play': 'def_epa_per_play',
            'success_rate': 'def_success_rate',
            'pass_epa': 'def_pass_epa',
            'rush_epa': 'def_rush_epa',
        })

        # Create matchup features
        logger.info("Creating matchup features...")
        matchups = self.create_matchup_features(schedule, weekly_stats)

        # Add target variables
        logger.info("Adding target variables...")
        matchups = self.add_target_variables(matchups)

        logger.info(f"Feature engineering complete. Created {len(matchups)} matchups with {len(matchups.columns)} features")

        return matchups


if __name__ == "__main__":
    # Example usage
    from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

    # Load raw data
    schedule = pd.read_parquet(RAW_DATA_DIR / "schedule.parquet")
    pbp = pd.read_parquet(RAW_DATA_DIR / "pbp.parquet")
    team_stats = pd.read_parquet(RAW_DATA_DIR / "team_stats.parquet")

    # Engineer features
    engineer = FeatureEngineeer()
    features = engineer.engineer_all_features(schedule, pbp, team_stats)

    # Save processed features
    features.to_parquet(PROCESSED_DATA_DIR / "game_features.parquet", index=False)
    logger.info("Features saved successfully")
