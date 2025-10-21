"""NFL data collection using nfl_data_py (NFLfastR)."""
import nfl_data_py as nfl
import pandas as pd
from typing import List, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NFLDataCollector:
    """Collect NFL data from NFLfastR using nfl_data_py."""

    def __init__(self, cache_dir: Optional[Path] = None):
        """
        Initialize the data collector.

        Args:
            cache_dir: Directory to cache downloaded data
        """
        self.cache_dir = cache_dir

    def collect_pbp_data(self, years: List[int]) -> pd.DataFrame:
        """
        Collect play-by-play data for specified years.

        Args:
            years: List of years to collect data for

        Returns:
            DataFrame with play-by-play data
        """
        logger.info(f"Collecting play-by-play data for years: {years}")
        pbp = nfl.import_pbp_data(years)
        logger.info(f"Collected {len(pbp)} plays")
        return pbp

    def collect_schedule_data(self, years: List[int]) -> pd.DataFrame:
        """
        Collect game schedule and results data.

        Args:
            years: List of years to collect data for

        Returns:
            DataFrame with schedule data including scores
        """
        logger.info(f"Collecting schedule data for years: {years}")
        schedule = nfl.import_schedules(years)
        logger.info(f"Collected {len(schedule)} games")
        return schedule

    def collect_team_stats(self, years: List[int]) -> pd.DataFrame:
        """
        Collect weekly team statistics.

        Args:
            years: List of years to collect data for

        Returns:
            DataFrame with weekly team stats
        """
        logger.info(f"Collecting team stats for years: {years}")
        stats = nfl.import_weekly_data(years)
        logger.info(f"Collected {len(stats)} team-week records")
        return stats

    def collect_rosters(self, years: List[int]) -> pd.DataFrame:
        """
        Collect team rosters.

        Args:
            years: List of years to collect data for

        Returns:
            DataFrame with roster data
        """
        logger.info(f"Collecting roster data for years: {years}")
        rosters = nfl.import_rosters(years)
        logger.info(f"Collected {len(rosters)} player records")
        return rosters

    def collect_injury_data(self, years: List[int]) -> pd.DataFrame:
        """
        Collect injury report data.

        Args:
            years: List of years to collect data for

        Returns:
            DataFrame with injury data
        """
        logger.info(f"Collecting injury data for years: {years}")
        try:
            injuries = nfl.import_injuries(years)
            logger.info(f"Collected {len(injuries)} injury records")
            return injuries
        except Exception as e:
            logger.warning(f"Could not collect injury data: {e}")
            return pd.DataFrame()

    def collect_all_data(self, start_year: int, end_year: int) -> dict:
        """
        Collect all available data types for a range of years.

        Args:
            start_year: First year to collect
            end_year: Last year to collect

        Returns:
            Dictionary with all data types
        """
        years = list(range(start_year, end_year + 1))

        data = {
            'pbp': self.collect_pbp_data(years),
            'schedule': self.collect_schedule_data(years),
            'team_stats': self.collect_team_stats(years),
            'rosters': self.collect_rosters(years),
        }

        # Injury data might not be available for all years
        injuries = self.collect_injury_data(years)
        if not injuries.empty:
            data['injuries'] = injuries

        return data

    def save_data(self, data: dict, output_dir: Path):
        """
        Save collected data to parquet files.

        Args:
            data: Dictionary of DataFrames
            output_dir: Directory to save files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in data.items():
            output_path = output_dir / f"{name}.parquet"
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {name} to {output_path}")


if __name__ == "__main__":
    # Example usage
    collector = NFLDataCollector()
    data = collector.collect_all_data(start_year=2020, end_year=2023)

    from config.config import RAW_DATA_DIR
    collector.save_data(data, RAW_DATA_DIR)
