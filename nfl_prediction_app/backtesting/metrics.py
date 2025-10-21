"""Performance metrics for betting strategies."""
import pandas as pd
import numpy as np
from typing import Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BettingMetrics:
    """Calculate performance metrics for betting strategies."""

    @staticmethod
    def calculate_all_metrics(bet_history: pd.DataFrame, initial_bankroll: float) -> Dict:
        """
        Calculate comprehensive betting metrics.

        Args:
            bet_history: DataFrame with bet history
            initial_bankroll: Starting bankroll

        Returns:
            Dictionary of metrics
        """
        if len(bet_history) == 0:
            return {
                'total_bets': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0,
                'final_bankroll': initial_bankroll,
            }

        metrics = {
            'total_bets': len(bet_history),
            'winning_bets': bet_history['won'].sum(),
            'losing_bets': (~bet_history['won']).sum(),
            'win_rate': bet_history['won'].mean(),
            'total_wagered': bet_history['bet_size'].sum(),
            'total_profit': bet_history['profit'].sum(),
            'avg_bet_size': bet_history['bet_size'].mean(),
            'avg_profit_per_bet': bet_history['profit'].mean(),
            'final_bankroll': bet_history['bankroll'].iloc[-1],
        }

        # ROI
        if metrics['total_wagered'] > 0:
            metrics['roi'] = metrics['total_profit'] / metrics['total_wagered']
        else:
            metrics['roi'] = 0

        # ROI on initial bankroll
        metrics['return_on_investment'] = (
            metrics['final_bankroll'] - initial_bankroll
        ) / initial_bankroll

        # Maximum drawdown
        metrics['max_drawdown'] = BettingMetrics.calculate_max_drawdown(
            bet_history['bankroll'].values
        )

        # Sharpe ratio
        if len(bet_history) > 1:
            returns = bet_history['profit'] / initial_bankroll
            metrics['sharpe_ratio'] = BettingMetrics.calculate_sharpe_ratio(returns)
        else:
            metrics['sharpe_ratio'] = 0

        # Win/loss streaks
        metrics['max_win_streak'] = BettingMetrics.calculate_max_streak(
            bet_history['won'].values, True
        )
        metrics['max_loss_streak'] = BettingMetrics.calculate_max_streak(
            bet_history['won'].values, False
        )

        # Average winning/losing bet
        winning_bets = bet_history[bet_history['won']]
        losing_bets = bet_history[~bet_history['won']]

        if len(winning_bets) > 0:
            metrics['avg_winning_profit'] = winning_bets['profit'].mean()
        else:
            metrics['avg_winning_profit'] = 0

        if len(losing_bets) > 0:
            metrics['avg_losing_profit'] = losing_bets['profit'].mean()
        else:
            metrics['avg_losing_profit'] = 0

        # Profit factor (gross wins / gross losses)
        total_wins = winning_bets['profit'].sum() if len(winning_bets) > 0 else 0
        total_losses = abs(losing_bets['profit'].sum()) if len(losing_bets) > 0 else 0

        if total_losses > 0:
            metrics['profit_factor'] = total_wins / total_losses
        else:
            metrics['profit_factor'] = float('inf') if total_wins > 0 else 0

        # Bet size statistics
        metrics['max_bet_size'] = bet_history['bet_size'].max()
        metrics['min_bet_size'] = bet_history['bet_size'].min()
        metrics['std_bet_size'] = bet_history['bet_size'].std()

        # Kelly criterion evaluation (risk of ruin estimate)
        metrics['risk_of_ruin'] = BettingMetrics.estimate_risk_of_ruin(
            metrics['win_rate'],
            metrics['avg_winning_profit'],
            abs(metrics['avg_losing_profit']),
            initial_bankroll
        )

        return metrics

    @staticmethod
    def calculate_max_drawdown(bankroll_history: np.ndarray) -> float:
        """
        Calculate maximum drawdown.

        Args:
            bankroll_history: Array of bankroll values over time

        Returns:
            Maximum drawdown as a percentage
        """
        if len(bankroll_history) == 0:
            return 0

        # Calculate running maximum
        running_max = np.maximum.accumulate(bankroll_history)

        # Calculate drawdown at each point
        drawdown = (bankroll_history - running_max) / running_max

        # Return maximum drawdown
        return abs(drawdown.min())

    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate

        Returns:
            Sharpe ratio
        """
        if len(returns) <= 1:
            return 0

        # Annualize returns (assuming ~16 bets per week, 17 weeks)
        periods_per_year = 16 * 17

        excess_returns = returns - (risk_free_rate / periods_per_year)
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()

    @staticmethod
    def calculate_max_streak(results: np.ndarray, target_value: bool) -> int:
        """
        Calculate maximum streak of a specific result.

        Args:
            results: Array of True/False results
            target_value: Value to calculate streak for

        Returns:
            Maximum streak length
        """
        if len(results) == 0:
            return 0

        max_streak = 0
        current_streak = 0

        for result in results:
            if result == target_value:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

        return max_streak

    @staticmethod
    def estimate_risk_of_ruin(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        bankroll: float,
        max_bet: float = None
    ) -> float:
        """
        Estimate risk of ruin using simplified formula.

        Args:
            win_rate: Probability of winning
            avg_win: Average winning amount
            avg_loss: Average losing amount (positive number)
            bankroll: Current bankroll
            max_bet: Maximum bet size (if None, uses avg_loss)

        Returns:
            Estimated probability of ruin
        """
        if win_rate >= 1 or win_rate <= 0:
            return 0 if win_rate >= 1 else 1

        if max_bet is None:
            max_bet = avg_loss

        if max_bet <= 0 or bankroll <= 0:
            return 1

        # Simplified risk of ruin formula
        # ROR = ((1-p)/p)^(bankroll/avg_bet)
        p = win_rate
        q = 1 - p

        if avg_win <= 0:
            return 1

        # Calculate win/loss ratio
        win_loss_ratio = avg_win / avg_loss

        # Units of bankroll
        units = bankroll / max_bet

        # Risk of ruin approximation
        if win_loss_ratio * p >= q:
            # Positive expectation
            ror = (q / (p * win_loss_ratio)) ** units
        else:
            # Negative expectation
            ror = 1.0

        return min(ror, 1.0)

    @staticmethod
    def print_metrics_report(metrics: Dict):
        """
        Print formatted metrics report.

        Args:
            metrics: Dictionary of metrics
        """
        print("\n" + "=" * 60)
        print("BETTING STRATEGY PERFORMANCE REPORT")
        print("=" * 60)

        print(f"\nBasic Statistics:")
        print(f"  Total Bets:        {metrics['total_bets']}")
        print(f"  Winning Bets:      {metrics['winning_bets']}")
        print(f"  Losing Bets:       {metrics['losing_bets']}")
        print(f"  Win Rate:          {metrics['win_rate']:.2%}")

        print(f"\nFinancial Performance:")
        print(f"  Total Wagered:     ${metrics['total_wagered']:,.2f}")
        print(f"  Total Profit:      ${metrics['total_profit']:,.2f}")
        print(f"  ROI:               {metrics['roi']:.2%}")
        print(f"  Return on Invest:  {metrics['return_on_investment']:.2%}")
        print(f"  Final Bankroll:    ${metrics['final_bankroll']:,.2f}")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown:      {metrics['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio:      {metrics['sharpe_ratio']:.2f}")
        print(f"  Risk of Ruin:      {metrics['risk_of_ruin']:.2%}")
        print(f"  Profit Factor:     {metrics['profit_factor']:.2f}")

        print(f"\nBet Sizing:")
        print(f"  Avg Bet Size:      ${metrics['avg_bet_size']:,.2f}")
        print(f"  Max Bet Size:      ${metrics['max_bet_size']:,.2f}")
        print(f"  Min Bet Size:      ${metrics['min_bet_size']:,.2f}")

        print(f"\nStreaks:")
        print(f"  Max Win Streak:    {metrics['max_win_streak']}")
        print(f"  Max Loss Streak:   {metrics['max_loss_streak']}")

        print("\n" + "=" * 60)
