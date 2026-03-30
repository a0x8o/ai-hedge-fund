"""Tests for backtest engine fixes: first-month skip and Sortino ratio capping."""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import BacktestEngine
from src.backtesting.metrics import PerformanceMetricsCalculator


# ---------------------------------------------------------------------------
# Helper to build PortfolioValuePoints (same pattern as test_metrics.py)
# ---------------------------------------------------------------------------

def _build_values(values: list[float]):
    start = datetime(2024, 1, 1)
    points = []
    for i, v in enumerate(values):
        points.append({
            "Date": start + timedelta(days=i),
            "Portfolio Value": v,
            "Long Exposure": 0.0,
            "Short Exposure": 0.0,
            "Gross Exposure": 0.0,
            "Net Exposure": 0.0,
            "Long/Short Ratio": None,
        })
    return points


# ---------------------------------------------------------------------------
# First-month skip fix
# ---------------------------------------------------------------------------

class TestFirstMonthSkipFix:
    """Verify the engine processes all business days, not skipping the first month."""

    @patch("src.backtesting.engine.get_prices")
    @patch("src.backtesting.engine.get_financial_metrics")
    @patch("src.backtesting.engine.get_insider_trades")
    @patch("src.backtesting.engine.get_company_news")
    @patch("src.backtesting.engine.get_price_data")
    def test_all_dates_processed(
        self,
        mock_price_data,
        mock_news,
        mock_insider,
        mock_metrics,
        mock_prices,
    ):
        # Use a short 5 business-day range
        start = "2024-01-01"
        end = "2024-01-05"
        dates = pd.date_range(start, end, freq="B")

        # Mock price_data to return a valid DataFrame for each call
        price_df = pd.DataFrame({"close": [150.0]})
        mock_price_data.return_value = price_df

        # Create a mock agent
        mock_agent = MagicMock()

        engine = BacktestEngine(
            agent=mock_agent,
            tickers=["AAPL"],
            start_date=start,
            end_date=end,
            initial_capital=100_000.0,
            model_name="test-model",
            model_provider="test-provider",
            selected_analysts=None,
            initial_margin_requirement=0.0,
        )

        # Mock the agent controller to return a simple hold decision
        engine._agent_controller.run_agent = MagicMock(return_value={
            "decisions": {"AAPL": {"action": "hold", "quantity": 0}},
            "analyst_signals": {},
        })

        # Mock benchmark
        engine._benchmark.get_return_pct = MagicMock(return_value=0.0)

        # Mock output builder to avoid printing
        engine._results.build_day_rows = MagicMock(return_value=[])
        engine._results.print_rows = MagicMock()

        engine.run_backtest()

        # The engine should have called run_agent for every business day
        # (not skipping the first month). With the lookback_start < current_date
        # guard, all dates after the first one where lookback is valid should run.
        # For dates in early January 2024, lookback_start = Dec 2023 < Jan 2024,
        # so ALL dates should be processed.
        assert engine._agent_controller.run_agent.call_count == len(dates)


# ---------------------------------------------------------------------------
# Sortino ratio capping
# ---------------------------------------------------------------------------

class TestSortinoCapping:
    def test_sortino_capped_for_constant_positive_returns(self):
        """When all returns are positive (no downside), sortino should be 999.99."""
        calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
        # Monotonically increasing values: all returns positive, zero downside deviation
        values = _build_values([100.0, 101.0, 102.0, 103.0, 104.0])
        result = calc.compute_metrics(values)
        assert result["sortino_ratio"] == 999.99

    def test_sortino_normal_case(self):
        """With mixed returns, sortino should be a finite non-capped value."""
        calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
        # Mix of up and down to produce non-zero downside deviation
        values = _build_values([100.0, 105.0, 98.0, 102.0, 97.0, 103.0])
        result = calc.compute_metrics(values)
        sortino = result["sortino_ratio"]
        assert sortino is not None
        assert sortino != 999.99
        assert np.isfinite(sortino)

    def test_sortino_zero_when_negative_mean_no_downside(self):
        """When mean excess return is zero and no downside, sortino should be 0."""
        calc = PerformanceMetricsCalculator(annual_trading_days=252, annual_rf_rate=0.0)
        # Constant values: zero returns, zero downside
        values = _build_values([100.0, 100.0, 100.0, 100.0])
        result = calc.compute_metrics(values)
        assert result["sortino_ratio"] == 0.0
