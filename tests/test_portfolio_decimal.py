"""Tests for Decimal precision in portfolio cost_basis and realized gains."""

import pytest

from src.backtesting.portfolio import Portfolio


class TestCostBasisPrecision:
    def test_many_small_trades_no_floating_point_drift(self):
        """Buy 1 share at $33.33 ten times. Cost basis should be exactly 33.33."""
        p = Portfolio(tickers=["AAPL"], initial_cash=1_000_000.0, margin_requirement=0.0)
        for _ in range(10):
            p.apply_long_buy("AAPL", quantity=1, price=33.33)

        snap = p.get_snapshot()
        assert snap["positions"]["AAPL"]["long"] == 10
        assert snap["positions"]["AAPL"]["long_cost_basis"] == pytest.approx(33.33, abs=1e-10)


class TestRealizedGainsPrecision:
    def test_realized_gain_small_price_difference(self):
        """Buy 3 shares at $33.33, sell 3 at $33.34.

        Expected realized gain = 3 * (33.34 - 33.33) = 3 * 0.01 = 0.03.
        """
        p = Portfolio(tickers=["AAPL"], initial_cash=1_000_000.0, margin_requirement=0.0)
        p.apply_long_buy("AAPL", quantity=3, price=33.33)
        p.apply_long_sell("AAPL", quantity=3, price=33.34)

        snap = p.get_snapshot()
        assert snap["realized_gains"]["AAPL"]["long"] == pytest.approx(0.03, abs=1e-10)
        assert snap["positions"]["AAPL"]["long"] == 0
        assert snap["positions"]["AAPL"]["long_cost_basis"] == 0.0
